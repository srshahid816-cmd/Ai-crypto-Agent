"""
Crypto Multi-Factor Market Scanner - v1 (Streamlit Cloud build)

Same backend logic as the Flask version (indicators, structure, regime,
signal engine, universe discovery) - only the UI layer changed, because
Streamlit Cloud runs Streamlit apps, not raw Flask/WSGI apps.

No fabricated data: every field traces to Binance's public REST API, or
shows DATA UNAVAILABLE / NOT_ENOUGH_DATA.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

import requests
import numpy as np
import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "binance_base_url": "https://api.binance.com",
    "request_timeout_seconds": 10,
    "max_retries": 3,
    "backoff_base_seconds": 1.5,

    "quote_asset": "USDT",
    "min_24h_quote_volume_usdt": 5_000_000,
    "min_price_usdt": 0.0000001,
    "exclude_stablecoins": True,
    "stablecoin_symbols": {"USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "DAIUSDT", "BUSDUSDT", "USDPUSDT"},
    "exclude_leveraged_tokens": True,
    "leveraged_token_suffixes": ("UP", "DOWN", "BULL", "BEAR"),
    "top_n_gainers": 10,
    "top_n_losers": 10,

    "primary_timeframe": "1h",
    "regime_timeframe_high": "4h",
    "klines_limit": 200,

    "signal_weights": {
        "btc_regime_alignment": 20,
        "structure_state": 25,
        "indicator_confluence": 20,
        "volume_confirmation": 15,
        "relative_strength_vs_btc": 10,
    },
    "min_score_buy_sell": 65,
    "min_data_quality_for_trade": 60,
    "scan_cache_ttl_seconds": 55,
}

# ============================================================
# DATA LAYER
# ============================================================

class MarketDataError(Exception):
    pass


KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "num_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]


class BinanceMarketData:
    def __init__(self):
        self.session = requests.Session()

    def _get(self, path: str, params: dict | None = None):
        url = f"{CONFIG['binance_base_url']}{path}"
        last_err = None
        for attempt in range(1, CONFIG["max_retries"] + 1):
            try:
                resp = self.session.get(url, params=params, timeout=CONFIG["request_timeout_seconds"])
                if resp.status_code in (429, 418):
                    time.sleep(CONFIG["backoff_base_seconds"] * (2 ** attempt))
                    last_err = MarketDataError(f"Rate limited ({resp.status_code}) on {path}")
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                last_err = e
                time.sleep(CONFIG["backoff_base_seconds"] * attempt)
        raise MarketDataError(f"Failed to fetch {path}: {last_err}")

    def get_exchange_info(self):
        return self._get("/api/v3/exchangeInfo")

    def get_ticker_24hr(self):
        return self._get("/api/v3/ticker/24hr")

    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
        raw = self._get("/api/v3/klines", params={"symbol": symbol, "interval": interval, "limit": limit})
        if not raw:
            raise MarketDataError(f"Empty klines for {symbol} {interval}")
        df = pd.DataFrame(raw, columns=KLINE_COLUMNS)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df


# ============================================================
# INDICATORS
# ============================================================

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result.where(avg_loss != 0, 100.0)


def macd(closes: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(closes, fast) - ema(closes, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


def atr(highs, lows, closes, period=14) -> pd.Series:
    prev_close = closes.shift(1)
    tr = pd.concat([highs - lows, (highs - prev_close).abs(), (lows - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def last_valid(series: pd.Series):
    if series is None or len(series) == 0 or pd.isna(series.iloc[-1]):
        return None, False
    return float(series.iloc[-1]), True


# ============================================================
# STRUCTURE
# ============================================================

class StructureState(Enum):
    BULLISH_STRUCTURE = "BULLISH_STRUCTURE"
    BEARISH_STRUCTURE = "BEARISH_STRUCTURE"
    RANGE = "RANGE"
    TRANSITION = "TRANSITION"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class SwingPoint:
    index: int
    price: float
    kind: str
    label: Optional[str] = None


def find_swing_points(highs: pd.Series, lows: pd.Series, lookback: int = 2) -> List[SwingPoint]:
    points: List[SwingPoint] = []
    n = len(highs)
    if n < (2 * lookback + 1):
        return points
    for i in range(lookback, n - lookback):
        window_h = highs.iloc[i - lookback: i + lookback + 1]
        window_l = lows.iloc[i - lookback: i + lookback + 1]
        prior_h = highs.iloc[i - lookback:i]
        prior_h_max = prior_h.max() if len(prior_h) else float("-inf")
        prior_l = lows.iloc[i - lookback:i]
        prior_l_min = prior_l.min() if len(prior_l) else float("inf")
        if highs.iloc[i] == window_h.max() and highs.iloc[i] != prior_h_max:
            points.append(SwingPoint(i, float(highs.iloc[i]), "HIGH"))
        if lows.iloc[i] == window_l.min() and lows.iloc[i] != prior_l_min:
            points.append(SwingPoint(i, float(lows.iloc[i]), "LOW"))
    points.sort(key=lambda p: p.index)
    return points


def label_swings(points: List[SwingPoint]) -> List[SwingPoint]:
    last_high = last_low = None
    for p in points:
        if p.kind == "HIGH":
            if last_high is not None:
                p.label = "HH" if p.price > last_high.price else "LH"
            last_high = p
        else:
            if last_low is not None:
                p.label = "HL" if p.price > last_low.price else "LL"
            last_low = p
    return points


def detect_structure(highs: pd.Series, lows: pd.Series, lookback: int = 2) -> dict:
    raw_points = find_swing_points(highs, lows, lookback)
    if len(raw_points) < 4:
        return {"state": StructureState.UNCERTAIN, "reason": "NOT_ENOUGH_DATA"}
    points = label_swings(raw_points)
    labels = [p.label for p in points if p.label is not None]
    recent = labels[-4:] if len(labels) >= 4 else labels
    bull = sum(1 for l in recent if l in ("HH", "HL"))
    bear = sum(1 for l in recent if l in ("LH", "LL"))
    if bull >= 3:
        state = StructureState.BULLISH_STRUCTURE
    elif bear >= 3:
        state = StructureState.BEARISH_STRUCTURE
    elif bull == bear:
        state = StructureState.RANGE
    else:
        state = StructureState.TRANSITION
    return {"state": state, "reason": f"last swings: {recent}", "points": points}


def nearest_support_resistance(highs, lows, current_price, lookback=2):
    points = label_swings(find_swing_points(highs, lows, lookback))
    highs_above = [p.price for p in points if p.kind == "HIGH" and p.price > current_price]
    lows_below = [p.price for p in points if p.kind == "LOW" and p.price < current_price]
    return (max(lows_below) if lows_below else None,
            min(highs_above) if highs_above else None)


# ============================================================
# MARKET REGIME
# ============================================================

class Regime(Enum):
    STRONG_BULLISH = "STRONG_BULLISH"
    BULLISH = "BULLISH"
    NEUTRAL = "NEUTRAL"
    BEARISH = "BEARISH"
    STRONG_BEARISH = "STRONG_BEARISH"
    UNCERTAIN = "UNCERTAIN"


def classify_regime(klines_1h: pd.DataFrame, klines_4h: pd.DataFrame) -> dict:
    if klines_1h is None or klines_4h is None or len(klines_1h) < 50 or len(klines_4h) < 50:
        return {"regime": Regime.UNCERTAIN, "reason": "NOT_ENOUGH_DATA"}

    closes_1h = klines_1h["close"]
    ema50_1h, ok50 = last_valid(ema(closes_1h, 50))
    ema200_1h, ok200 = last_valid(ema(closes_1h, 200))
    price, _ = last_valid(closes_1h)
    if not (ok50 and price is not None):
        return {"regime": Regime.UNCERTAIN, "reason": "NOT_ENOUGH_DATA for EMA"}

    struct_4h = detect_structure(klines_4h["high"], klines_4h["low"])
    struct_1h = detect_structure(klines_1h["high"], klines_1h["low"])

    score = 0
    reasons = []
    if price > ema50_1h:
        score += 1; reasons.append("price>EMA50")
    else:
        score -= 1; reasons.append("price<EMA50")
    if ok200:
        if price > ema200_1h:
            score += 1; reasons.append("price>EMA200")
        else:
            score -= 1; reasons.append("price<EMA200")
    if struct_4h["state"] == StructureState.BULLISH_STRUCTURE:
        score += 2; reasons.append("4h bullish")
    elif struct_4h["state"] == StructureState.BEARISH_STRUCTURE:
        score -= 2; reasons.append("4h bearish")
    if struct_1h["state"] == StructureState.BULLISH_STRUCTURE:
        score += 1; reasons.append("1h bullish")
    elif struct_1h["state"] == StructureState.BEARISH_STRUCTURE:
        score -= 1; reasons.append("1h bearish")

    if score >= 4:
        regime = Regime.STRONG_BULLISH
    elif score >= 2:
        regime = Regime.BULLISH
    elif score <= -4:
        regime = Regime.STRONG_BEARISH
    elif score <= -2:
        regime = Regime.BEARISH
    else:
        regime = Regime.NEUTRAL
    return {"regime": regime, "reason": "; ".join(reasons)}


def relative_strength_vs_btc(sym_klines, btc_klines, lookback=24):
    if sym_klines is None or btc_klines is None or len(sym_klines) < lookback + 1 or len(btc_klines) < lookback + 1:
        return None
    sym_chg = (sym_klines["close"].iloc[-1] / sym_klines["close"].iloc[-lookback] - 1) * 100
    btc_chg = (btc_klines["close"].iloc[-1] / btc_klines["close"].iloc[-lookback] - 1) * 100
    return float(sym_chg - btc_chg)


# ============================================================
# SIGNAL ENGINE
# ============================================================

class Decision(Enum):
    BUY_SETUP = "BUY SETUP"
    SELL_SETUP = "SELL SETUP"
    HOLD_WAIT = "HOLD / WAIT"
    NO_TRADE = "NO TRADE"


@dataclass
class FactorScore:
    name: str
    value: Optional[str]
    score: float
    data_quality: str


def score_symbol(symbol: str, klines: pd.DataFrame, btc_regime: Regime, rel_strength: Optional[float]) -> dict:
    struct = detect_structure(klines["high"], klines["low"])
    state = struct["state"]
    factors = []

    bullish_regimes = {Regime.STRONG_BULLISH, Regime.BULLISH}
    bearish_regimes = {Regime.STRONG_BEARISH, Regime.BEARISH}
    if btc_regime == Regime.UNCERTAIN:
        factors.append(FactorScore("btc_regime_alignment", btc_regime.value, 0.0, "UNAVAILABLE"))
    else:
        if btc_regime in bullish_regimes and state == StructureState.BULLISH_STRUCTURE:
            s = 1.0 if btc_regime == Regime.STRONG_BULLISH else 0.6
        elif btc_regime in bearish_regimes and state == StructureState.BEARISH_STRUCTURE:
            s = -1.0 if btc_regime == Regime.STRONG_BEARISH else -0.6
        else:
            s = 0.0
        factors.append(FactorScore("btc_regime_alignment", f"{btc_regime.value} vs {state.value}", s, "GOOD"))

    struct_map = {StructureState.BULLISH_STRUCTURE: 1.0, StructureState.BEARISH_STRUCTURE: -1.0,
                  StructureState.RANGE: 0.0, StructureState.TRANSITION: 0.0}
    if state == StructureState.UNCERTAIN:
        factors.append(FactorScore("structure_state", state.value, 0.0, "PARTIAL"))
    else:
        factors.append(FactorScore("structure_state", state.value, struct_map[state], "GOOD"))

    closes = klines["close"]
    if len(closes) < 30:
        factors.append(FactorScore("indicator_confluence", None, 0.0, "UNAVAILABLE"))
    else:
        rsi_val, rsi_ok = last_valid(rsi(closes))
        macd_line, signal_line = macd(closes)
        macd_val, macd_ok = last_valid(macd_line)
        sig_val, sig_ok = last_valid(signal_line)
        ema20_val, e20_ok = last_valid(ema(closes, 20))
        ema50_val, e50_ok = last_valid(ema(closes, 50))
        price, price_ok = last_valid(closes)
        if not all([rsi_ok, macd_ok, sig_ok, e20_ok, e50_ok, price_ok]):
            factors.append(FactorScore("indicator_confluence", None, 0.0, "PARTIAL"))
        else:
            votes, total = 0, 0
            total += 1
            votes += 1 if rsi_val > 55 else (-1 if rsi_val < 45 else 0)
            total += 1
            votes += 1 if macd_val > sig_val else -1
            total += 1
            if price > ema20_val > ema50_val:
                votes += 1
            elif price < ema20_val < ema50_val:
                votes -= 1
            factors.append(FactorScore("indicator_confluence", f"RSI={rsi_val:.1f}", votes / total, "GOOD"))

    vol = klines["volume"]
    if len(vol) < 21:
        factors.append(FactorScore("volume_confirmation", None, 0.0, "UNAVAILABLE"))
    else:
        recent, avg = vol.iloc[-1], vol.iloc[-21:-1].mean()
        if avg == 0 or pd.isna(avg):
            factors.append(FactorScore("volume_confirmation", None, 0.0, "PARTIAL"))
        else:
            ratio = recent / avg
            price_up = klines["close"].iloc[-1] > klines["close"].iloc[-2]
            s = (0.7 if price_up else -0.7) if ratio > 1.3 else (-0.2 if ratio < 0.6 else 0.0)
            factors.append(FactorScore("volume_confirmation", f"vol/avg={ratio:.2f}", s, "GOOD"))

    if rel_strength is None:
        factors.append(FactorScore("relative_strength_vs_btc", None, 0.0, "UNAVAILABLE"))
    else:
        s = max(-1.0, min(1.0, rel_strength / 5.0))
        factors.append(FactorScore("relative_strength_vs_btc", f"{rel_strength:+.2f}% vs BTC", s, "GOOD"))

    unavailable = sum(1 for f in factors if f.data_quality == "UNAVAILABLE")
    partial = sum(1 for f in factors if f.data_quality == "PARTIAL")
    quality_ratio = 1 - (unavailable + 0.5 * partial) / max(len(factors), 1)

    weights = CONFIG["signal_weights"]
    total_weight = sum(weights.values()) or 1
    weighted = sum(f.score * weights.get(f.name, 0) for f in factors) / total_weight
    confidence = abs(weighted) * 100 * quality_ratio
    direction = "BUY" if weighted > 0 else "SELL" if weighted < 0 else "NONE"
    data_quality_status = "GOOD" if quality_ratio >= 0.8 else "PARTIAL" if quality_ratio >= 0.5 else "STALE"

    decision = Decision.HOLD_WAIT
    entry_zone = invalidation = targets = risk_reward = None

    if quality_ratio * 100 < CONFIG["min_data_quality_for_trade"]:
        decision = Decision.NO_TRADE
    elif confidence >= CONFIG["min_score_buy_sell"] and direction != "NONE":
        price, price_ok = last_valid(closes)
        atr_val, atr_ok = last_valid(atr(klines["high"], klines["low"], closes))
        support, resistance = nearest_support_resistance(klines["high"], klines["low"], price or 0)
        if price_ok and atr_ok:
            if direction == "BUY":
                ref = support if support else price - atr_val
                if price - ref > 2.5 * atr_val:
                    entry_zone = "NO ENTRY — WAIT FOR RETEST"
                else:
                    decision = Decision.BUY_SETUP
                    entry_zone = f"{ref:.6g} - {price:.6g}"
                    invalidation = ref - atr_val
                    targets = [round(price + atr_val * m, 8) for m in (1.5, 2.5, 4.0)]
                    risk = price - invalidation
                    risk_reward = round((targets[0] - price) / risk, 2) if risk > 0 else None
            else:
                ref = resistance if resistance else price + atr_val
                if ref - price > 2.5 * atr_val:
                    entry_zone = "NO ENTRY — WAIT FOR RETEST"
                else:
                    decision = Decision.SELL_SETUP
                    entry_zone = f"{price:.6g} - {ref:.6g}"
                    invalidation = ref + atr_val
                    targets = [round(price - atr_val * m, 8) for m in (1.5, 2.5, 4.0)]
                    risk = invalidation - price
                    risk_reward = round((price - targets[0]) / risk, 2) if risk > 0 else None
        else:
            decision = Decision.NO_TRADE

    return {
        "symbol": symbol,
        "decision": decision.value,
        "confidence": round(confidence, 1),
        "historical_win_probability": None,
        "historical_note": "Not available in v1 — needs the v2 backtest module.",
        "entry_zone": entry_zone,
        "invalidation": round(invalidation, 8) if invalidation else None,
        "targets": targets,
        "risk_reward": risk_reward,
        "data_quality": data_quality_status,
    }


# ============================================================
# UNIVERSE + SCAN
# ============================================================

def get_eligible_universe(client: BinanceMarketData) -> List[dict]:
    info = client.get_exchange_info()
    tradable = {s["symbol"] for s in info["symbols"] if s["status"] == "TRADING" and s["quoteAsset"] == CONFIG["quote_asset"]}
    tickers = client.get_ticker_24hr()
    eligible = []
    for t in tickers:
        symbol = t["symbol"]
        if symbol not in tradable:
            continue
        if CONFIG["exclude_stablecoins"] and symbol in CONFIG["stablecoin_symbols"]:
            continue
        if CONFIG["exclude_leveraged_tokens"] and any(symbol.replace(CONFIG["quote_asset"], "").endswith(s) for s in CONFIG["leveraged_token_suffixes"]):
            continue
        try:
            qv, price, pct = float(t["quoteVolume"]), float(t["lastPrice"]), float(t["priceChangePercent"])
        except (KeyError, ValueError):
            continue
        if qv < CONFIG["min_24h_quote_volume_usdt"] or price < CONFIG["min_price_usdt"]:
            continue
        eligible.append({"symbol": symbol, "last_price": price, "quote_volume_24h": qv, "pct_change_24h": pct})
    return eligible


def _safe_klines(client, symbol, interval, limit):
    try:
        return client.get_klines(symbol, interval, limit)
    except MarketDataError as e:
        logger.warning("Klines fetch failed for %s %s: %s", symbol, interval, e)
        return None


def analyze_symbol(client, symbol, btc_klines_1h, btc_regime):
    klines = _safe_klines(client, symbol, CONFIG["primary_timeframe"], CONFIG["klines_limit"])
    if klines is None or len(klines) < 30:
        return {"symbol": symbol, "status": "DATA UNAVAILABLE"}
    rel = relative_strength_vs_btc(klines, btc_klines_1h) if btc_klines_1h is not None else None
    result = score_symbol(symbol, klines, btc_regime, rel)
    result["status"] = "OK"
    result["price"] = float(klines["close"].iloc[-1])
    return result


@st.cache_data(ttl=CONFIG["scan_cache_ttl_seconds"], show_spinner=False)
def run_scan_cached() -> dict:
    client = BinanceMarketData()
    limit = CONFIG["klines_limit"]
    btc_1h = _safe_klines(client, "BTCUSDT", "1h", limit)
    btc_4h = _safe_klines(client, "BTCUSDT", CONFIG["regime_timeframe_high"], limit)

    if btc_1h is not None and btc_4h is not None:
    
