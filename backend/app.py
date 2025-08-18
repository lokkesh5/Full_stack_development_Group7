# app.py — Headline Impact Analyzer (batched quotes; correct change when market is closed)
# - Loads HF model from models/impact_* (or HEADLINE_MODEL_PATH)
# - Extracts tickers (aliases + cashtags)
# - BATCH fetch via yfinance.download for intraday & daily
# - Falls back to Stooq (.US + CSV) and returns last_close + prior_close
# - Shows yesterday's move when market is closed (no intraday)
# - Short-TTL cache + order preservation
import os, re, glob, time, io, urllib.request
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional

import pandas as pd
import torch
import yfinance as yf
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from pandas_datareader import data as pdr

# ---------------- Paths & defaults ----------------
API_PORT   = int(os.environ.get("PORT", "8000"))
BASE_DIR   = os.path.dirname(__file__)
MODELS_DIR = os.path.abspath(os.path.join(BASE_DIR, ".", "models"))
DATA_DIR   = os.path.join(BASE_DIR, "data")
TICKER_CSV = os.path.join(DATA_DIR, "tickers.csv")

DEFAULT_MODEL_ID = "ProsusAI/finbert"  # only used if no local impact_* folder is found

# ---------------- Overlay tunables ----------------
IMPACT_MAX_MOVE_PCT = float(os.environ.get("IMPACT_MAX_MOVE_PCT", "1.0"))
IMPACT_MAX_ABS_PCT  = float(os.environ.get("IMPACT_MAX_ABS_PCT",  "5.0"))
IMPACT_PROB_GAMMA   = float(os.environ.get("IMPACT_PROB_GAMMA",   "1.0"))
IMPACT_KW_MIN_MULT  = float(os.environ.get("IMPACT_KW_MIN_MULT",  "0.7"))
IMPACT_KW_MAX_MULT  = float(os.environ.get("IMPACT_KW_MAX_MULT",  "1.7"))
IMPACT_VOL_BASE_PCT = float(os.environ.get("IMPACT_VOL_BASE_PCT", "2.0"))
IMPACT_VOL_MIN_MULT = float(os.environ.get("IMPACT_VOL_MIN_MULT", "0.5"))
IMPACT_VOL_MAX_MULT = float(os.environ.get("IMPACT_VOL_MAX_MULT", "2.0"))

# ---------------- Fetch/caching knobs ----------------
YF_RETRIES   = int(os.environ.get("YF_RETRIES", "2"))
YF_BACKOFF   = float(os.environ.get("YF_BACKOFF", "0.6"))
QUOTE_TTL_S  = int(os.environ.get("YF_CACHE_TTL_SECONDS", "120"))
MAX_WORKERS  = int(os.environ.get("YF_MAX_WORKERS", "6"))

# ---------------- small utils ----------------
def clip(v, lo, hi): return max(lo, min(hi, v))

def resolve_latest_model(patterns):
    cands = []
    for pat in patterns:
        cands.extend(glob.glob(os.path.join(MODELS_DIR, pat)))
    dirs = [d for d in cands if os.path.isdir(d) and os.path.isfile(os.path.join(d, "config.json"))]
    if not dirs: return None
    dirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
    return dirs[0]

def load_pipeline(path_or_id):
    tok = AutoTokenizer.from_pretrained(path_or_id)
    model = AutoModelForSequenceClassification.from_pretrained(path_or_id)
    device = 0 if torch.cuda.is_available() else -1
    pipe = TextClassificationPipeline(model=model, tokenizer=tok,
                                      return_all_scores=True, device=device, truncation=True)
    return pipe, tok, model, device

def impact_from_scores(scores, up_threshold=0.55, margin=0.10):
    lbls = {s["label"].strip().lower(): float(s["score"]) for s in scores}
    if "up" in lbls and ("non_up" in lbls or "down" in lbls):
        up = lbls.get("up", 0.0); other = max(lbls.get("non_up", 0.0), lbls.get("down", 0.0))
        return "up" if (up >= up_threshold and up >= other + margin) else "down"
    pos, neg, neu = lbls.get("positive", 0.0), lbls.get("negative", 0.0), lbls.get("neutral", 0.0)
    if (pos or neg or neu):
        other = max(neg, neu)
        return "up" if (pos >= up_threshold and pos >= other + margin) else "down"
    lab0, lab1 = lbls.get("label_0", 0.0), lbls.get("label_1", 0.0)
    if lab0 or lab1:
        return "up" if (lab1 >= up_threshold and lab1 >= lab0 + margin) else "down"
    top = max(scores, key=lambda s: s["score"])
    return "up" if top["label"].lower().endswith("_1") else "down"

def prob_factor(scores, gamma=1.0):
    if not scores: return 0.0, 0.0
    probs = sorted([float(s["score"]) for s in scores], reverse=True)
    margin = probs[0] - (probs[1] if len(probs) > 1 else 0.5)
    return float(max(0.0, margin) ** gamma), float(max(0.0, margin))

# ---------------- keyword severity factor ----------------
SEV_POS = ("unveil", "launch", "partnership", "record", "beat", "outperform", "surge", "growth")
SEV_NEG = ("fraud", "probe", "lawsuit", "ban", "halt", "scandal", "recall", "investigation", "layoff", "bankruptcy")

def keyword_factor(text, arrow, floor=IMPACT_KW_MIN_MULT, cap=IMPACT_KW_MAX_MULT):
    t = text.lower()
    if arrow == "up":
        k = sum(1 for w in SEV_POS if w in t)
        return float(min(cap, max(floor, 1.0 + 0.15 * k))), [w for w in SEV_POS if w in t], k/5.0
    k = sum(1 for w in SEV_NEG if w in t)
    return float(min(cap, max(floor, 1.0 + 0.20 * k))), [w for w in SEV_NEG if w in t], k/5.0

# ---------------- ticker extraction (CSV + cashtags/aliases) ----------------
if os.path.isfile(TICKER_CSV):
    tickers_df = pd.read_csv(TICKER_CSV)
else:
    tickers_df = pd.DataFrame(
        [["AAPL", "Apple", "apple|iphone|ipad|mac|macbook"],
         ["MSFT", "Microsoft", "microsoft|windows|azure|xbox|office|bing"],
         ["AMZN", "Amazon", "amazon|aws|prime"],
         ["GOOGL", "Alphabet", "alphabet|google|youtube"],
         ["TSLA", "Tesla", "tesla|elon|musk|model 3|model y|autopilot"],
         ["NVDA", "NVIDIA", "nvidia|geforce|cuda|h100|a100"]],
        columns=["symbol", "name", "aliases"]
    )
tickers_df["symbol"] = tickers_df["symbol"].astype(str).str.upper()
SYMBOL_SET = set(tickers_df["symbol"].tolist())
SYN_MAP = {}
for _, r in tickers_df.iterrows():
    sym = r["symbol"]; aliases = {sym.lower()}
    if isinstance(r.get("name"), str): aliases.add(r["name"].lower())
    if isinstance(r.get("aliases"), str):
        for a in str(r["aliases"]).split("|"):
            a = a.strip().lower()
            if a: aliases.add(a)
    SYN_MAP[sym] = sorted(aliases)

CASHTAG_RE = re.compile(r"\$[A-Za-z]{1,5}\b")
WORD_SPLIT = re.compile(r"[^\w$]+")
def extract_tickers(text: str):
    text_lower = text.lower(); hits = set()
    for token in CASHTAG_RE.findall(text):
        sym = token[1:].upper()
        if sym in SYMBOL_SET: hits.add(sym)
    words = set([w for w in WORD_SPLIT.split(text_lower) if w])
    for sym, aliases in SYN_MAP.items():
        for alias in aliases:
            if " " in alias:
                if alias in text_lower: hits.add(sym); break
            else:
                if alias in words: hits.add(sym); break
    return sorted(hits)

# ---------------- caching ----------------
quote_cache: Dict[str, Dict] = {}  # {sym: {"data": row, "ts": epoch}}

def _serve_from_cache(sym: str):
    entry = quote_cache.get(sym)
    if entry and (time.time() - entry["ts"] <= QUOTE_TTL_S):
        return entry["data"]
    return None

def _store_cache(sym: str, data: dict):
    quote_cache[sym] = {"data": data, "ts": time.time()}

# ---------------- helpers for change ----------------
def _compute_change(price: Optional[float], prev_close: Optional[float]) -> Tuple[float, float]:
    if price is None or prev_close in (None, 0): return 0.0, 0.0
    delta = float(price) - float(prev_close)
    pct   = (delta / float(prev_close)) * 100.0 if prev_close else 0.0
    return round(delta, 6), round(pct, 6)

# Stooq fallback → returns (last_close, prior_close, debug)
def _stooq_last_and_prior(sym: str):
    dbg = {"tried": [], "hit": None, "errors": []}
    candidates = [
        sym.strip().upper(), f"{sym.strip().upper()}.US",
        sym.strip().lower(), f"{sym.strip().lower()}.us",
    ]
    # pandas-datareader first
    for s in candidates:
        try:
            dbg["tried"].append(f"pdr:{s}")
            df = pdr.DataReader(s, "stooq")
            if df is not None and not df.empty and "Close" in df.columns:
                df = df.sort_index()
                last = df["Close"].dropna().iloc[-1]
                prior = df["Close"].dropna().iloc[-2] if len(df["Close"].dropna()) > 1 else None
                if pd.notna(last):
                    dbg["hit"] = f"pdr:{s}"
                    return (float(last), float(prior) if prior is not None else None, dbg)
        except Exception as e:
            dbg["errors"].append(f"pdr:{s}:{type(e).__name__}:{e}")
    # CSV fallback
    for s in candidates:
        try:
            dbg["tried"].append(f"csv:{s}")
            url = f"https://stooq.com/q/d/l/?s={s}&i=d"
            with urllib.request.urlopen(url, timeout=6) as resp:
                csv_bytes = resp.read()
            df = pd.read_csv(io.BytesIO(csv_bytes))
            if df is not None and not df.empty and "Close" in df.columns:
                closes = df["Close"].dropna()
                if not closes.empty:
                    last = float(closes.iloc[-1])
                    prior = float(closes.iloc[-2]) if len(closes) > 1 else None
                    dbg["hit"] = f"csv:{s}"
                    return (last, prior, dbg)
        except Exception as e:
            dbg["errors"].append(f"csv:{s}:{type(e).__name__}:{e}")
    return (None, None, dbg)

def _finalize_row(sym: str, price, prev_close, is_stale, debug):
    now = datetime.now(timezone.utc)
    # if we still don't have a price but we do have prev_close, display prev_close
    if price is None and prev_close is not None:
        price = float(prev_close)
        debug["fell_back_to_prev_close"] = True
    change, change_pct = _compute_change(price, prev_close)
    row = {
        "symbol": sym,
        "price": round(float(price), 6) if price is not None else 0.0,
        "prev_close": round(float(prev_close), 6) if prev_close is not None else 0.0,
        "change": change, "change_pct": change_pct,
        "asof": now.isoformat(),
        "is_stale": bool(is_stale), "stale_seconds": None,
        "debug_providers": debug,  # useful if something looks off
    }
    _store_cache(sym, row)
    return row

# ---------------- batched quote fetch ----------------
def fetch_quotes_batched(symbols: List[str]) -> List[Dict]:
    syms = [s.strip().upper() for s in symbols if s and s.strip()]
    if not syms: return []

    # 0) cache first
    result: Dict[str, Dict] = {}
    missing = []
    for s in syms:
        cached = _serve_from_cache(s)
        if cached is not None: result[s] = cached
        else: missing.append(s)
    if not missing:
        return [result[s] for s in syms]

    # 1) Yahoo intraday (1m last)
    intra_last: Dict[str, float] = {}
    try:
        intraday = yf.download(
            tickers=missing, period="1d", interval="1m",
            group_by="ticker", auto_adjust=False, threads=True, progress=False
        )
    except Exception:
        intraday = None
    if intraday is not None and not isinstance(intraday, pd.Series):
        if isinstance(intraday.columns, pd.MultiIndex):
            for s in missing:
                try:
                    close = intraday[s]["Close"].dropna()
                    if not close.empty:
                        intra_last[s] = float(close.iloc[-1])
                except Exception:
                    pass
        else:
            if "Close" in intraday.columns:
                close = intraday["Close"].dropna()
                if not close.empty:
                    intra_last[missing[0]] = float(close.iloc[-1])

    # 2) Yahoo daily close (need **last & prior** close)
    last_close: Dict[str, float] = {}
    prior_close: Dict[str, float] = {}
    try:
        daily = yf.download(
            tickers=missing, period="10d", interval="1d",
            group_by="ticker", auto_adjust=False, threads=True, progress=False
        )
    except Exception:
        daily = None
    def _fill_last_prior_for_single_frame(df: pd.DataFrame, sym: str):
        closes = df["Close"].dropna() if "Close" in df.columns else pd.Series(dtype=float)
        if not closes.empty:
            last_close[sym] = float(closes.iloc[-1])
            if len(closes) > 1:
                prior_close[sym] = float(closes.iloc[-2])

    if daily is not None and not isinstance(daily, pd.Series):
        if isinstance(daily.columns, pd.MultiIndex):
            for s in missing:
                try:
                    _fill_last_prior_for_single_frame(daily[s], s)
                except Exception:
                    pass
        else:
            _fill_last_prior_for_single_frame(daily, missing[0])

    # 3) Assemble rows with correct change semantics
    for s in missing:
        debug = {"yahoo_intraday": None, "yahoo_daily_last": None, "yahoo_daily_prior": None, "stooq": None}
        is_stale = False
        price = None
        prev = None

        # intraday price available (market open) → compare to yesterday's close
        if s in intra_last:
            price = intra_last[s]
            prev = last_close.get(s)  # yesterday's official close
            is_stale = False
            debug["yahoo_intraday"] = "ok"

        # no intraday (market closed) → show yesterday's move (last vs prior)
        if price is None and s in last_close:
            price = last_close[s]              # yesterday close (displayed as 'Price')
            prev = prior_close.get(s, None)    # prior day's close (for change)
            is_stale = True                    # mark as EOD
            debug["yahoo_intraday"] = "none"

        debug["yahoo_daily_last"]  = "ok" if s in last_close else "none"
        debug["yahoo_daily_prior"] = "ok" if s in prior_close else "none"

        # if still missing prev (or both missing), try fast_info for previous_close
        if prev is None:
            try:
                fi = yf.Ticker(s).fast_info
                prev = float(fi.get("previous_close")) if fi.get("previous_close") is not None else prev
                if debug.get("yahoo_fastinfo_prev") is None:
                    debug["yahoo_fastinfo_prev"] = "ok" if prev is not None else "none"
            except Exception as e:
                debug["yahoo_fastinfo_prev"] = f"error:{type(e).__name__}:{e}"

        # Stooq backup if price is still None
        if price is None:
            stq_last, stq_prior, stq_dbg = _stooq_last_and_prior(s)
            debug["stooq"] = stq_dbg
            if stq_last is not None:
                price = stq_last
                if prev is None:
                    prev = stq_prior
                is_stale = True

        # finalize row (handles remaining None cases safely)
        result[s] = _finalize_row(s, price, prev, is_stale, debug)

    # preserve input order
    return [result.get(s, _serve_from_cache(s)) for s in syms]

# ---------------- volatility proxy (ATR%) ----------------
def _atr_pct(sym: str):
    try:
        h = yf.Ticker(sym).history(period="1mo", interval="1d", auto_adjust=False)
        if h.empty: return None
        closes = h["Close"].dropna()
        if len(closes) < 10: return None
        rets = closes.pct_change().dropna()
        vol = float(rets.rolling(14).std().dropna().iloc[-1]) * 100.0
        return round(vol, 6)
    except Exception:
        return None

def vol_factor(sym: str, base_pct: float, vmin: float, vmax: float):
    atrp = _atr_pct(sym)
    if atrp is None or base_pct <= 0: return 1.0, atrp
    ratio = atrp / float(base_pct)
    mult = max(min(ratio, vmax), vmin)
    return float(mult), atrp

# ---------------- Flask app & model load ----------------
app = Flask(__name__)
CORS(app)

headline_env = os.environ.get("HEADLINE_MODEL_PATH")
HEADLINE_MODEL_PATH = headline_env or resolve_latest_model(["impact_*", "impact_avisheksood_*"])

if HEADLINE_MODEL_PATH:
    headline_pipe, _, headline_model, device = load_pipeline(HEADLINE_MODEL_PATH)
    HEADLINE_MODEL_NAME = HEADLINE_MODEL_PATH
else:
    headline_pipe, _, headline_model, device = load_pipeline(DEFAULT_MODEL_ID)
    HEADLINE_MODEL_NAME = DEFAULT_MODEL_ID

@app.get("/health")
def health():
    return {
        "status": "ok",
        "headline_model": HEADLINE_MODEL_NAME,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "overlay": {
            "base_pct": IMPACT_MAX_MOVE_PCT,
            "max_abs_pct": IMPACT_MAX_ABS_PCT,
            "prob_gamma": IMPACT_PROB_GAMMA,
            "kw_min_mult": IMPACT_KW_MIN_MULT,
            "kw_max_mult": IMPACT_KW_MAX_MULT,
            "vol_base_pct": IMPACT_VOL_BASE_PCT,
            "vol_min_mult": IMPACT_VOL_MIN_MULT,
            "vol_max_mult": IMPACT_VOL_MAX_MULT,
        },
    }

@app.post("/analyze_headline")
def analyze_headline():
    payload = request.get_json(silent=True) or {}
    headline = (payload.get("headline") or "").strip()
    max_tickers = int(payload.get("max_tickers") or 10)
    if not headline:
        return jsonify({"error": "headline is required"}), 400

    # 1) model
    scores = headline_pipe(headline)[0]
    best = max(scores, key=lambda s: s["score"])
    sentiment_label = best["label"]
    confidence = float(best["score"])
    arrow = impact_from_scores(scores)

    # 2) overlay factors
    f_prob, prob_margin = prob_factor(scores, gamma=IMPACT_PROB_GAMMA)
    f_kw, kw_matched, kw_raw = keyword_factor(headline, arrow, IMPACT_KW_MIN_MULT, IMPACT_KW_MAX_MULT)
    BASE = IMPACT_MAX_MOVE_PCT / 100.0
    CAP  = IMPACT_MAX_ABS_PCT  / 100.0
    sign = 1.0 if arrow == "up" else -1.0

    # 3) tickers & quotes (batched)
    tickers = extract_tickers(headline)[:max_tickers]
    quotes  = fetch_quotes_batched(tickers)

    # 4) per-ticker projection + enrich rows
    for q in quotes:
        f_vol, atrp = vol_factor(q["symbol"], IMPACT_VOL_BASE_PCT, IMPACT_VOL_MIN_MULT, IMPACT_VOL_MAX_MULT)
        projected = sign * BASE * f_prob * f_kw * f_vol
        projected = clip(projected, -CAP, +CAP)
        px = q.get("price", 0.0) or 0.0
        q["predicted_change_pct"] = round(projected * 100, 3)
        q["projected_price"] = round(px * (1.0 + projected), 4) if px else 0.0
        q["overlay_factors"] = {
            "prob": round(f_prob, 3),
            "prob_margin": round(prob_margin, 3),
            "keywords_mult": round(f_kw, 3),
            "keywords_matched": kw_matched,
            "vol_mult": round(f_vol, 3),
            "atr_pct": (round(atrp, 3) if atrp is not None else None),
        }

    return jsonify({
        "headline": headline,
        "sentiment": sentiment_label,
        "confidence": confidence,
        "predicted_impact": arrow,
        "tickers": tickers,
        "quotes": quotes,
        "scores": scores,
        "overlay": {
            "base_pct": IMPACT_MAX_MOVE_PCT,
            "max_abs_pct": IMPACT_MAX_ABS_PCT,
            "prob_gamma": IMPACT_PROB_GAMMA,
            "kw_min_mult": IMPACT_KW_MIN_MULT,
            "kw_max_mult": IMPACT_KW_MAX_MULT,
            "vol_base_pct": IMPACT_VOL_BASE_PCT,
            "vol_min_mult": IMPACT_VOL_MIN_MULT,
            "vol_max_mult": IMPACT_VOL_MAX_MULT,
            "prob_margin": round(prob_margin, 3),
            "kw_raw": round(kw_raw, 3),
        }
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=API_PORT, debug=False, use_reloader=False)
