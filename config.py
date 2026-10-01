"""
config.py — Loader konfigurasi global dengan versioning system.
Dipakai oleh semua script di scripts/ dan menus/.
Versi 5: Tambah model versioning stabil, safety thresholds, dan market limits.
"""
import hashlib
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Root project = folder tempat config.py berada
ROOT_DIR = Path(__file__).resolve().parent

# Load .env
load_dotenv(ROOT_DIR / ".env")


# === VERSIONING SYSTEM ===
MODEL_VERSION_PREFIX = "v5"
N_LEAGUES = len([x.strip() for x in os.getenv("LEAGUES", "").split(",") if x.strip()])
TRAINING_DATE = datetime.now().strftime("%Y%m%d")


def _stable_model_version() -> str:
    """Gunakan hash dataset bila data features ada agar model version stabil."""
    cache = ROOT_DIR / "data/processed/matches_features.csv"
    if cache.exists():
        try:
            digest = hashlib.md5(cache.read_bytes()).hexdigest()[:8]
            return f"{MODEL_VERSION_PREFIX}_{N_LEAGUES}league_{digest}"
        except Exception:
            pass
    return f"{MODEL_VERSION_PREFIX}_{N_LEAGUES}league_{TRAINING_DATE}"


MODEL_VERSION = _stable_model_version()


# === Path Helper ===
def path(*parts) -> Path:
    """Gabung path relatif ke root project."""
    return ROOT_DIR.joinpath(*parts)


# === API ===
API_KEY = os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openfootapi.com/v1")

# === Liga & Musim ===
LEAGUES = [x.strip() for x in os.getenv("LEAGUES", "").split(",") if x.strip()]
SEASONS = [x.strip() for x in os.getenv("SEASONS", "").split(",") if x.strip()]

# === O/U Lines ===
OU_LINES = [float(x.strip()) for x in os.getenv("OU_LINES", "1.5,2.5,3.5").split(",") if x.strip()]

# === Filter Dasar ===
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.55"))
MIN_VALUE = float(os.getenv("MIN_VALUE", "0.03"))

# === Tier Thresholds ===
TIER_S = {
    "conf": float(os.getenv("TIER_S_CONF", "0.72")),
    "value": float(os.getenv("TIER_S_VALUE", "0.10")),
    "skor": float(os.getenv("TIER_S_SKOR", "0.48")),
    "stake": int(os.getenv("STAKE_S", "10000")),
}

TIER_A = {
    "conf": float(os.getenv("TIER_A_CONF", "0.65")),
    "value": float(os.getenv("TIER_A_VALUE", "0.06")),
    "skor": float(os.getenv("TIER_A_SKOR", "0.40")),
    "stake": int(os.getenv("STAKE_A", "10000")),
}

TIER_B = {
    "conf": float(os.getenv("TIER_B_CONF", "0.55")),
    "value": float(os.getenv("TIER_B_VALUE", "0.03")),
    "skor": float(os.getenv("TIER_B_SKOR", "0.30")),
    "stake": int(os.getenv("STAKE_B", "10000")),
}

# === Bankroll Management ===
MAX_DAILY_STAKE = int(os.getenv("MAX_DAILY_STAKE", "100000"))
MAX_STAKE_PER_MATCH = int(os.getenv("MAX_STAKE_PER_MATCH", "50000"))
MAX_MARKETS_PER_MATCH = int(os.getenv("MAX_MARKETS_PER_MATCH", "2"))
MAX_MARKETS_PER_LEAGUE = int(os.getenv("MAX_MARKETS_PER_LEAGUE", "20"))
TRACKING_RETENTION_DAYS = int(os.getenv("TRACKING_RETENTION_DAYS", "90"))
STOP_LOSS_DAILY = int(os.getenv("STOP_LOSS_DAILY", "-50000"))

# === Data Validation Thresholds ===
MIN_HISTORICAL_MATCHES = int(os.getenv("MIN_HISTORICAL_MATCHES", "100"))
MIN_DATA_FRESHNESS_DAYS = int(os.getenv("MIN_DATA_FRESHNESS_DAYS", "7"))
MAX_MISSING_VALUE_PCT = float(os.getenv("MAX_MISSING_VALUE_PCT", "0.15"))

# === Cross-Validation & Model Safety ===
CV_FOLDS = int(os.getenv("CV_FOLDS", "5"))
CV_OVERFITTING_THRESHOLD = float(os.getenv("CV_OVERFITTING_THRESHOLD", "0.15"))
MIN_BACKTEST_ROI = float(os.getenv("MIN_BACKTEST_ROI", "0.05"))
MIN_BACKTEST_ACCURACY = float(os.getenv("MIN_BACKTEST_ACCURACY", "0.55"))

# === Telegram ===
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# === Database ===
DB_PATH = path(os.getenv("DB_PATH", "data/processed/matches_normalized.csv"))
CACHE_DIR = path(os.getenv("CACHE_DIR", "data/raw/cache"))

# === Bobot Skor ===
SKOR_BOBOT_CONF = float(os.getenv("SKOR_BOBOT_CONF", "0.6"))
SKOR_BOBOT_VALUE = float(os.getenv("SKOR_BOBOT_VALUE", "0.4"))

# === Mapping Liga ke Kode football-data.co.uk ===
LEAGUE_CODES = {
    "EPL": "E0",
    "Championship": "E1",
    "LaLiga": "SP1",
    "SerieA": "I1",
    "Bundesliga": "D1",
    "Ligue1": "F1",
    "Eredivisie": "N1",
    "PrimeiraLiga": "P1",
    "LigaMX": "MEX",
    "Brasileirao": "BRA",
}

# === League-specific confidence boosts ===
LEAGUE_CONFIDENCE_ADJUST = {
    "EPL": 1.0,
    "LaLiga": 0.95,
    "SerieA": 0.90,
    "Bundesliga": 0.92,
    "Ligue1": 0.85,
    "Eredivisie": 0.80,
    "PrimeiraLiga": 0.80,
    "Championship": 0.85,
    "LigaMX": 0.75,
    "Brasileirao": 0.70,
}


def hitung_skor(confidence: float, value: float) -> float:
    """Hitung skor komposit dari confidence dan value."""
    return (confidence * SKOR_BOBOT_CONF) + (value * SKOR_BOBOT_VALUE)


def klasifikasi_tier(confidence: float, value: float) -> str:
    """Tentukan tier berdasarkan confidence, value, dan skor."""
    skor = hitung_skor(confidence, value)
    if confidence >= TIER_S["conf"] and value >= TIER_S["value"] and skor >= TIER_S["skor"]:
        return "S"
    if confidence >= TIER_A["conf"] and value >= TIER_A["value"] and skor >= TIER_A["skor"]:
        return "A"
    if confidence >= TIER_B["conf"] and value >= TIER_B["value"] and skor >= TIER_B["skor"]:
        return "B"
    return "X"


def stake_untuk_tier(tier: str) -> int:
    """Return stake IDR untuk tier tertentu."""
    return {"S": TIER_S["stake"], "A": TIER_A["stake"], "B": TIER_B["stake"]}.get(tier, 0)


def adjust_confidence_by_league(confidence: float, league: str) -> float:
    """Adjust confidence based on league-specific factor."""
    adjust_factor = LEAGUE_CONFIDENCE_ADJUST.get(league, 1.0)
    return confidence * adjust_factor


if __name__ == "__main__":
    print("=== CONFIG TEST ===")
    print(f"Model Version: {MODEL_VERSION}")
    print(f"N Leagues    : {N_LEAGUES}")
    print(f"Root         : {ROOT_DIR}")
    print(f"Leagues      : {LEAGUES}")
    print(f"Seasons      : {SEASONS}")
    print(f"O/U Lines    : {OU_LINES}")
    print()
    print("Tier Thresholds:")
    print(f"  Tier S   : {TIER_S}")
    print(f"  Tier A   : {TIER_A}")
    print(f"  Tier B   : {TIER_B}")
    print()
    print("Safety Thresholds:")
    print(f"  Max daily stake     : Rp {MAX_DAILY_STAKE:,}")
    print(f"  Max per match       : Rp {MAX_STAKE_PER_MATCH:,}")
    print(f"  Max markets/match   : {MAX_MARKETS_PER_MATCH}")
    print(f"  Max markets/league  : {MAX_MARKETS_PER_LEAGUE}")
    print(f"  Retention days      : {TRACKING_RETENTION_DAYS}")
    print(f"  Stop loss daily     : Rp {STOP_LOSS_DAILY:,}")
    print(f"  Min backtest ROI    : {MIN_BACKTEST_ROI:.1%}")
    print(f"  Min backtest acc    : {MIN_BACKTEST_ACCURACY:.1%}")
    print()
    print("Test klasifikasi:")
    for c, v in [(0.80, 0.15), (0.70, 0.08), (0.60, 0.04), (0.50, 0.02)]:
        t = klasifikasi_tier(c, v)
        s = hitung_skor(c, v)
        print(f"  conf={c} value={v} → skor={s:.4f} tier={t}")
    print()
    print("✓ Config loaded successfully")
