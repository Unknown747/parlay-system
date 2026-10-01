# ============================================
# .env.example — Konfigurasi Parlay System
# Duplikasi jadi .env dan isi dengan data Anda
# ============================================

# ===== API KEYS (Rotasi) =====
OPENFOOT_API_KEYS=your_openfoot_key1,your_openfoot_key2
ODDS_API_KEYS=your_odds_key1,your_odds_key2
API_KEY=your_openfoot_key
ODDS_API_KEY=your_odds_key
API_BASE_URL=https://api.openfootapi.com/v1

# ===== LIGA & MUSIM =====
LEAGUES=EPL,LaLiga,SerieA,Bundesliga,Ligue1,Eredivisie,PrimeiraLiga,Championship
SEASONS=2021,2022,2023,2024,2025

# ===== O/U LINES =====
OU_LINES=1.5,2.5,3.5

# ===== CONFIDENCE & VALUE FILTER =====
MIN_CONFIDENCE=0.55
MIN_VALUE=0.03
MIN_HISTORICAL_MATCHES=100

# ===== TIER CONFIGURATION =====
TIER_S_CONF=0.72
TIER_S_VALUE=0.10
TIER_S_SKOR=0.48
STAKE_S=10000

TIER_A_CONF=0.65
TIER_A_VALUE=0.06
TIER_A_SKOR=0.40
STAKE_A=10000

TIER_B_CONF=0.55
TIER_B_VALUE=0.03
TIER_B_SKOR=0.30
STAKE_B=10000

# ===== STAKE LIMIT =====
MAX_DAILY_STAKE=100000
MAX_STAKE_PER_MATCH=50000
MAX_MARKETS_PER_MATCH=2
MAX_MARKETS_PER_LEAGUE=20
TRACKING_RETENTION_DAYS=90

# ===== CROSS-VALIDATION & SAFETY =====
CV_FOLDS=5
CV_OVERFITTING_THRESHOLD=0.15
MIN_BACKTEST_ROI=0.05
MIN_BACKTEST_ACCURACY=0.55

# ===== TELEGRAM NOTIFICATION =====
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# ===== DATABASE & CACHE =====
DB_PATH=data/processed/matches_normalized.csv
CACHE_DIR=data/raw/cache

# ===== SKOR WEIGHTING =====
SKOR_BOBOT_CONF=0.6
SKOR_BOBOT_VALUE=0.4

# ===== APIFY (Opsional) =====
APIFY_TOKEN=your_apify_token

# ===== LOG & DEBUG =====
DEBUG=False
LOG_LEVEL=INFO

# ===== LEGACY COMPATIBILITY =====
MIN_DATA_FRESHNESS_DAYS=7
MAX_MISSING_VALUE_PCT=0.15
STOP_LOSS_DAILY=-50000

# ===== MODEL =====
MODEL_VERSION_PREFIX=v5
