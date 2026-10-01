# PARLAY SYSTEM - COMPLETE SPECIFICATION

## PROJECT OVERVIEW

**Parlay System** adalah automated betting prediction platform yang menggunakan ensemble machine learning (XGBoost + LightGBM) untuk prediksi pertandingan football/soccer.

### Core Purpose
- Fetch fixture data dari API (OpenFootAPI)
- Train ML models untuk 3 market types: 1X2, O/U, BTTS
- Generate daily predictions dengan confidence score
- Classify predictions ke tier S/A/B berdasarkan expected value
- Track results & backtest performance

---

## COMPLETE MODULE SPECIFICATION

### 1. CONFIG.PY - Global Configuration Module

**Responsibility**: Centralized config, path management, constant definitions, helper functions

**Key Components**:
```
- path(relative_path) → Path object pointing to data/models/etc
- Constants:
  - TIER_S = {skor: 0.80, conf: 0.80, stake: 500000}
  - TIER_A = {skor: 0.50, conf: 0.70, stake: 250000}
  - TIER_B = {skor: 0.30, conf: 0.60, stake: 100000}
  - MAX_DAILY_STAKE = 2000000
  
- Functions:
  - hitung_skor(confidence, value) → float
  - klasifikasi_tier(confidence, value) → str ("S" | "A" | "B" | "X")
  - stake_untuk_tier(tier_name) → int
  - _stable_model_version() → str
```

**Input**: .env file with API_KEY, ALLOW_MANUAL_FIXTURES, etc
**Output**: Configuration object for other modules

---

### 2. REQUIREMENTS.TXT - Dependencies

```
pandas==2.0.0+       # Data manipulation
numpy==1.24.0+       # Numerical computing
scikit-learn==1.3.0+ # ML preprocessing, metrics
xgboost==2.0.0+      # Gradient boosting
lightgbm==4.0.0+     # LightGBM boosting
joblib==1.3.0+       # Model persistence
python-dotenv==1.0+  # .env loading
requests==2.31.0+    # HTTP API calls
```

---

### 3. SCRIPTS/FETCH_ALL.PY - Data Fetching Module

**Responsibility**: Fetch fixture data from external API, validate, save

**Input**: 
- OpenFootAPI endpoint (from .env)
- League whitelist (from config or .env)

**Processing**:
1. Connect to API_BASE_URL + API_KEY
2. For each enabled league:
   - Fetch upcoming fixtures for next 7-30 days
   - Parse JSON response
   - Extract: match_id, league, date, home_team, away_team, odds (1x2, ou, btts)
3. Normalize data:
   - Standardize team names
   - Validate date format (YYYYMMDD or ISO)
   - Check odds range (1.0 - 50.0)
4. Validate against schema

**Output**: `data/raw/fixtures_today.csv` with columns:
```
match_id | league | date | home_team | away_team | 
odds_home | odds_draw | odds_away | 
odds_over_1_5 | odds_under_1_5 | 
odds_over_2_5 | odds_under_2_5 | 
odds_over_3_5 | odds_under_3_5 | 
odds_btts_yes | odds_btts_no
```

**Error Handling**:
- Retry failed API calls (3x with backoff)
- Log API errors
- Fallback to cached fixtures if API down
- Validation errors halt processing with report

---

### 4. SCRIPTS/VALIDATE_FIXTURES.PY - Data Validation Module

**Responsibility**: Ensure fixture data quality before ML processing

**Checks**:
1. **Schema**: All required columns present
2. **Types**: Correct data types (date as datetime, odds as float)
3. **Ranges**:
   - Odds: 1.0 - 50.0
   - Confidence: 0.0 - 1.0
   - Probability sum to ~1.0
4. **Consistency**:
   - No duplicate match_ids
   - Team names consistent (whitespace, case)
   - Dates in future (not past)
5. **Integrity**:
   - No null odds for major markets
   - Match at least 1 hour away

**Input**: fixtures_today.csv

**Output**: 
- Valid: Boolean
- Errors: List[str] with specific issues
- Warnings: List[str] for non-blocking issues

---

### 5. SCRIPTS/LEAGUE_REGISTRY.PY - League Configuration

**Responsibility**: Define league-specific parameters

**Components**:
```python
@dataclass
class LeagueSpec:
    code: str              # "EPL", "LaLiga", etc
    name: str              # Full name
    country: str           # Country code (EN, ES, etc)
    tier: int              # 1 (top) to 4 (lower)
    enabled: bool          # Include in fetch/predict
    api_endpoint: str      # League-specific API path
    weight: float          # Scoring multiplier (0.5-1.5)
    min_confidence: float  # Minimum confidence threshold
```

**Predefined Leagues**:
- Top tier (weight 1.2-1.3): Eredivisie, Ligue1
- Mid tier (weight 0.8-1.0): EPL, LaLiga, Bundesliga, SerieA
- Lower tier (weight 0.5-0.8): Championship, LigaMX, Brasileirao

**Usage**: Loaded by predict.py for league weighting

---

### 6. SCRIPTS/TRAIN.PY - ML Training Module

**Responsibility**: Train ensemble models on historical data

**Input**: 
- Historical matches CSV (date, home_team, away_team, home_goals, away_goals, etc)
- Feature engineering config

**Processing**:

1. **Data Loading & Cleaning**:
   - Load matches_normalized.csv
   - Filter to last 3 years
   - Remove low-tier leagues
   - Handle missing values

2. **Feature Engineering**:
   ```
   - Team form: form_5 (last 5 games), form_10 (last 10)
   - Scoring: goals_for_avg_5, goals_against_avg_5
   - Strength: win_rate, draw_rate
   - League avg: league_avg_home_goals, league_avg_away_goals
   - Odds features: 1/odds_* (implied probability)
   ```

3. **Label Creation**:
   - 1X2: home_goals > away_goals → "H", else "D", else "A"
   - O/U: (home_goals + away_goals) > threshold
   - BTTS: both teams scored (home_goals > 0 AND away_goals > 0)

4. **Model Training**:
   For each task (1X2, O/U 1.5/2.5/3.5, BTTS):
   - Split: 80% train, 20% test (time-ordered)
   - Train XGBoost classifier
   - Train LightGBM classifier
   - Calculate weights to optimize log-loss on test set
   - K-fold validation (5-fold)
   - Store both models + weights

5. **Model Evaluation**:
   - Accuracy, Precision, Recall, F1-score per class
   - ROC-AUC, Logloss
   - Output metrics_train.json

**Output**: 
- `models/model_1x2_multiclass.pkl` (ensemble XGB + LGBM)
- `models/model_ou_15.pkl`, `model_ou_25.pkl`, `model_ou_35.pkl`
- `models/model_btts.pkl`
- `models/model_weights.json` (XGB/LGBM weights per task)

**Config**: 
- Train/test split ratio
- K-fold k value
- XGB hyperparams: max_depth=5, learning_rate=0.1, n_estimators=100
- LGBM hyperparams: max_depth=5, learning_rate=0.1, n_estimators=100

---

### 7. SCRIPTS/PREDICT.PY - Daily Prediction Module

**Responsibility**: Generate predictions for today's fixtures using trained models

**Input**:
- fixtures_today.csv
- Trained models from models/ folder
- Historical stats for team form calculation
- (Optional) manual_bets.csv

**Processing**:

1. **Load Models**:
   - model_1x2_multiclass.pkl → ensemble (XGB + LGBM)
   - model_ou_*.pkl → 3 models for 1.5, 2.5, 3.5 lines
   - model_btts.pkl → binary ensemble

2. **Feature Building**:
   For each fixture:
   - Get team stats from last 10 games (form, goals avg, win rate)
   - Get league averages
   - Build feature vector matching training features
   - Scale/normalize

3. **Ensemble Prediction** (for each task):
   ```python
   xgb_prob = xgb_model.predict_proba(X)
   lgbm_prob = lgbm_model.predict_proba(X)
   
   final_prob = weights["xgb"] * xgb_prob + weights["lgbm"] * lgbm_prob
   
   prediction = argmax(final_prob) if multiclass
              = (prob[:, 1] > 0.5) if binary
   ```

4. **Market Generation**:
   For each fixture, generate:
   
   a) **1X2 Market**:
      - Get class probabilities [P(Home), P(Draw), P(Away)]
      - Select highest confidence class
      - Pick matching odds (odds_home | odds_draw | odds_away)
      - Result: {"market": "1X2", "prediction": "Home", "confidence": 0.65, "odds": 1.95}
   
   b) **O/U Markets** (1.5, 2.5, 3.5):
      - Get P(Over) from model
      - If P(Over) > 0.5: Over else Under
      - Pick matching odds
      - Result: {"market": "O/U 2.5", "prediction": "Over", "confidence": 0.58, "odds": 1.85}
   
   c) **BTTS Market**:
      - Get P(BTTS=Yes) from model
      - If > 0.5: Yes else No
      - Pick matching odds
      - Result: {"market": "BTTS", "prediction": "Yes", "confidence": 0.62, "odds": 1.72}

5. **Load Manual Bets**:
   - Call load_manual_data.load_manual_bets()
   - Append to results

**Output Structure**: 
```json
{
  "date": "2026-10-01",
  "total_scan": 150,
  "results": [
    {
      "match_id": "EPL_20261001_ManCity_Liverpool",
      "league": "EPL",
      "home": "Manchester City",
      "away": "Liverpool",
      "date": "2026-10-01",
      "predictions": [
        {
          "market": "1X2",
          "prediction": "Home",
          "confidence": 0.72,
          "odds": 2.10
        },
        {
          "market": "O/U 2.5",
          "prediction": "Over",
          "confidence": 0.65,
          "odds": 1.90
        },
        {
          "market": "BTTS",
          "prediction": "Yes",
          "confidence": 0.58,
          "odds": 1.85
        }
      ]
    }
  ]
}
```

---

### 8. SCRIPTS/PREDICT_UPDATED.PY (Alternative)

**Extends predict.py with**:
- Additional score weighting by league
- Tier filtering with MIN_WEIGHTED_SCORE threshold
- Output formatting with text report
- Automatic tracking append

**Key Additions**:
```python
LEAGUE_WEIGHTS = {
    "Eredivisie": 1.30, "Ligue1": 1.20,
    "PrimeiraLiga": 0.95, "Championship": 0.90,
    "LaLiga": 0.70, "SerieA": 0.65,
    "EPL": 0.60, "Bundesliga": 0.55,
    "LigaMX": 0.80, "Brasileirao": 0.75
}

MIN_WEIGHTED_SCORE = 0.30  # Minimum score to include
```

---

### 9. SCRIPTS/FILTER_AND_CLASSIFY.PY - Tier Classification

**Responsibility**: Convert raw predictions to actionable tier-based bets

**Input**: predictions (list of match predictions)

**Processing for each prediction**:

1. **Expected Value Calculation**:
   ```
   value = confidence × odds - 1
   ```

2. **Base Score Calculation**:
   ```
   score_base = hitung_skor(confidence, value)
   ```

3. **League-Weighted Score**:
   ```
   score_final = score_base × league_weight[league]
   ```

4. **Tier Assignment**:
   ```
   if score_final >= TIER_S["skor"] and confidence >= TIER_S["conf"]:
       tier = "S"
   elif score_final >= TIER_A["skor"] and confidence >= TIER_A["conf"]:
       tier = "A"
   elif score_final >= TIER_B["skor"] and confidence >= TIER_B["conf"]:
       tier = "B"
   else:
       skip  # Below minimum threshold
   ```

5. **Filter**:
   - Remove if score_final < MIN_WEIGHTED_SCORE
   - Skip if odds <= 1.0 (no value)

**Output**: 
```
tier_s: List[Bet]  # 10-20 top bets
tier_a: List[Bet]  # 20-50 medium confidence
tier_b: List[Bet]  # 50+ entertainment bets
```

Each Bet contains:
```
{
  "match_id", "league", "home", "away", "date",
  "market", "prediction", "confidence", "odds",
  "value", "skor_dasar", "skor", "bobot_liga", "stake"
}
```

---

### 10. SCRIPTS/BACKTEST.PY - Performance Validation

**Responsibility**: Validate model accuracy against actual match results

**Input**:
- Historical predictions (JSON/CSV)
- Actual match results

**Processing**:

1. **Load Results**:
   - Match predictions with actual outcomes
   - Link via match_id + market

2. **Outcome Classification**:
   - Win: prediction == actual result
   - Loss: prediction != actual result

3. **Statistics by Category**:
   - By league: EPL, LaLiga, etc
   - By market: 1X2, O/U, BTTS
   - By tier: S, A, B
   
   Metrics per category:
   ```
   - Total bets
   - Wins / Losses
   - Win rate (%)
   - ROI (profit / total_stake * 100)
   - Average odds
   - Average confidence
   - Cumulative profit (assuming stake per tier)
   ```

4. **Model Comparison**:
   - S tier performance vs A tier vs B tier
   - Confidence calibration (confidence 0.80 predictions win 80%?)
   - Odds efficiency (high odds predictions profitable?)

**Output**: `data/processed/backtest_stats.csv` with columns:
```
category | metric | value | win_rate | roi | avg_odds | profit_loss
```

---

### 11. SCRIPTS/LOAD_MANUAL_DATA.PY - Manual Data Integration

**Responsibility**: Load user-provided data (fixtures, bets) to augment AI predictions

**Functions**:

1. **load_manual_fixtures()**:
   - Input: `data/raw/manual_fixtures.csv`
   - Columns: league, date, home_team, away_team, odds_*
   - Returns: DataFrame
   - Usage: Backfill missing historical data

2. **load_manual_bets()**:
   - Input: `data/raw/manual_bets.csv`
   - Columns: liga, home_team, away_team, tanggal, market, prediksi, odds, confidence, catatan
   - Returns: List[Dict] matching predictions structure
   - Usage: Add user-found bets to daily predictions
   - Generates match_id automatically if not provided

**Config**:
- ALLOW_MANUAL_FIXTURES env var (default True)
- ALLOW_MANUAL_BETS env var (default True)

---

### 12. SCRIPTS/VERIFY_SETUP.PY - Health Check

**Responsibility**: Validate system readiness before running

**Checks**:

1. **check_files()**:
   - Required: config.py, requirements.txt, scripts/*.py
   - Folders: data/, models/, menus/
   - Data: .env, .env.example
   - Output: List[Dict] with status

2. **check_env()**:
   - API_BASE_URL set
   - API_KEY valid
   - Path vars accessible
   - Env file permissions OK

3. **check_models()**:
   - model_1x2_multiclass.pkl exists
   - All o/u models exist
   - model_btts.pkl exists
   - Models loadable (joblib.load test)

4. **check_api()**:
   - Ping API_BASE_URL
   - Verify API_KEY authentication
   - Check rate limit headers

5. **check_data()**:
   - data/raw/ writable
   - data/processed/ writable
   - models/ readable
   - No missing critical files

**Output**: Verification report (print + JSON)

---

### 13. SETUP.SH - Installation Script

**Steps**:
1. Check Python 3.9+
2. Create .venv virtual environment
3. Activate venv
4. pip install -r requirements.txt
5. Copy .env.example → .env (if not exists)
6. Create data/raw, data/processed, models folders
7. Run verify_setup.py
8. Run train.py (optional, can be skipped on first setup)

---

### 14. MAIN_MENU.SH - Interactive CLI

**Menu Options**:
1. Fetch today's fixtures
2. Train models
3. Run predictions
4. Backtest performance
5. Verify setup
6. Clean generated files
7. Exit

---

### 15. CLEANUP_GENERATED.SH - Reset Script

**Removes**:
- Generated predictions (predictions_today.*)
- Model files (models/*.pkl)
- Backtest results
- Temporary CSV files

**Preserves**:
- .env configuration
- Manual data files
- Historical tracking

---

## DATA FLOW DIAGRAM

```
┌──────────────────────────────────────────────────────────────┐
│                      PARLAY SYSTEM FLOW                       │
└──────────────────────────────────────────────────────────────┘

                        DAILY ROUTINE
                            │
                            ▼
                    ┌────────────────┐
                    │  FETCH_ALL.PY  │
                    │  (API call)    │
                    └────────┬───────┘
                             │
                      fixtures_today.csv
                             │
                             ▼
                  ┌───────────────────────┐
                  │ VALIDATE_FIXTURES.PY  │
                  │ (schema, ranges, etc) │
                  └───────────┬───────────┘
                              │
                         ✓ Valid
                              │
        ┌─────────────────────┴──────────────────────┐
        │                                            │
        ▼                                            ▼
    ┌───────────┐                          ┌─────────────────┐
    │ TRAIN.PY  │◄─────────────────────────│ Historical data │
    │ (monthly) │     (matches.csv)        │ (updated daily) │
    └─────┬─────┘                          └─────────────────┘
          │
       models/
      (saved)
          │
          └──────────────────────┐
                                 │
                                 ▼
                        ┌──────────────────┐
                        │  PREDICT.PY      │
                        │ (ensemble infer) │
                        └────────┬─────────┘
                                 │
                        predictions (JSON)
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
    ┌──────────┐          ┌─────────────┐          ┌─────────┐
    │ TIER S   │          │  TIER A     │          │ TIER B  │
    │ (premium)│          │ (standard)  │          │(casual) │
    └────┬─────┘          └──────┬──────┘          └────┬────┘
         │                       │                      │
         └───────────────────────┼──────────────────────┘
                                 │
                                 ▼
                    ┌─────────────────────┐
                    │ FILTER & CLASSIFY   │
                    │ (score + weighting) │
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
    predictions_today.json  predictions_today.txt  tracking_pending.csv
            │                  │                  │
            └──────┬───────────┴──────────┬───────┘
                   │ (append)             │
                   ▼                      ▼
            ┌─────────────────────────────────────────┐
            │  USER REVIEW & MANUAL BET ENTRY         │
            │  (check predictions, add custom bets)   │
            └─────────────────────────────────────────┘
                   │
                   ▼
         ┌──────────────────┐
         │ PLACE BETS       │
         │ (betting site)   │
         └──────────────────┘
                   │
              (match day)
                   │
                   ▼
         ┌──────────────────┐
         │ ACTUAL RESULTS   │
         │ (outcomes)       │
         └──────────┬───────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  BACKTEST.PY         │
         │ (analyze accuracy)   │
         └──────────────────────┘
                    │
                    ▼
         backtest_stats.csv
```

---

## ENVIRONMENT VARIABLES (.ENV)

```ini
# API Configuration
API_BASE_URL=https://api.openfootapi.com/v1
API_KEY=your_api_key_here

# Feature Flags
ALLOW_MANUAL_FIXTURES=True
ALLOW_MANUAL_BETS=True
APPEND_TRACKING=True

# Betting Limits
MAX_DAILY_STAKE=2000000

# League Configuration
LEAGUE_WHITELIST=EPL,LaLiga,Bundesliga,SerieA,Ligue1,Eredivisie

# Model Configuration
MODEL_VERSION=1.0.0
USE_CACHED_MODELS=True

# Logging
LOG_LEVEL=INFO
DEBUG=False
```

---

## FOLDER STRUCTURE

```
parlay-system/
│
├── ROOT FILES
│   ├── config.py                    # Global configuration
│   ├── requirements.txt             # Python dependencies
│   ├── .env                         # Config (secrets, gitignore)
│   ├── .env.example                 # Config template
│   ├── setup.sh                     # Initial setup
│   ├── main_menu.sh                 # CLI menu
│   ├── cleanup_generated.sh         # Cleanup script
│   ├── AUDIT_REPORT.md              # Audit log
│   └── SYSTEM_SPECIFICATION.md      # This file
│
├── DATA FOLDER
│   ├── data/raw/
│   │   ├── fixtures_today.csv       # Today's API fixtures
│   │   ├── manual_fixtures.csv      # User-provided fixtures (optional)
│   │   └── manual_bets.csv          # User-provided bets (optional)
│   │
│   └── data/processed/
│       ├── matches_normalized.csv   # Historical normalized matches
│       ├── predictions_today.json   # Daily predictions (JSON)
│       ├── predictions_today.txt    # Daily predictions (formatted text)
│       ├── tracking_pending.csv     # Bets to track (appends daily)
│       └── backtest_stats.csv       # Backtest results
│
├── MODELS FOLDER
│   └── models/
│       ├── model_1x2_multiclass.pkl       # 1X2 predictions
│       ├── model_ou_15.pkl                # O/U 1.5 predictions
│       ├── model_ou_25.pkl                # O/U 2.5 predictions
│       ├── model_ou_35.pkl                # O/U 3.5 predictions
│       ├── model_btts.pkl                 # BTTS predictions
│       └── model_weights.json             # Ensemble weights
│
├── SCRIPTS FOLDER
│   └── scripts/
│       ├── fetch_all.py              # Fetch fixtures from API
│       ├── fetch_all_updated.py      # Alternative fetch
│       ├── train.py                  # Train ML models
│       ├── predict.py                # Generate predictions
│       ├── predict_updated.py        # Alternative predict (with tiers)
│       ├── backtest.py               # Backtest analysis
│       ├── verify_setup.py           # Health check
│       ├── validate_fixtures.py      # Fixture validation
│       ├── load_manual_data.py       # Load manual data
│       └── league_registry.py        # League definitions
│
└── MENUS FOLDER
    └── menus/ (optional, TBD)
        ├── main_menu.py              # Main menu UI
        ├── fetch_menu.py             # Fetch options
        ├── train_menu.py             # Training options
        ├── predict_menu.py           # Prediction options
        └── backtest_menu.py          # Backtest options
```

---

## USAGE EXAMPLES

### 1. First Time Setup

```bash
cd parlay-system
bash setup.sh
```

### 2. Daily Prediction Workflow

```bash
# Activate venv
source .venv/bin/activate

# Fetch today's fixtures
python scripts/fetch_all.py

# Generate predictions
python scripts/predict.py

# View predictions
cat data/processed/predictions_today.txt

# (Optional) Run backtest if you have results data
python scripts/backtest.py
```

### 3. Retraining Models (Monthly)

```bash
# Get latest historical data
python scripts/fetch_all.py  # or manually update matches.csv

# Train models
python scripts/train.py

# Verify models loaded
python scripts/verify_setup.py
```

### 4. Adding Manual Bets

Create `data/raw/manual_bets.csv`:
```csv
liga,home_team,away_team,tanggal,market,prediksi,odds,confidence,catatan
EPL,Man City,Liverpool,2026-10-01,1X2,Home,2.10,0.72,Strong form
EPL,Man City,Liverpool,2026-10-01,O/U 2.5,Over,1.90,0.65,Both attack-minded
```

Then run:
```bash
python scripts/predict.py
```

Manual bets will be appended to predictions.

### 5. Cleanup

```bash
bash cleanup_generated.sh
```

---

## TIER SYSTEM REFERENCE

| Tier | Min Score | Min Confidence | Typical Stake | Risk | Use Case |
|------|-----------|----------------|---------------|------|----------|
| **S** | 0.80+ | 80%+ | Rp 500k | Lowest | All-in safe bets |
| **A** | 0.50+ | 70%+ | Rp 250k | Low | Main portfolio |
| **B** | 0.30+ | 60%+ | Rp 100k | Medium | Entertainment/Hedge |

**Stake Allocation Example** (Rp 2M daily limit):
- Tier S: 2-3 bets × Rp 500k = Rp 1M
- Tier A: 3-5 bets × Rp 250k = Rp 625k-1M
- Tier B: Remaining (if any)

---

## ERROR HANDLING & TROUBLESHOOTING

| Issue | Cause | Solution |
|-------|-------|----------|
| "API connection failed" | Network/API down | Check API status, retry with backoff |
| "Models not found" | Models not trained | Run `python scripts/train.py` |
| "Fixtures invalid" | API format changed | Update validate_fixtures.py schema |
| "Confidence 0.0 for all" | Team stats missing | Check historical data completeness |
| "FileNotFoundError: .env" | Config not initialized | Run `cp .env.example .env` |
| "Path too long error" (Windows) | Filename exceeds 260 chars | Use short alias paths, enable long paths |

---

## INTEGRATION POINTS FOR OTHER AIs

To rebuild parts of this system with another AI (Claude, Gemini, etc), provide them with:

1. **Input Requirements**: See each module's "Input" section
2. **Processing Logic**: See each module's "Processing" section
3. **Output Format**: See each module's "Output" section
4. **Config Schema**: See ENVIRONMENT VARIABLES section
5. **Error Handling**: See this section

Example prompt for Claude to rebuild predict.py:

```
I want you to rebuild predict.py for a sports betting prediction system.

Requirements:
- Input: fixtures_today.csv, trained ensemble models from models/ folder
- Processing: Load fixtures, build features, run ensemble XGBoost+LightGBM inference
- Output: Predictions JSON with structure [match_id, league, predictions[{market, prediction, confidence, odds}]]
- Must support: 1X2, O/U 1.5/2.5/3.5, BTTS markets

See COMPLETE MODULE SPECIFICATION section 7 for full details.
```

---

## VERSION HISTORY

- v0.1: Initial scaffold
- v0.2: API integration
- v0.3: ML training pipeline
- v0.4: Prediction generation
- v0.5: Tier classification & tracking
- v1.0: Production release

---

*End of Specification Document*
