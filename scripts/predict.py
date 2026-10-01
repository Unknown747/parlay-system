"""
predict.py — Prediksi fixtures + filter 3 tier + bobot liga.
Versi 7: support 1X2 multiclass, O/U spesifik, metadata model, market limits, whitelist, tracking enforcement.
"""
import sys
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    path, OU_LINES, TIER_S, TIER_A, TIER_B,
    MAX_DAILY_STAKE, MAX_MARKETS_PER_MATCH, MAX_MARKETS_PER_LEAGUE,
    hitung_skor, klasifikasi_tier, stake_untuk_tier
)
from train import FEATURES
from validate_fixtures import validate as validate_fixtures
from league_whitelist import filter_fixtures_by_whitelist

LEAGUE_WEIGHTS = {"Eredivisie": 1.30, "Ligue1": 1.20, "PrimeiraLiga": 0.95, "Championship": 0.90, "LaLiga": 0.70, "SerieA": 0.65, "EPL": 0.60, "Bundesliga": 0.55, "LigaMX": 0.80, "Brasileirao": 0.75}
MIN_WEIGHTED_SCORE = 0.30
FIXTURE_COLUMNS = ["match_id", "league", "date", "home_team", "away_team", "odds_home", "odds_draw", "odds_away", "odds_over_2_5", "odds_under_2_5", "odds_over_1_5", "odds_under_1_5", "odds_over_3_5", "odds_under_3_5"]


def format_match_date(value):
    if value is None or pd.isna(value):
        return "-"
    try:
        if isinstance(value, str):
            value = value.strip()
            if len(value) == 8 and value.isdigit():
                try:
                    return pd.to_datetime(value, format="%Y%m%d").strftime("%d %b %Y")
                except Exception:
                    pass
            try:
                return pd.to_datetime(value, errors="coerce").strftime("%d %b %Y")
            except Exception:
                return value
        return pd.to_datetime(value, errors="coerce").strftime("%d %b %Y")
    except Exception:
        return str(value)


def load_models():
    models = {}
    p_1x2 = path("models/model_1x2_multiclass.pkl")
    if not p_1x2.exists():
        raise FileNotFoundError(f"Model tidak ada: {p_1x2}")
    models["model_1x2_multiclass"] = joblib.load(p_1x2)
    for line in [1.5, 2.5, 3.5]:
        model_name = f"model_ou_{int(line*10)}"
        p = path(f"models/{model_name}.pkl")
        if p.exists():
            models[model_name] = joblib.load(p)
    p_btts = path("models/model_btts.pkl")
    if not p_btts.exists():
        raise FileNotFoundError(f"Model tidak ada: {p_btts}")
    models["model_btts"] = joblib.load(p_btts)
    return models


def load_model_metadata():
    meta_path = path("logs/training_log.json")
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text())
        except Exception:
            return {"model_version": "unknown"}
    return {"model_version": "unknown"}


def load_fixtures() -> pd.DataFrame:
    p = path("data/raw/fixtures_today.csv")
    if not p.exists():
        raise FileNotFoundError(f"Fixtures tidak ada: {p}\nBuat file CSV dengan kolom: {','.join(FIXTURE_COLUMNS)}")
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for i, row in df.iterrows():
        mid = str(row.get("match_id", "")).strip()
        if not mid or mid.lower() in ("nan", "none", ""):
            liga = str(row.get("league", "UNK")).strip()
            dt = pd.to_datetime(row.get("date"), errors="coerce")
            dt_str = dt.strftime("%Y%m%d") if pd.notna(dt) else "00000000"
            home = str(row.get("home_team", "")).replace(" ", "")
            away = str(row.get("away_team", "")).replace(" ", "")
            df.at[i, "match_id"] = f"{liga}_{dt_str}_{home}_{away}"
    return df


def load_historical_stats() -> pd.DataFrame:
    p = path("data/processed/matches_normalized.csv")
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, low_memory=False)


def get_team_stats(hist: pd.DataFrame, team: str, is_home: int) -> dict:
    if hist.empty:
        return {}
    mask = (hist["home_team"] == team) | (hist["away_team"] == team)
    team_df = hist[mask].sort_values("date").tail(10)
    if team_df.empty:
        return {}
    form_col = "home_form_5" if is_home else "away_form_5"
    form_col10 = "home_form_10" if is_home else "away_form_10"
    gf_col = "home_goals_for_avg_5" if is_home else "away_goals_for_avg_5"
    ga_col = "home_goals_against_avg_5" if is_home else "away_goals_against_avg_5"
    win_col = "home_win_rate" if is_home else "away_win_rate"
    return {
        "form_5": team_df[form_col].mean() if form_col in team_df else 1.5,
        "form_10": team_df[form_col10].mean() if form_col10 in team_df else 1.5,
        "goals_for_avg_5": team_df[gf_col].mean() if gf_col in team_df else 1.5,
        "goals_against_avg_5": team_df[ga_col].mean() if ga_col in team_df else 1.5,
        "win_rate": team_df[win_col].mean() if win_col in team_df else 0.5,
    }


def build_features(fixtures: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    if not hist.empty:
        league_avg = hist.groupby("league").agg({
            "home_goals": "mean", "away_goals": "mean", "home_win_rate": "mean", "away_win_rate": "mean",
            "h2h_avg_goals": "mean", "h2h_home_wins": "mean"
        }).to_dict()
    else:
        league_avg = {}
    rows = []
    for _, f in fixtures.iterrows():
        liga = f.get("league", "EPL")
        avg = league_avg.get(liga, {})
        home = f.get("home_team", "")
        away = f.get("away_team", "")
        home_stats = get_team_stats(hist, home, is_home=1)
        away_stats = get_team_stats(hist, away, is_home=0)
        rows.append({
            "match_id": f.get("match_id"), "league": liga, "date": f.get("date"), "home_team": home, "away_team": away,
            "odds_home": f.get("odds_home"), "odds_draw": f.get("odds_draw"), "odds_away": f.get("odds_away"),
            "odds_over_2_5": f.get("odds_over_2_5"), "odds_under_2_5": f.get("odds_under_2_5"),
            "odds_over_1_5": f.get("odds_over_1_5"), "odds_under_1_5": f.get("odds_under_1_5"),
            "odds_over_3_5": f.get("odds_over_3_5"), "odds_under_3_5": f.get("odds_under_3_5"),
            "home_form_5": home_stats.get("form_5", 1.5), "away_form_5": away_stats.get("form_5", 1.5),
            "home_form_10": home_stats.get("form_10", 1.5), "away_form_10": away_stats.get("form_10", 1.5),
            "home_goals_for_avg_5": home_stats.get("goals_for_avg_5", 1.5), "away_goals_for_avg_5": away_stats.get("goals_for_avg_5", 1.5),
            "home_goals_against_avg_5": home_stats.get("goals_against_avg_5", 1.5), "away_goals_against_avg_5": away_stats.get("goals_against_avg_5", 1.5),
            "home_win_rate": home_stats.get("win_rate", 0.5), "away_win_rate": away_stats.get("win_rate", 0.5),
            "h2h_avg_goals": avg.get("h2h_avg_goals", 2.5), "h2h_home_wins": avg.get("h2h_home_wins", 0.5),
            "is_home": 1,
        })
    return pd.DataFrame(rows)


def predict_with_ensemble(model_dict, X, task="binary"):
    if not isinstance(model_dict, dict) or "xgb" not in model_dict or "lgbm" not in model_dict:
        raise TypeError(f"Model bukan ensemble dict: {type(model_dict)}")
    xgb = model_dict["xgb"]
    lgbm = model_dict["lgbm"]
    weights = model_dict.get("weights", {"xgb": 0.5, "lgbm": 0.5})
    xgb_prob = xgb.predict_proba(X)
    lgbm_prob = lgbm.predict_proba(X)
    prob = weights["xgb"] * xgb_prob + weights["lgbm"] * lgbm_prob
    if task == "binary":
        pred = (prob[:, 1] > 0.5).astype(int)
        return prob, pred
    pred = np.argmax(prob, axis=1)
    return prob, pred


def predict_all(models, df: pd.DataFrame) -> list[dict]:
    X = df[[c for c in FEATURES if c in df.columns]].astype(float)
    results = []
    model_1x2 = models["model_1x2_multiclass"]
    prob_1x2, _ = predict_with_ensemble(model_1x2, X, task="multiclass")
    ou_models = {line: models[f"model_ou_{int(line*10)}"] for line in [1.5, 2.5, 3.5] if f"model_ou_{int(line*10)}" in models}
    prob_btts, _ = predict_with_ensemble(models["model_btts"], X, task="binary")

    for i, row in df.iterrows():
        preds = []
        p_h = prob_1x2[i][0]
        p_d = prob_1x2[i][1]
        p_a = prob_1x2[i][2]
        choices = [("Home Win", p_h, row.get("odds_home")), ("Draw", p_d, row.get("odds_draw")), ("Away Win", p_a, row.get("odds_away"))]
        best_label, best_prob, best_odds = max(choices, key=lambda x: x[1])
        preds.append({"market": "1X2", "prediction": best_label, "confidence": round(float(best_prob), 4), "odds": float(best_odds) if pd.notna(best_odds) else None})

        for line in [1.5, 2.5, 3.5]:
            if line not in ou_models:
                continue
            prob_ou, _ = predict_with_ensemble(ou_models[line], X.iloc[[i]], task="binary")
            p_over = prob_ou[0][1]
            p_under = 1 - p_over
            odds_over_col = f"odds_over_{line}"
            odds_under_col = f"odds_under_{line}"
            if p_over >= p_under:
                preds.append({"market": f"O/U {line}", "prediction": "Over", "confidence": round(float(p_over), 4), "odds": row.get(odds_over_col)})
            else:
                preds.append({"market": f"O/U {line}", "prediction": "Under", "confidence": round(float(p_under), 4), "odds": row.get(odds_under_col)})

        p_btts = prob_btts[i][1]
        if p_btts >= 0.5:
            preds.append({"market": "BTTS", "prediction": "Yes", "confidence": round(float(p_btts), 4), "odds": row.get("odds_btts_yes")})
        else:
            preds.append({"market": "BTTS", "prediction": "No", "confidence": round(float(1 - p_btts), 4), "odds": row.get("odds_btts_no")})

        results.append({"match_id": row["match_id"], "league": row["league"], "home": row["home_team"], "away": row["away_team"], "date": row.get("date"), "predictions": preds})
    return results


def filter_and_classify(results: list[dict]) -> tuple[list, list, list]:
    tier_s, tier_a, tier_b = [], [], []
    match_counts = defaultdict(int)
    league_counts = defaultdict(int)
    for match in results:
        liga = match["league"]
        bobot = LEAGUE_WEIGHTS.get(liga, 0.80)
        for p in match["predictions"]:
            conf = p["confidence"]
            odds = p["odds"]
            if odds is None or pd.isna(odds) or odds <= 1:
                continue
            value = conf * odds - 1
            skor_dasar = hitung_skor(conf, value)
            skor_final = skor_dasar * bobot
            if skor_final < MIN_WEIGHTED_SCORE:
                continue
            tier = klasifikasi_tier(conf, value)
            if skor_final >= TIER_S["skor"] * 1.1 and conf >= TIER_S["conf"]:
                tier = "S"
            elif skor_final >= TIER_A["skor"] * 1.1 and conf >= TIER_A["conf"]:
                tier = "A"
            elif tier == "X":
                tier = "B"
            item = {
                "match_id": match["match_id"], "league": liga, "home": match["home"], "away": match["away"], "date": match.get("date"),
                "market": p["market"], "prediction": p["prediction"], "confidence": conf, "odds": float(odds), "value": float(value),
                "skor": float(skor_final), "skor_dasar": float(skor_dasar), "bobot_liga": float(bobot), "stake": stake_untuk_tier(tier),
            }
            match_key = (match["match_id"], p["market"])
            if match_counts[match["match_id"]] >= MAX_MARKETS_PER_MATCH:
                continue
            if league_counts[liga] >= MAX_MARKETS_PER_LEAGUE:
                continue
            match_counts[match["match_id"]] += 1
            league_counts[liga] += 1
            if tier == "S":
                tier_s.append(item)
            elif tier == "A":
                tier_a.append(item)
            elif tier == "B":
                tier_b.append(item)
    tier_s.sort(key=lambda x: x["skor"], reverse=True)
    tier_a.sort(key=lambda x: x["skor"], reverse=True)
    tier_b.sort(key=lambda x: x["skor"], reverse=True)
    return tier_s, tier_a, tier_b


def render_txt(tier_s, tier_a, tier_b, total_scan: int) -> str:
    lines = ["=" * 60, f"   PREDIKSI PARLAY — {datetime.now().strftime('%d %B %Y')}", f"   Total scan: {total_scan} laga", f"   Lolos seleksi: {len(tier_s)+len(tier_a)+len(tier_b)} market", "=" * 60]

    def render_tier(items, title, emoji):
        if not items:
            lines.append(f"{emoji} {title} — (kosong)")
            lines.append("")
            return
        lines.append(f"{emoji} {title} ({len(items)} market)")
        lines.append("-" * 60)
        for i, it in enumerate(items, 1):
            lines.append(f"{i}. [{it['league']}] {it['home']} vs {it['away']}")
            lines.append(f"   Tanggal   : {format_match_date(it.get('date'))}")
            lines.append(f"   Match ID  : {it['match_id']}")
            lines.append(f"   Market    : {it['market']} — {it['prediction']}")
            lines.append(f"   Conf      : {it['confidence']:.0%} | Odds: {it['odds']} | Value: {it['value']:+.2%}")
            lines.append(f"   Skor      : {it['skor']:.4f} (dasar {it['skor_dasar']:.4f} × bobot {it['bobot_liga']:.2f})")
            lines.append(f"   Stake     : Rp {it['stake']:,}")
            lines.append("")
    render_tier(tier_s, "LAPIS S — PREMIUM", "🏆")
    render_tier(tier_a, "LAPIS A — LOLOS", "✅")
    render_tier(tier_b, "LAPIS B — HIBURAN", "🎲")
    total_stake = min(sum(it["stake"] for it in tier_s + tier_a + tier_b), MAX_DAILY_STAKE)
    lines.append("=" * 60)
    lines.append(f"TOTAL STAKE: Rp {total_stake:,} (maks Rp {MAX_DAILY_STAKE:,})")
    lines.append("Catatan: Skor sudah dibobot per liga. AI hanya penyaring awal.")
    lines.append("=" * 60)
    return "\n".join(lines)


def apply_daily_limit(all_items, max_daily=MAX_DAILY_STAKE):
    """Enforce limit harian di tracking dan output."""
    all_items_sorted = sorted(all_items, key=lambda x: x["skor"], reverse=True)
    running_total = 0
    limited_items = []
    for it in all_items_sorted:
        stake = int(it.get("stake", 0))
        if running_total + stake <= max_daily:
            limited_items.append(it)
            running_total += stake
    return limited_items


def save_tracking(tier_s, tier_a, tier_b):
    all_items = apply_daily_limit(tier_s + tier_a + tier_b, MAX_DAILY_STAKE)
    if not all_items:
        return
    today = datetime.now().strftime("%Y-%m-%d")
    rows = []
    for i, it in enumerate(all_items, 1):
        rows.append({
            "id": i, "tanggal": today, "tanggal_match": format_match_date(it.get("date")), "match_id": it["match_id"], "liga": it["league"],
            "home": it["home"], "away": it["away"], "market": it["market"], "prediksi": it["prediction"], "odds": it["odds"],
            "stake_saya": it["stake"], "confidence": it["confidence"], "value": it["value"], "skor": it["skor"], "hasil": "",
            "profit": 0,
        })
    df = pd.DataFrame(rows)
    out = path("data/processed/tracking_pending.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def main():
    print("=== PREDIKSI HARI INI (dengan 1X2 multiclass + O/U terpisah) ===")
    models = load_models()
    print("[OK] Models loaded")
    is_valid, errors = validate_fixtures()
    if not is_valid:
        print("[ERROR] Fixtures tidak valid:")
        for e in errors:
            print(f"  - {e}")
        return
    fixtures = load_fixtures()
    fixtures = filter_fixtures_by_whitelist(fixtures)
    hist = load_historical_stats()
    df = build_features(fixtures, hist)
    results = predict_all(models, df)
    tier_s, tier_a, tier_b = filter_and_classify(results)
    out_json = path("data/processed/predictions_today.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "model_version": models.get("model_1x2_multiclass").get("model_version", "unknown") if isinstance(models.get("model_1x2_multiclass"), dict) else "unknown",
        "model_metadata": load_model_metadata(),
        "total_scan": len(results),
        "tier_s": tier_s,
        "tier_a": tier_a,
        "tier_b": tier_b,
    }
    out_json.write_text(json.dumps(payload, indent=2, default=str))
    txt = render_txt(tier_s, tier_a, tier_b, len(results))
    out_txt = path("data/processed/predictions_today.txt")
    out_txt.write_text(txt, encoding="utf-8")
    save_tracking(tier_s, tier_a, tier_b)
    print(txt)


if __name__ == "__main__":
    main()
