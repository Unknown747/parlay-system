"""
load_manual_data.py — Helper untuk load manual fixtures dan bets.
Dipakai oleh fetch_all.py dan predict.py.
"""
import os
from pathlib import Path

import pandas as pd

from config import path


def load_manual_fixtures():
    """
    Load manual fixtures untuk backfill data lama yang terlewat.
    File: data/raw/manual_fixtures.csv
    Kolom: match_id, league, date, home_team, away_team, odds_home, odds_draw, odds_away, odds_over_2_5, odds_under_2_5
    """
    if not os.getenv("ALLOW_MANUAL_FIXTURES", "True").lower() == "true":
        return pd.DataFrame()
    
    manual_path = path("data/raw/manual_fixtures.csv")
    if not manual_path.exists():
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(manual_path)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        print(f"[OK] Loaded {len(df)} manual fixtures from {manual_path}")
        return df
    except Exception as e:
        print(f"[ERROR] Manual fixtures load error: {e}")
        return pd.DataFrame()


def load_manual_bets():
    """
    Load manual bets yang user temukan sendiri.
    File: data/raw/manual_bets.csv
    Kolom: tanggal, liga, home_team, away_team, market, prediksi, odds, confidence, catatan
    """
    if not os.getenv("ALLOW_MANUAL_BETS", "True").lower() == "true":
        return []
    
    manual_path = path("data/raw/manual_bets.csv")
    if not manual_path.exists():
        return []
    
    try:
        df = pd.read_csv(manual_path)
        results = []
        for _, row in df.iterrows():
            liga = str(row.get("liga", "UNK")).strip()
            home = str(row.get("home_team", "")).strip()
            away = str(row.get("away_team", "")).strip()
            date = pd.to_datetime(row.get("tanggal"), errors="coerce")
            date_str = date.strftime("%Y%m%d") if pd.notna(date) else "00000000"
            
            match_id = f"{liga}_{date_str}_{home.replace(' ', '')}_{away.replace(' ', '')}"
            
            results.append({
                "match_id": match_id,
                "league": liga,
                "home": home,
                "away": away,
                "date": date,
                "predictions": [{
                    "market": str(row.get("market", "1X2")).strip(),
                    "prediction": str(row.get("prediksi", "")).strip(),
                    "confidence": float(row.get("confidence", 0.55)),
                    "odds": float(row.get("odds", 1.5)),
                }],
                "source": "manual_bet",
                "catatan": str(row.get("catatan", "")).strip(),
            })
        print(f"[OK] Loaded {len(results)} manual bets from {manual_path}")
        return results
    except Exception as e:
        print(f"[ERROR] Manual bets load error: {e}")
        return []
