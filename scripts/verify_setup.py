"""
scripts/verify_setup.py — Verifikasi setup.
Exit 0 kalau OK, 1 kalau ada masalah.
"""
import sys
from pathlib import Path

import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from config import path
from colorama import Fore, Style, init
from tabulate import tabulate

init(autoreset=True)


def check_files() -> list[dict]:
    rows = []
    files = [
        ("config.py", "config"),
        (".env", "env"),
        ("requirements.txt", "req"),
        ("data/processed/matches_normalized.csv", "db"),
    ]
    for f, label in files:
        p = ROOT / f
        rows.append({
            "Cek": f"File {label}",
            "Status": "✓" if p.exists() else "✗",
            "Info": str(p.relative_to(ROOT)) if p.exists() else "TIDAK ADA",
        })
    return rows


def check_env() -> list[dict]:
    rows = []
    env_p = ROOT / ".env"
    if not env_p.exists():
        return [{"Cek": "ENV", "Status": "✗", "Info": ".env tidak ada"}]

    from dotenv import dotenv_values
    env = dotenv_values(env_p)

    for key in ["API_KEY", "TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID"]:
        val = env.get(key, "")
        ok = bool(val) and "your_" not in val
        rows.append({
            "Cek": f"ENV {key}",
            "Status": "✓" if ok else "⚠",
            "Info": "terisi" if ok else "kosong/placeholder",
        })
    return rows


def check_csv() -> list[dict]:
    rows = []
    db = path("data/processed/matches_normalized.csv")
    if db.exists():
        try:
            df = pd.read_csv(db, nrows=5)
            rows.append({
                "Cek": "CSV normalized",
                "Status": "✓",
                "Info": f"{len(df.columns)} kolom",
            })
        except Exception as e:
            rows.append({"Cek": "CSV normalized", "Status": "✗", "Info": str(e)[:50]})
    return rows


def check_leakage() -> list[dict]:
    rows = []
    db = path("data/processed/matches_normalized.csv")
    if not db.exists():
        return rows
    df = pd.read_csv(db, nrows=5)
    leakage_cols = [c for c in df.columns if "seasonpos" in c.lower() or "seasonpoints" in c.lower()]
    rows.append({
        "Cek": "Leakage kolom",
        "Status": "✓" if not leakage_cols else "✗",
        "Info": "bersih" if not leakage_cols else f"ada: {leakage_cols}",
    })
    return rows


def check_models() -> list[dict]:
    rows = []
    required = [
        "model_1x2_multiclass",
        "model_ou_15",
        "model_ou_25",
        "model_ou_35",
        "model_btts",
    ]
    for name in required:
        p = path(f"models/{name}.pkl")
        if not p.exists():
            rows.append({"Cek": f"Model {name}", "Status": "✗", "Info": "TIDAK ADA"})
            continue
        try:
            joblib.load(p)
            rows.append({"Cek": f"Model {name}", "Status": "✓", "Info": "load OK"})
        except Exception as e:
            rows.append({"Cek": f"Model {name}", "Status": "✗", "Info": str(e)[:50]})
    return rows


def main():
    print(f"{Fore.CYAN}═══ VERIFY SETUP ═══{Style.RESET_ALL}\n")

    rows = []
    rows.extend(check_files())
    rows.extend(check_env())
    rows.extend(check_csv())
    rows.extend(check_leakage())
    rows.extend(check_models())

    print(tabulate(rows, headers="keys", tablefmt="grid"))

    n_fail = sum(1 for r in rows if r["Status"] == "✗")
    n_warn = sum(1 for r in rows if r["Status"] == "⚠")

    print()
    if n_fail == 0 and n_warn == 0:
        print(f"{Fore.GREEN}✓ Setup OK.{Style.RESET_ALL}")
        sys.exit(0)
    elif n_fail == 0:
        print(f"{Fore.YELLOW}⚠ {n_warn} warning.{Style.RESET_ALL}")
        sys.exit(0)
    else:
        print(f"{Fore.RED}✗ {n_fail} gagal, {n_warn} warning.{Style.RESET_ALL}")
        sys.exit(1)


if __name__ == "__main__":
    main()
