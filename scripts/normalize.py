"""
normalize.py — DEPRECATED: Normalisasi sudah dipindahkan ke train.py
File ini hanya untuk referensi historis.

Pada versi terbaru:
- Scaler di-fit HANYA pada training set (train.py line 95-96)
- Test set ditransform menggunakan scaler yang sudah di-fit (bukan re-fit)
- Scaler disimpan di models/{model_name}_scaler.pkl
- Predict.py memuat scaler dan menerapkannya pada fixtures baru

Jangan gunakan script ini lagi di pipeline utama.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import path


# Koefisien kekuatan liga (relatif terhadap EPL = 1.00)
LEAGUE_COEF = {
    "EPL":          1.00,
    "LaLiga":       0.98,
    "SerieA":       0.96,
    "Bundesliga":   0.95,
    "Ligue1":       0.92,
    "Eredivisie":   0.85,
    "PrimeiraLiga": 0.84,
    "Championship": 0.80,
    "LigaMX":       0.78,
    "Brasileirao":  0.76,
}


def main():
    print("="*70)
    print("NORMALISASI DEPRECATED")
    print("="*70)
    print()
    print("⚠️  Script ini sudah tidak digunakan dalam pipeline production.")
    print()
    print("Normalisasi sudah dipindahkan ke train.py untuk mencegah leakage:")
    print("  1. Scaler di-fit HANYA pada training set")
    print("  2. Test set ditransform tanpa re-fit")
    print("  3. Scaler disimpan bersama model")
    print()
    print("Untuk detail implementasi, lihat:")
    print("  - scripts/train.py (line 93-100)")
    print("  - scripts/predict.py (load scaler)")
    print()
    print("✓ Leakage prevention sekarang GUARANTEED.")
    print()


if __name__ == "__main__":
    main()
