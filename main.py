"""
EV Purchase Intention — Main Entry Point

Analysis pipeline:
  1. Ordinal logistic regression (VIF check, two-model comparison)
  2. Mediation analysis (Bootstrap 5000 iterations, 5 paths)
  3. Heterogeneity analysis (5 demographic dimensions, LR test)
  4. Machine learning + SHAP interpretability

Usage:
    python main.py
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'data.csv')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from src.ordered_logit import load_data as load_data_ologit, run_ordered_logit_analysis
from src.mediation import load_data as load_data_med, run_mediation_analysis
from src.heterogeneity import load_data as load_data_het, analyze_heterogeneity
from src.ml_shap import load_data as load_data_ml, run_ml_shap_analysis


def print_banner():
    print('=' * 55)
    print('Impact of Autonomous Driving on EV Purchase Intention')
    print('=' * 55)
    print()
    print('Modules:')
    print('  1. Ordinal Logistic Regression')
    print('  2. Mediation Analysis')
    print('  3. Heterogeneity Analysis')
    print('  4. ML + SHAP')
    print()


def check_data():
    if not os.path.exists(DATA_PATH):
        print(f'Error: data file not found at {DATA_PATH}')
        print('Expected structure: data/raw/data.csv')
        return False
    return True


def main():
    print_banner()
    if not check_data():
        return

    print(f'Loading data from {DATA_PATH}...')
    df = load_data_ologit(DATA_PATH)
    print(f'N = {len(df)}')
    print()

    # 1. Ordinal logistic regression
    print('=' * 55)
    print('1. Ordinal Logistic Regression')
    print('=' * 55)
    try:
        run_ordered_logit_analysis(df, OUTPUT_DIR)
    except Exception as e:
        print(f'Error: {e}')
    print()

    # 2. Mediation analysis
    print('=' * 55)
    print('2. Mediation Analysis')
    print('=' * 55)
    try:
        df_med = load_data_med(DATA_PATH)
        run_mediation_analysis(df_med, OUTPUT_DIR)
    except Exception as e:
        print(f'Error: {e}')
    print()

    # 3. Heterogeneity analysis
    print('=' * 55)
    print('3. Heterogeneity Analysis')
    print('=' * 55)
    try:
        df_het = load_data_het(DATA_PATH)
        analyze_heterogeneity(df_het)
    except Exception as e:
        print(f'Error: {e}')
    print()

    # 4. ML + SHAP
    print('=' * 55)
    print('4. ML + SHAP Analysis')
    print('=' * 55)
    try:
        df_ml = load_data_ml(DATA_PATH)
        run_ml_shap_analysis(df_ml, OUTPUT_DIR)
    except Exception as e:
        print(f'Error: {e}')
    print()

    print('=' * 55)
    print('Done. Figures saved to:')
    for f in os.listdir(OUTPUT_DIR):
        print(f'  {f}')
    print()


if __name__ == '__main__':
    main()
