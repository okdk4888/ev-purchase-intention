"""
EV Purchase Intention — Ordinal Logistic Regression Module

Uses OrderedModel from statsmodels to estimate the effect of tech trust
and perceived value on EV purchase intention (ordinal outcome).
"""

import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

DATA_PATH = 'data/raw/data.csv'

Q_COLS = {
    'Y': '21.您愿意为智能驾驶功能支付溢价?',
    'tech_trust_q1': '15.您认为智能驾驶功能对新能源汽车很重要?',
    'tech_trust_q2': '16.您认为智能驾驶功能可以提高驾驶安全性?',
    'perceived_value': '22.您愿意为智能驾驶的自适应巡航功能影响购买意愿?',
}

CONTROL_COLS = {
    'gender': '1.您的性别是?',
    'age': '2.您的年龄是?',
    'education': '4.您的最高学历是?',
    'income': '6.您的月收入范围是?',
    'driving_exp': '9.您的驾龄是?',
    'driving_freq': '12.您每周驾驶的频率是?',
}

MAIN_VARS = ['tech_trust', 'perceived_value']


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

def load_data(data_path=DATA_PATH):
    df = pd.read_csv(data_path, encoding='utf-8')
    df['Y'] = df[Q_COLS['Y']]
    df['tech_trust'] = df[[Q_COLS['tech_trust_q1'], Q_COLS['tech_trust_q2']]].mean(axis=1)
    df['perceived_value'] = df[Q_COLS['perceived_value']]
    for new_name, col in CONTROL_COLS.items():
        df[new_name] = df[col]
    return df


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def p_stars(p):
    if np.isnan(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    if p < 0.1:
        return '+'
    return ''


def run_vif_test(X, var_names):
    vif_data = []
    for i, name in enumerate(var_names):
        vif = variance_inflation_factor(X.values.astype(float), i)
        vif_data.append({'Variable': name, 'VIF': round(vif, 2)})
    return pd.DataFrame(vif_data)


# ──────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────

def run_ordered_logit_analysis(df, output_dir='figures'):
    os.makedirs(output_dir, exist_ok=True)

    all_vars = MAIN_VARS + list(CONTROL_COLS.keys())
    X = df[all_vars].copy()
    y = df['Y'].copy()
    valid = ~(X.isnull().any(axis=1) | y.isnull())
    X_v = X[valid].reset_index(drop=True)
    y_v = y[valid].reset_index(drop=True)

    var_names_cn = ['Tech Trust', 'Perceived Value']

    print('=' * 55)
    print('Ordinal Logistic Regression')
    print('=' * 55)
    print(f'Valid N = {valid.sum()}')

    # VIF check
    print('\n--- VIF ---')
    vif_df = run_vif_test(X_v, all_vars)
    print(vif_df.to_string(index=False))

    # Model 1: core vars only
    print('\n--- Model 1: Core Variables ---')
    X1 = X_v[MAIN_VARS]
    model1 = OrderedModel(y_v.values, X1.values, distr='logit')
    res1 = model1.fit(method='bfgs', disp=False)
    print(f'Pseudo R2 = {res1.prsquared:.4f}')
    for i, name in enumerate(var_names_cn):
        b, p = res1.params[i], res1.pvalues[i]
        print(f'  {name}: B={b:.4f}, OR={np.exp(b):.4f}, p={p:.4f} {p_stars(p)}')

    # Model 2: with controls
    print('\n--- Model 2: With Controls ---')
    model2 = OrderedModel(y_v.values, X_v.values, distr='logit')
    res2 = model2.fit(method='bfgs', disp=False)
    print(f'Pseudo R2 = {res2.prsquared:.4f}')
    for i, name in enumerate(var_names_cn):
        b, p = res2.params[i], res2.pvalues[i]
        print(f'  {name}: B={b:.4f}, OR={np.exp(b):.4f}, p={p:.4f} {p_stars(p)}')

    # OR bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ors = [np.exp(res2.params[i]) for i in range(2)]
    pvals = [res2.pvalues[i] for i in range(2)]
    colors = ['#2ecc71', '#3498db']
    bars = ax.barh(var_names_cn, ors, color=colors, alpha=0.85, height=0.5)
    ax.axvline(x=1, color='gray', linestyle='--', linewidth=1.5)
    for bar, or_val, p_val in zip(bars, ors, pvals):
        ax.text(or_val + 0.05, bar.get_y() + bar.get_height() / 2,
                f'{or_val:.2f}{p_stars(p_val)}', va='center', fontsize=11, fontweight='bold')
    ax.set_xlabel('Odds Ratio (OR)', fontsize=12)
    ax.set_title('Impact of Tech Trust & Perceived Value on Purchase Intention', fontsize=13)
    ax.set_xlim(0, max(ors) + 0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'odds_ratios.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'\nSaved: {os.path.join(output_dir, "odds_ratios.png")}')

    return {'model1': res1, 'model2': res2, 'vif': vif_df, 'n_obs': valid.sum()}


if __name__ == '__main__':
    df = load_data()
    run_ordered_logit_analysis(df)
