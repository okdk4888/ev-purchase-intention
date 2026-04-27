"""
EV Purchase Intention — Heterogeneity Analysis Module

Tests whether regression coefficients differ across demographic subgroups
using likelihood ratio tests.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

DATA_PATH = 'data/raw/data.csv'

Q_COLS = {
    'Y': '21.您愿意为智能驾驶功能支付溢价?',
    'tech_trust_q1': '15.您认为智能驾驶功能对新能源汽车很重要?',
    'tech_trust_q2': '16.您认为智能驾驶功能可以提高驾驶安全性?',
    'perceived_value': '22.您愿意为智能驾驶的自适应巡航功能影响购买意愿?',
    'gender': '1.您的性别是?',
    'age': '2.您的年龄是?',
    'income': '6.您的月收入范围是?',
    'driving_exp': '9.您的驾龄是?',
    'driving_freq': '12.您每周驾驶的频率是?',
}


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

def load_data(data_path=DATA_PATH):
    df = pd.read_csv(data_path, encoding='utf-8')
    df['Y'] = df[Q_COLS['Y']]
    df['tech_trust'] = df[[Q_COLS['tech_trust_q1'], Q_COLS['tech_trust_q2']]].mean(axis=1)
    df['perceived_value'] = df[Q_COLS['perceived_value']]

    for col in ['gender', 'age', 'income', 'driving_exp', 'driving_freq']:
        df[col] = df[Q_COLS[col]]

    df['gender_gp'] = df['gender'].map({1: 'Male', 2: 'Female'})
    df['age_gp'] = df['age'].map({1: '<25', 2: '26-35', 3: '36-45', 4: '46-55', 5: '>55'})
    df['income_gp'] = df['income'].map({1: 'Low', 2: 'Lower-mid', 3: 'Middle', 4: 'Upper-mid', 5: 'High'})
    df['exp_gp'] = df['driving_exp'].map({1: '<1yr', 2: '1-3yr', 3: '3-5yr', 4: '5-10yr', 5: '>10yr'})
    df['freq_gp'] = df['driving_freq'].map({1: 'Rarely', 2: '1-2/wk', 3: '3-4/wk', 4: 'Daily', 5: 'Multiple/d'})

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


def _fit_ologit(y, X):
    try:
        model = OrderedModel(y, X, distr='logit')
        res = model.fit(method='bfgs', disp=False)
        return res.llf, res.params
    except Exception:
        return np.nan, None


def lr_test(llf_full, llf_restricted, df_diff):
    lr_stat = 2 * (llf_full - llf_restricted)
    p_value = 1 - stats.chi2.cdf(lr_stat, df_diff)
    return lr_stat, p_value


# ──────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────

def analyze_group(df, group_var, y_var='Y', x_vars=['tech_trust', 'perceived_value'],
                  control_vars=['age', 'income', 'driving_exp']):
    all_vars = [y_var] + x_vars + control_vars + [group_var]
    df_sub = df[all_vars].dropna()

    groups = df_sub[group_var].unique()
    group_coefs = {}
    llfs = []

    for g in groups:
        g_data = df_sub[df_sub[group_var] == g]
        y = g_data[y_var].values
        X = g_data[x_vars + control_vars].values
        llf, coef = _fit_ologit(y, X)
        if not np.isnan(llf):
            llfs.append(llf)
            group_coefs[g] = coef[0]

    # Unrestricted model
    y_all = df_sub[y_var].values
    X_all = df_sub[x_vars + control_vars].values
    llf_full, _ = _fit_ologit(y_all, X_all)

    # Restricted: sum of group log-likelihoods
    llf_r = sum(llfs) if llfs else np.nan

    df_diff = len(groups) - 1
    lr_stat, p_value = lr_test(llf_full, llf_r, df_diff)

    return {
        'group_var': group_var,
        'groups': list(groups),
        'group_coefs': group_coefs,
        'lr_stat': lr_stat,
        'lr_pvalue': p_value,
        'n_groups': len(groups),
    }


def analyze_heterogeneity(df):
    group_vars = ['gender_gp', 'age_gp', 'income_gp', 'exp_gp', 'freq_gp']
    available = [g for g in group_vars if g in df.columns]

    print('=' * 55)
    print('Heterogeneity Analysis (Likelihood Ratio Test)')
    print('=' * 55)

    results = []
    for gv in available:
        result = analyze_group(df, gv)
        results.append(result)
        sig = '***' if result['lr_pvalue'] < 0.001 else '**' if result['lr_pvalue'] < 0.01 else '*' if result['lr_pvalue'] < 0.05 else ''

        print(f"\n{gv}: groups={result['n_groups']}, LR chi2={result['lr_stat']:.3f}, p={result['lr_pvalue']:.4f} {sig}")
        if result['lr_pvalue'] < 0.1:
            for g, c in result['group_coefs'].items():
                print(f"  {g}: {c:.4f}")

    return results


if __name__ == '__main__':
    df = load_data()
    analyze_heterogeneity(df)
