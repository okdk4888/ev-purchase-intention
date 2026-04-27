"""
EV Purchase Intention — Mediation Analysis Module

Five mediation paths tested via OLS regression with Bootstrap confidence intervals.
"""

import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

DATA_PATH = 'data/raw/data.csv'
OUTPUT_DIR = 'figures'

Q_COLS = {
    'Y': '21.您愿意为智能驾驶功能支付溢价?',
    'X1': '15.您认为智能驾驶功能对新能源汽车很重要?',
    'X2': '16.您认为智能驾驶功能可以提高驾驶安全性?',
    'X_pv': '22.您愿意为智能驾驶的自适应巡航功能影响购买意愿?',
    'M1': '18.您认为智能驾驶功能可以提升驾驶乐趣?',
    'M2': '19.您认为智能驾驶功能可以提高出行效率?',
    'gender': '1.您的性别是?',
    'age': '2.您的年龄是?',
    'education': '4.您的最高学历是?',
    'income': '6.您的月收入范围是?',
    'driving_exp': '9.您的驾龄是?',
    'driving_freq': '12.您每周驾驶的频率是?',
}

CONTROL_VARS = ['gender', 'age', 'education', 'income', 'driving_exp', 'driving_freq']


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

def load_data(data_path=DATA_PATH):
    df = pd.read_csv(data_path, encoding='utf-8')
    df['Y'] = df[Q_COLS['Y']]
    df['X1'] = df[[Q_COLS['X1'], Q_COLS['X2']]].mean(axis=1)
    df['X_pv'] = df[Q_COLS['X_pv']]
    df['M1'] = df[Q_COLS['M1']]
    df['M2'] = df[Q_COLS['M2']]
    for name, col in [
        ('gender', Q_COLS['gender']),
        ('age', Q_COLS['age']),
        ('education', Q_COLS['education']),
        ('income', Q_COLS['income']),
        ('driving_exp', Q_COLS['driving_exp']),
        ('driving_freq', Q_COLS['driving_freq']),
    ]:
        df[name] = df[col]

    for v in ['Y', 'X1', 'X_pv', 'M1', 'M2']:
        df[v + '_z'] = (df[v] - df[v].mean()) / df[v].std()

    return df[['X1_z', 'X_pv_z', 'M1_z', 'M2_z', 'Y_z'] + CONTROL_VARS].dropna().reset_index(drop=True)


# ──────────────────────────────────────────────
# Mediation
# ──────────────────────────────────────────────

def _ols_fit(Y, X_list, data):
    X = sm.add_constant(data[X_list].values)
    return sm.OLS(data[Y].values, X).fit()


def p_stars(p):
    if np.isnan(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return ''


def mediation_path(X_var, M_var, Y_var, data, X_name, M_name):
    x_idx, m_idx = 1, 2

    res_c = _ols_fit(Y_var, [X_var] + CONTROL_VARS, data)
    c_total = res_c.params[x_idx]
    c_p = res_c.pvalues[x_idx]

    res_a = _ols_fit(M_var, [X_var] + CONTROL_VARS, data)
    a, a_p = res_a.params[x_idx], res_a.pvalues[x_idx]

    res_bc = _ols_fit(Y_var, [X_var, M_var] + CONTROL_VARS, data)
    b, b_p = res_bc.params[m_idx], res_bc.pvalues[m_idx]
    c_direct, c_prime_p = res_bc.params[x_idx], res_bc.pvalues[x_idx]

    indirect = a * b

    # Sobel test
    se_indirect = np.sqrt(a**2 * res_bc.bse[m_idx]**2 + b**2 * res_a.bse[x_idx]**2)
    z_sobel = indirect / se_indirect if se_indirect > 0 else 0
    p_sobel = 2 * (1 - norm.cdf(abs(z_sobel)))

    # Bootstrap
    N = len(data)
    np.random.seed(42)
    boot_indirect = []
    for _ in range(5000):
        idx = np.random.choice(N, N, replace=True)
        bd = data.iloc[idx]
        try:
            ra = _ols_fit(M_var, [X_var] + CONTROL_VARS, bd)
            rb = _ols_fit(Y_var, [X_var, M_var] + CONTROL_VARS, bd)
            boot_indirect.append(ra.params[x_idx] * rb.params[m_idx])
        except Exception:
            pass

    boot_indirect = np.array(boot_indirect)
    ci_low, ci_high = np.percentile(boot_indirect, [2.5, 97.5])
    p_boot = 2 * min(np.mean(boot_indirect >= 0), np.mean(boot_indirect <= 0))
    ratio = indirect / c_total * 100 if c_total != 0 else 0

    if p_boot < 0.05 and c_prime_p < 0.05:
        med_type = "Partial mediation"
    elif p_boot < 0.05 and c_prime_p >= 0.05:
        med_type = "Full mediation"
    else:
        med_type = "Not significant"

    return {
        'X': X_name, 'M': M_name,
        'c': c_total, 'c_p': c_p,
        'a': a, 'a_p': a_p,
        'b': b, 'b_p': b_p,
        'c_prime': c_direct, 'c_prime_p': c_prime_p,
        'indirect': indirect,
        'z_sobel': z_sobel, 'p_sobel': p_sobel,
        'ci_low': ci_low, 'ci_high': ci_high,
        'p_boot': p_boot,
        'ratio': ratio,
        'med_type': med_type,
        'boot_indirect': boot_indirect,
    }


# ──────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────

def run_mediation_analysis(df, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    print('Mediation Analysis (Bootstrap 5000)')
    print(f'Valid N = {len(df)}')

    paths = [
        ('X1_z', 'M1_z', 'Y_z', 'Tech Trust', 'Driving Pleasure'),
        ('X1_z', 'M2_z', 'Y_z', 'Tech Trust', 'Travel Efficiency'),
        ('X_pv_z', 'M1_z', 'Y_z', 'Perceived Value', 'Driving Pleasure'),
        ('X_pv_z', 'M2_z', 'Y_z', 'Perceived Value', 'Travel Efficiency'),
        ('X1_z', 'X_pv_z', 'Y_z', 'Tech Trust', 'Perceived Value'),
    ]

    all_results = {}
    for x_v, m_v, y_v, x_n, m_n in paths:
        key = f"{x_n} -> {m_n}"
        r = mediation_path(x_v, m_v, y_v, df, x_n, m_n)
        all_results[key] = r
        sig = 'sig' if r['p_boot'] < 0.05 else 'n.s.'
        print(f"\n{key}")
        print(f"  c={r['c']:.4f}{p_stars(r['c_p'])}, a={r['a']:.4f}{p_stars(r['a_p'])}, b={r['b']:.4f}{p_stars(r['b_p'])}")
        print(f"  Indirect={r['indirect']:.4f}, Sobel z={r['z_sobel']:.3f}, 95%CI=[{r['ci_low']:.4f},{r['ci_high']:.4f}] {sig}")
        print(f"  {r['med_type']}, ratio={r['ratio']:.1f}%")

    # Bootstrap distribution plot
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    for idx, (key, r) in enumerate(all_results.items()):
        ax = axes[idx // 3, idx % 3]
        ax.hist(r['boot_indirect'], bins=40, color=colors[idx], alpha=0.7, edgecolor='white')
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
        ax.axvline(r['ci_low'], color='orange', linestyle=':', linewidth=1.5)
        ax.axvline(r['ci_high'], color='orange', linestyle=':', linewidth=1.5)
        ax.set_xlabel('Indirect Effect', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(f"{r['X']} -> {r['M']}", fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[1, 2].axis('off')
    plt.suptitle('Bootstrap Distribution of Indirect Effects', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mediation_bootstrap_dist.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {os.path.join(output_dir, 'mediation_bootstrap_dist.png')}")

    # Save results table
    rows = []
    for key, r in all_results.items():
        rows.append({
            'Path': f"{r['X']} -> {r['M']}",
            'c_total': round(r['c'], 4),
            'a': round(r['a'], 4),
            'b': round(r['b'], 4),
            "c_prime": round(r['c_prime'], 4),
            'Indirect': round(r['indirect'], 4),
            'Sobel_z': round(r['z_sobel'], 4),
            'CI_low': round(r['ci_low'], 4),
            'CI_high': round(r['ci_high'], 4),
            'Bootstrap_p': round(r['p_boot'], 4),
            'Ratio': f"{r['ratio']:.1f}%",
            'Type': r['med_type'],
        })
    pd.DataFrame(rows).to_excel(os.path.join(output_dir, 'mediation_results.xlsx'), index=False)
    print(f"Saved: {os.path.join(output_dir, 'mediation_results.xlsx')}")

    return all_results


if __name__ == '__main__':
    df = load_data()
    run_mediation_analysis(df)
