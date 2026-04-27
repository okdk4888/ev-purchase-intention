"""
EV Purchase Intention — ML + SHAP Interpretability Module

Uses Random Forest to predict purchase intention class, then applies
SHAP to explain feature importance.
"""

import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

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
    'M1': '18.您认为智能驾驶功能可以提升驾驶乐趣?',
    'M2': '19.您认为智能驾驶功能可以提高出行效率?',
    'gender': '1.您的性别是?',
    'age': '2.您的年龄是?',
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
    df['M1'] = df[Q_COLS['M1']]
    df['M2'] = df[Q_COLS['M2']]
    df['gender'] = df[Q_COLS['gender']]
    df['age'] = df[Q_COLS['age']]
    df['driving_freq'] = df[Q_COLS['driving_freq']]
    return df


# ──────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────

def run_ml_shap_analysis(df, output_dir='figures'):
    os.makedirs(output_dir, exist_ok=True)

    print('=' * 55)
    print('ML + SHAP Analysis')
    print('=' * 55)

    feature_vars = ['tech_trust', 'perceived_value', 'M1', 'M2', 'driving_freq', 'age']
    var_labels = {
        'tech_trust': 'Tech Trust',
        'perceived_value': 'Perceived Value',
        'M1': 'Driving Pleasure',
        'M2': 'Travel Efficiency',
        'driving_freq': 'Driving Frequency',
        'age': 'Age',
    }

    X = df[feature_vars].dropna()
    y = df.loc[X.index, 'Y']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f'N = {len(y_enc)}, classes = {len(le.classes_)}')

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    rf.fit(X, y_enc)

    # 5-fold CV
    cv_acc = cross_val_score(rf, X, y_enc, cv=5, scoring='accuracy').mean()
    cv_f1 = cross_val_score(rf, X, y_enc, cv=5, scoring='f1_macro').mean()

    print(f'\n5-fold CV Accuracy: {cv_acc:.3f}')
    print(f'5-fold CV F1 (macro): {cv_f1:.3f}')

    # SHAP
    try:
        import shap
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X)

        # Use first class (or average across classes)
        if isinstance(shap_values, list):
            sv = shap_values[0]
        else:
            sv = shap_values

        mean_abs_shap = np.abs(sv).mean(axis=0)
        importance_df = pd.DataFrame({
            'variable': feature_vars,
            'label': [var_labels.get(v, v) for v in feature_vars],
            'importance': mean_abs_shap,
        }).sort_values('importance', ascending=False)

        print('\nSHAP Feature Importance')
        for _, row in importance_df.iterrows():
            print(f'  {row["label"]}: {row["importance"]:.4f}')

        # Plots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Bar chart
        sorted_idx = np.argsort(mean_abs_shap)
        axes[0].barh(
            [var_labels.get(feature_vars[i], feature_vars[i]) for i in sorted_idx],
            mean_abs_shap[sorted_idx],
            color='#3498db', alpha=0.8,
        )
        axes[0].set_xlabel('Mean |SHAP Value|')
        axes[0].set_title('SHAP Feature Importance')

        # Beeswarm
        shap.summary_plot(sv, X, feature_names=[var_labels.get(v, v) for v in feature_vars], show=False)
        plt.title('SHAP Value Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'shap_analysis.png'), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'\nSaved: {os.path.join(output_dir, "shap_analysis.png")}')

        shap_success = True
    except ImportError:
        print('\nNote: shap not installed, skipping SHAP analysis')
        shap_values = None
        shap_success = False

    return {
        'accuracy': cv_acc,
        'f1_macro': cv_f1,
        'shap_values': shap_values if shap_success else None,
    }


if __name__ == '__main__':
    df = load_data()
    run_ml_shap_analysis(df)
