# EV Purchase Intention: Impact of Autonomous Driving Features

> **Competition**: 15th National College Student Market Research Competition  
> **Timeline**: December 2024 - March 2025  
> **License**: MIT

---

## Overview

This project investigates how autonomous driving features influence consumers' purchase intention for New Energy Vehicles (NEVs). Based on 622 survey responses, we employ:

- **Ordered Logit Regression** — handle ordinal dependent variables
- **Bootstrap Mediation Analysis** — test indirect effects through driving enjoyment and travel efficiency
- **Heterogeneity Analysis** — identify group differences across demographics
- **ML + SHAP** — validate results with machine learning interpretability

## Key Findings

| Hypothesis | Path | Result | Conclusion |
|------------|------|--------|-----------|
| H1 | Tech Trust → Purchase Intention | OR=2.77, p<0.001 | ✅ Supported |
| H2 | Perceived Value → Purchase Intention | OR=2.20, p<0.001 | ✅ Supported |
| H3 | Tech Trust → Enjoyment → Intention | 35% mediation | ✅ Partial mediation |
| H4 | Value → Efficiency → Intention | Significant | ✅ Partial mediation |

## Key Improvements

| Issue | Solution | Impact |
|-------|----------|--------|
| Multicollinearity (VIF=22.3) | Selective control strategy | Core variables remain significant |
| Heterogeneous effects | LR test by demographic groups | Identified key segments |
| Non-linear relationships | ML + SHAP validation | Robust results confirmed |

## Data Sources

- **Sample**: 622 consumers surveyed in China (2024-2025)
- **Variables**: 39 items covering tech trust, perceived value, driving enjoyment, travel efficiency, demographics
- **Location**: `data/raw/data.csv`

## Project Structure

```
ev-purchase-intention/
├── src/
│   ├── ordered_logit.py      # Ordered logit regression + VIF
│   ├── mediation.py          # Bootstrap mediation analysis
│   ├── heterogeneity.py      # Group comparison + LR test
│   └── ml_shap.py           # ML prediction + SHAP
├── notebooks/
│   └── 01_ev_purchase_intention.ipynb  # Complete analysis workflow
├── figures/                  # Output visualizations
├── data/raw/                 # Raw survey data
├── main.py                   # CLI entry point
├── README.md
├── DATA.md
├── requirements.txt
└── LICENSE
```

## Quick Start

```bash
# Clone and install dependencies
git clone https://github.com/YOUR_USERNAME/ev-purchase-intention.git
cd ev-purchase-intention
pip install -r requirements.txt

# Run all analyses
python main.py

# Or open notebook
jupyter notebook notebooks/01_ev_purchase_intention.ipynb
```

## Methodology

### 1. Ordered Logit Regression

Custom implementation of the proportional odds model. Handles the ordinal nature of purchase intention (1-5 Likert scale).

```python
from src.ordered_logit import OrderedLogitRegression

model = OrderedLogitRegression()
model.fit(X, y)
results = model.summary()
```

### 2. Mediation Analysis (Bootstrap)

5 mediation paths tested using 5,000 bootstrap samples:

- Tech Trust → Driving Enjoyment → Purchase Intention
- Tech Trust → Travel Efficiency → Purchase Intention
- Perceived Value → Driving Enjoyment → Purchase Intention
- Perceived Value → Travel Efficiency → Purchase Intention
- Tech Trust → Perceived Value → Purchase Intention

### 3. Heterogeneity Analysis

LR test for group differences across:
- Age groups (Youth / Middle-aged / Senior)
- Income levels (Low / Middle / High)
- Driving experience (Novice / Experienced / Expert)

### 4. ML + SHAP

Random Forest classifier with SHAP values for:
- Model performance validation
- Feature importance ranking
- Prediction interpretability

## Limitations & Future Work

- **Cross-sectional data**: Causal inference limited
- **Sample scope**: Single country, urban respondents
- **Future directions**:
  - Longitudinal tracking of attitude changes
  - Field experiment with EV test drives
  - Cross-cultural comparison

## Tech Stack

| Category | Tools |
|----------|-------|
| Data | pandas, numpy |
| Statistics | scipy, statsmodels |
| ML | scikit-learn, shap |
| Visualization | matplotlib, seaborn |

## References

- Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics*
- Hayes, A. F. (2017). *Introduction to Mediation, Moderation, and Conditional Process Analysis*
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions
