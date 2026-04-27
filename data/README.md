# Data Documentation

## Data Overview

| Property | Description |
|----------|-------------|
| Sample Size | 622 respondents |
| Survey Method | Online questionnaire |
| Target Population | Chinese consumers considering EV purchase |

---

## Data Files

| File | Description |
|------|-------------|
| `data.csv` | Main survey dataset (Chinese questionnaire) |
| `data.xlsx` | Excel backup with additional sheets |

---

## Variable Definitions

### Dependent Variable

| Variable | Description | Scale |
|----------|-------------|-------|
| `支付意愿` | Willingness to pay premium for intelligent driving features | 1-5 (Ordinal) |

### Independent Variables

| Variable | Description | Survey Questions |
|----------|-------------|-----------------|
| `技术信任` (Tech Trust) | Trust in intelligent driving technology | Q15, Q16 (averaged) |
| `感知价值` (Perceived Value) | Perceived value of adaptive cruise control | Q22 |

### Mediation Variables

| Variable | Description | Survey Question |
|----------|-------------|-----------------|
| `驾驶乐趣` (Driving Pleasure) | Perceived enhancement of driving enjoyment | Q18 |
| `出行效率` (Travel Efficiency) | Perceived improvement in travel efficiency | Q19 |

### Control Variables

| Variable | Description | Scale |
|----------|-------------|-------|
| Gender | 1=Male, 2=Female | 1-2 |
| Age | 1-5 (Under 25 to Over 56) | 1-5 |
| Education | 1-4 (High school to Graduate) | 1-4 |
| Income | Monthly income range | 1-5 |
| Driving Experience | Years of driving license | 1-5 |
| Driving Frequency | Times per week | 1-5 |

---

## Survey Questions Mapping

| Variable | Original Chinese Question |
|----------|--------------------------|
| `技术信任` | Q15: 您认为智能驾驶功能对新能源汽车很重要? |
| | Q16: 您认为智能驾驶功能可以提高驾驶安全性? |
| `感知价值` | Q22: 您愿意为智能驾驶的自适应巡航功能影响购买意愿? |
| `驾驶乐趣` | Q18: 您认为智能驾驶功能可以提升驾驶乐趣? |
| `出行效率` | Q19: 您认为智能驾驶功能可以提高出行效率? |
| `支付意愿` | Q21: 您愿意为智能驾驶功能支付溢价? |

---

## Data Quality Notes

- **Missing Values**: Handled via listwise deletion
- **Outliers**: No extreme values detected after inspection
- **Multicollinearity**: VIF=22.3 between tech trust and perceived value
  - **Solution**: Selective control variable strategy (Angrist & Pischke, 2009)

---

## Data Access

The raw data file (`data.csv`) is not included in the repository due to privacy concerns.

**To obtain the data:**
1. Collect survey responses using the original questionnaire
2. Anonymize all personal identifiers
3. Place the cleaned `data.csv` in `data/raw/`

---

## Citation

If you use this dataset in your research, please cite the original survey design and acknowledge the data collection effort.
