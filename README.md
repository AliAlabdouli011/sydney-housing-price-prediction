# Sydney Housing Price Prediction and Decision Support System

SIT307/SIT720 8.1 Distinction Task — Machine Learning Mini Project.

Predicts residential sale prices across three Sydney suburbs (Mosman, Parramatta, Campbelltown) using a Decision Tree Regressor trained on 104 real properties collected from Domain.com.au sold listings.

## Contents

| File | Description |
|---|---|
| `analysis.ipynb` | Full analysis notebook: data collection discussion, EDA, feature engineering, model comparison (Linear Regression, Decision Tree, KNN), error analysis, deployment |
| `raw_collected.csv` | The 104-property dataset collected for this project |
| `app.py` | Streamlit web app for interactive price prediction |
| `model.joblib` | Trained Decision Tree model (`max_depth=5`) |
| `model_columns.joblib` | Exact feature column order used at training time |
| `suburb_options.joblib` | Valid suburb category values |
| `property_type_options.joblib` | Valid property type category values |
| `screenshots/` | Verified screenshots of the deployed app in use |
| `SIT307_8.1_DistinctionTask_Report.pdf` | Full project report |

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running the notebook

```bash
jupyter notebook analysis.ipynb
```

Requires `raw_collected.csv` to be present in the same folder.

## Running the app

```bash
streamlit run app.py
```

Requires `model.joblib`, `model_columns.joblib`, `suburb_options.joblib`, and `property_type_options.joblib` to be present in the same folder as `app.py`. Opens at `http://localhost:8501`.

## Model summary

- **Algorithm**: Decision Tree Regressor, `max_depth=5`
- **Training data**: 83 of 104 collected properties (80:20 train/test split)
- **Test performance**: R²=0.846, RMSE=$616,121, MAE=$346,075
- **Known limitations**: least reliable for Mosman properties (small, heterogeneous training sample); produces identical predictions for structurally different properties that fall into the same tree leaf. See the report for full details.

## Data source

Sale prices, dates, and property details manually collected from [Domain.com.au](https://www.domain.com.au) sold-listing pages, 14-15 September 2026.
