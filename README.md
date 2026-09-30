# 🏡 Airbnb Price Prediction — Seattle

A data-driven project that explores **what makes an Airbnb listing in Seattle more expensive**, using **Exploratory Data Analysis (EDA)**, **review sentiment analysis**, and **regression modeling**.
The project identifies the key factors influencing nightly prices and builds interpretable models to help hosts price their listings.

---

## 📊 Project Overview
The analysis is organised around one question, broken into three sub-problems, followed by a modeling stage:

| Notebook | Question |
|----------|----------|
| `01_exploratory_data_analysis.ipynb` | Which **property features and amenities** affect price? (room type, property type, bedrooms, amenities) |
| `02_exploratory_data_analysis.ipynb` | Are there **locations/neighbourhoods** in Seattle where listings fetch higher prices? |
| `03_exploratory_data_analysis.ipynb` | Do the **listing summary text and review sentiment** affect price? |
| `machine_learning.ipynb` | Can we **predict price** from the factors identified in the EDA? |

---

## ⚙️ Key Features
- **Comprehensive EDA** on ~3,800 Seattle listings: price distributions by room type, property type, bedrooms and neighbourhood, plus amenity and summary word clouds.
- **Sentiment Analysis:** Scored ~85k guest reviews with NLTK's *VADER* analyzer (English reviews filtered with `langdetect`) to test whether sentiment relates to price.
- **Data Preprocessing & Feature Engineering:** Price cleaning, one-hot encoding of room/property types, grouping rare property types (< 30 listings), and turning the free-text amenities field (41 distinct amenities) into 19 binary amenity features.
- **Regression Modeling:** Compared *Linear Regression, Ridge, Lasso, Random Forest, XGBoost and CatBoost*. Ridge/Lasso regularization strength was chosen with 5-fold cross-validation on the training set.
- **Evaluation:** 80/20 train/test split **and** 10-fold cross-validation, with feature scaling fitted inside each training fold to avoid leakage.
- **Explainability:** Feature importance and *TreeInterpreter* to break individual predictions down into per-feature contributions.

**Predictors:** room type, property type, bedrooms, number of reviews, amenities
**Target:** nightly price (USD)

---

## 📈 Model Comparison

| Model | Train R² | Test R² (80/20 split) | Test RMSE | 10-fold CV R² | 10-fold CV RMSE |
|-------|------|------|------|------|------|
| Linear Regression | 0.513 | 0.543 | $60.9 | 0.508 | $62.4 |
| Ridge Regression | 0.512 | 0.543 | $60.9 | 0.509 | $62.4 |
| Lasso Regression | 0.512 | 0.544 | $60.9 | 0.509 | $62.4 |
| Random Forest | 0.560 | 0.528 | $61.9 | 0.511 | $62.5 |
| XGBoost | 0.749 | 0.531 | $61.7 | 0.510 | $62.0 |
| CatBoost | 0.728 | 0.550 | $60.4 | 0.509 | $62.2 |

*Train/test split uses `random_state=42`; cross-validation uses `KFold(n_splits=10, shuffle=True, random_state=100)`; all models use fixed random seeds, so results are reproducible.*

- **All six models perform about the same** under 10-fold cross-validation (R² ≈ 0.51, RMSE ≈ $62, against a median price of about $100). CatBoost's slightly better score on the single test split does not hold up under cross-validation.
- **More complex models do not help.** XGBoost and CatBoost reach R² ≈ 0.73–0.75 on the training set but only ≈ 0.51 on unseen data, so they overfit rather than beat Linear Regression.
- R² of about 0.5 shows that these structural features explain roughly half of the price variation. The rest likely comes from factors not modeled here, such as exact location, seasonality, listing quality and host pricing strategy.

---

## 🔍 Key Insights
- **Room type is the biggest price driver.** Entire homes/apartments (median $137) cost about twice as much as private rooms ($68); shared rooms are cheapest ($40).
- **Price rises with the number of bedrooms** (median $87 for 1 bedroom to $548 for 6). Bedrooms is the most important predictor in every model.
- **Location matters, but less than room type** (neighbourhood explains ~11% of price variance vs ~23% for room type). **Magnolia**, **Downtown** and **Queen Anne** are the most expensive districts, and **Belltown** and **West Queen Anne** combine many listings with high prices.
- **Premium amenities** such as a hot tub/pool, gym, elevator, fireplace and air conditioning contribute to higher predicted prices, while common amenities (internet, kitchen, washer/dryer) are in almost every listing.
- Summaries of expensive listings use **"view"** (41% vs 7% of the cheapest listings) and **"modern"** (21% vs 0%) much more often.
- **Reviews do not affect price.** About 96% of reviews are strongly positive, so review sentiment has no correlation with price (r ≈ 0), and the number of reviews has only a very weak negative correlation (r ≈ -0.12).

---

## 🧠 Tech Stack
- **Language:** Python 3.12
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, CatBoost, TreeInterpreter, NLTK (VADER), langdetect, WordCloud, Matplotlib, Seaborn
- **Environment:** Jupyter Notebook / VS Code, `uv`
- **Version Control:** Git & GitHub

---

## 📂 Project Structure
```
Airbnb_Seattle/
│
├── code/                                   # Jupyter notebooks (run from this folder)
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_exploratory_data_analysis.ipynb
│   └── machine_learning.ipynb
├── data/                                   # Dataset CSVs (not tracked in git, see below)
│   └── data_columns.txt                    # Column descriptions
├── pyproject.toml / uv.lock                # Dependencies (uv)
├── requirements.txt                        # Dependencies (pip)
└── README.md
```

---

## 🚀 Getting Started

1. **Get the data.** Download the [Seattle Airbnb Open Data](https://www.kaggle.com/datasets/airbnb/seattle) from Kaggle and place `listings.csv`, `reviews.csv` and `calendar.csv` in the `data/` folder.
2. **Install dependencies.** The tree visualization in the modeling notebook also needs the [Graphviz](https://graphviz.org/download/) system package (`brew install graphviz` on macOS). Then install the Python packages using either
   ```bash
   uv sync
   ```
   or
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the notebooks** from inside the `code/` folder. They read data from `../data/`. The NLTK resources are downloaded automatically into `code/nltk_data/` on first run, and notebook 03 writes `polarity_reviews.csv` there.

---

## 🔮 Future Enhancements
- Add geospatial analysis (latitude/longitude) with **Folium / Plotly Maps** for location-based pricing.
- Use the `calendar.csv` data to model **seasonal** price variation.
- Deploy a web app (Streamlit / Flask) for interactive price predictions.
- Extend the text analysis with **topic modeling** of reviews and listing descriptions.

---

## ✨ Author
**Rutika Avinash Kadam**
📧 [rutikakadam2727@gmail.com](mailto:rutikakadam2727@gmail.com)
🔗 [LinkedIn](https://linkedin.com/in/rutika-kadam) | [GitHub](https://github.com/RutikaKadam10)
