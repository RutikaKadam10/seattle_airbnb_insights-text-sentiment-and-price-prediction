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
- **Data Preprocessing & Feature Engineering:** Price cleaning, one-hot encoding of room/property types, grouping rare property types (< 30 listings), and turning the free-text amenities field into binary amenity features.
- **Regression Modeling:** Compared *Linear Regression, Ridge, Lasso, Random Forest, XGBoost and CatBoost*, with hyperparameters tuned via `GridSearchCV`.
- **Evaluation:** 80/20 train/test split **and** 10-fold cross-validation.
- **Explainability:** Feature importance and *TreeInterpreter* to break individual predictions down into per-feature contributions.

**Predictors:** room type, property type, bedrooms, number of reviews, amenities
**Target:** nightly price (USD)

---

## 📈 Model Comparison

| Model | Test R² (80/20 split) | Test MSE (80/20 split) | 10-fold CV R² | 10-fold CV MSE |
|-------|------|------|------|------|
| Linear Regression | 0.525 | 3855 | 0.496 | 4098 |
| Ridge Regression | 0.525 | 3855 | 0.459 | 4470 |
| Lasso Regression | 0.525 | 3854 | 0.472 | 4332 |
| Random Forest | **0.531** | **3807** | **0.519** | **3953** |
| XGBoost | 0.502 | 4039 | 0.494 | 4059 |
| CatBoost | 0.470 | 4301 | 0.499 | 4011 |

*Train/test split uses `random_state=42`; cross-validation uses `KFold(n_splits=10, shuffle=True, random_state=100)`.*

- **Random Forest** performs best on both the held-out test set and 10-fold cross-validation, so it was used for the feature-importance and TreeInterpreter analysis.
- **XGBoost and CatBoost overfit.** They reach R² ≈ 0.68–0.69 on the training set but fall to 0.47–0.50 on unseen data.
- R² of about 0.5 shows that these structural features explain roughly half of the price variation. The rest likely comes from factors not modeled here, such as exact location, seasonality, listing quality and host pricing strategy.

---

## 🔍 Key Insights
- **Entire homes/apartments** fetch the highest prices, followed by private rooms, then shared rooms.
- **Price rises with the number of bedrooms.** Bedrooms is the strongest predictor in the Ridge/Lasso models.
- **Location matters.** Neighbourhoods such as **Belltown** and **West Queen Anne** have both many listings and high prices.
- **Premium amenities** such as a hot tub/sauna/pool, gym and elevator contribute to higher predicted prices.
- Summaries of expensive listings often use words like **"view"**, **"modern"** and **"walk"**.
- **Reviews have little impact on price.** Most reviews are overwhelmingly positive, and the number of reviews has only a very weak negative correlation with price.

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
