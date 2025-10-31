# 🏡 Airbnb Price Prediction

A data-driven machine learning project that predicts Airbnb listing prices in Seattle through **Exploratory Data Analysis (EDA)**, **regression modeling**, and **sentiment analysis**.  
The project identifies key factors influencing rental prices and builds interpretable models to help hosts optimize their listings.

---

## 📊 Project Overview
This project focuses on understanding and predicting Airbnb listing prices using a combination of data preprocessing, feature engineering, and advanced regression models.  
It includes a comprehensive analysis of listings data, reviews sentiment, and amenities to uncover what drives property pricing on Airbnb.

---

## ⚙️ Key Features
- **Comprehensive EDA** on Airbnb Seattle dataset to identify trends, correlations, and pricing patterns.  
- **Data Preprocessing:** Handling missing values, encoding categorical features, removing outliers, and feature scaling.  
- **Regression Modeling:** Implemented and compared multiple models —  
  *Linear Regression, Ridge, Lasso, Random Forest, XGBoost, CatBoost*.  
- **Performance Evaluation:**  
  Achieved the best performance with **Random Forest Regression** having  
  **R² ≈ 0.68** and **MSE ≈ 3597**, validated through **10-fold cross-validation**.  
- **Explainability:** Used *Feature Importance* and *TreeInterpreter* to explain predictions.  
- **Visualization:** Created rich visual insights using **Matplotlib** and **Seaborn**.  
- **Sentiment Analysis:** Analyzed Airbnb review text using *VADER* to understand the impact of customer sentiment on pricing.

---

## 🧠 Tech Stack
- **Languages:** Python  
- **Libraries & Tools:** Pandas, NumPy, Scikit-learn, XGBoost, CatBoost, Matplotlib, Seaborn, NLTK  
- **Environment:** Jupyter Notebook / VS Code  
- **Version Control:** Git & GitHub  

---

## 📈 Model Comparison Summary
| Model | R² Score | MSE | Remarks |
|-------|-----------|------|---------|
| Linear Regression | 0.56 | 4853 | Baseline model |
| Ridge Regression | 0.59 | 4510 | Reduced overfitting |
| Lasso Regression | 0.58 | 4591 | Sparse feature selection |
| Random Forest | **0.68** | **3597** | Best performing model |
| XGBoost | 0.66 | 3742 | Strong performance |
| CatBoost | 0.65 | 3820 | Good generalization |

---

## 🔍 Key Insights
- Listings with **entire homes**, **higher bedroom counts**, and **premium amenities** (gym, pool, elevator) have significantly higher prices.  
- Positive **review sentiment** correlates with increased nightly rates.  
- Properties near city centers and landmarks show higher demand and occupancy rates.  

---

## 📂 Project Structure
```
Airbnb_Price_Prediction/
│
├── data/                  # Raw and cleaned datasets
├── notebooks/             # Jupyter notebooks for EDA and modeling
├── images/                # Graphs and visualizations
├── src/                   # Helper scripts (data processing, modeling, evaluation)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🚀 Future Enhancements
- Integrate geospatial analysis using **Folium / Plotly Maps** for location-based pricing.  
- Deploy a web application (Streamlit / Flask) for interactive predictions.  
- Extend sentiment analysis to include **topic modeling** for deeper insights into reviews.  

---

## ✨ Author
**Rutika Avinash Kadam**  
📧 [rutikakadam2727@gmail.com](mailto:rutikakadam2727@gmail.com)  
🔗 [LinkedIn](https://linkedin.com/in/rutika-kadam) | [GitHub](https://github.com/RutikaKadam10)
