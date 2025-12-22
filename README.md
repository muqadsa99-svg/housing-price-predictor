# housing-price-predictor
An end-to-end Machine Learning web app that predicts real estate prices using a Stacking Regressor (XGBoost, LightGBM, Lasso, Ridge). Features advanced engineering and Scikit-Learn pipelines.
# 🏡 Advanced Housing Price Prediction

An end-to-end Machine Learning project that predicts residential housing prices using the Ames Housing Dataset. This project moves beyond simple linear regression by implementing a **Stacking Regressor** ensemble technique to minimize error and maximize predictive power.

**[🔴 Live Demo](LINK_TO_YOUR_STREAMLIT_APP_HERE)**

## 🚀 Key Features
* **Ensemble Modeling:** Combines the strengths of **Ridge**, **Lasso**, **XGBoost**, and **LightGBM** using a Stacking Regressor.
* **Advanced Feature Engineering:** * Created interaction features (e.g., `TotalSqFt` combining basement and floor areas).
    * Applied Log-Transformation to target variables to handle skewness.
* **Production-Grade Pipelines:** Utilized Scikit-Learn `Pipeline` and `ColumnTransformer` to prevent data leakage during preprocessing.
* **Interactive Deployment:** Built a user-friendly frontend using **Streamlit** for real-time predictions.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Scikit-Learn, XGBoost, LightGBM, Pandas, NumPy
* **Deployment:** Streamlit Cloud
* **Environment:** Google Colab (Training) -> GitHub (Version Control)

## 📊 Model Architecture
The model uses a Stacking approach where the predictions of base models serve as inputs for a final meta-learner:
1.  **Base Models:** * *XGBoost & LightGBM:* For capturing non-linear patterns.
    * *Lasso & Ridge:* For handling multicollinearity and regularization.
2.  **Meta-Learner:**
    * *Ridge Regression:* Aggregates base predictions to output the final price.

## 📂 Project Structure
```bash
├── app.py                  # Streamlit application entry point
├── house_price_model.pkl   # Trained model pipeline (saved via Joblib)
├── requirements.txt        # Dependencies for deployment
├── sample_input.csv        # Sample data structure for the app
└── README.md               # Project documentation
