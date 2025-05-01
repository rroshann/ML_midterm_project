# 🔍 Sales Playbook Optimization with Machine Learning

This project integrates machine learning into B2B sales strategy to enhance deal outcome predictions, customer segmentation, and data-driven decision-making. Built using three datasets—HubSpot Companies, Deals, and Tickets—this solution leverages XGBoost, clustering techniques, and a Streamlit dashboard to empower sales teams with real-time insights and prioritization tools.

## 📊 Overview

Traditional sales playbooks are often static and reactive. This project builds a dynamic alternative using predictive analytics and classification models, delivering an intelligent, evolving guide to improve B2B deal closures. The final deliverable is a Streamlit dashboard backed by a trained ML model and wrapped in a Dockerized deployment.

## 🧱 Key Components

### 🔎 Data Exploration & Cleaning
- **Companies Dataset**: Cleaned 19,851 company records, removed high-missing columns, and handled outliers.
- **Deals Dataset**: Processed 593 deal records, with extensive feature engineering and imputation.
- **Tickets Dataset**: Narrowed to 79 observations, used mainly for EDA and exploratory visualizations.

### 🧠 Machine Learning Models
Trained and compared:
- **Random Forest** (AUC 0.997)
- **XGBoost** (AUC 1.000)
- **AdaBoost**, **KNN**, **Decision Tree**, **Logistic Regression**

XGBoost was selected as the final model due to perfect performance on the test set and robust generalization.

### 🧮 Feature Engineering
- Custom fields for revenue buckets, tech stack indicators, deal size categories.
- Categorical encoding, scaling, and imputation strategies ensured clean and leak-proof features.

### 🧭 Customer Segmentation
Used KMeans to group companies by revenue, engagement, and age into:
- **High-Value Clients**
- **Active Clients**
- **Low-Value Clients**

These segments inform prioritization strategies in the dashboard.

## 📈 Interactive Dashboard Features (Streamlit)

Deployed as a fully functional dashboard with the following capabilities:

1. **Deal Outcome Predictor**: Predicts win/loss status for new deals based on entered parameters.
2. **Dataset Filter Tool**: Drill down into data by any column and value.
3. **Sales Summary & Insights**: Visualize win rates and lead scoring.
4. **High-Value Segments**: Identify top-converting customer types, industries, and company sizes.
5. **Cross-Segment Comparison**: Compare win rates across customer attributes (e.g., Industry vs Deal Type).


## 🛠How to Run

### Clone the repository
```
git clone https://github.com/rroshann/ML_midterm_project.git

cd ML_midterm_project/Streamlit
```

### Build the Docker image (Ensure docker is running in the background)

```
docker build -t streamlit-deal-app .
```

### Run the app
```
docker run -p 8501:8501 streamlit-deal-app
```
### Open in Browser

Go to: http://localhost:8501

# Files Included for the dashboard (Included in the Streamlit folder)

app.py: Main Streamlit app

xgb_model.pkl: Trained XGBoost model

scaler.pkl: StandardScaler used during training

binary_df.csv: Dataset for filtering in Feature 2

requirements.txt: Python dependencies

Dockerfile: Container config


## 📝 Future Improvements
- Add real-time data pipelines for live predictions

- Integrate ticket dataset for post-sale success prediction

- Expand segmentation with unsupervised learning

- Add NLP-based feature extraction from deal notes

## 👨‍💻 Contributors
- Roshan Siddartha Sivakumar

- Xiaochen Liu

- Anna Lorenz

- Najma Thomas-Akpanoko