# ML_midterm_project

# Deal Outcome Predictor Dashboard

This project is a Streamlit dashboard that predicts whether a deal will be **Won** or **Lost**, and also allows for **interactive filtering** of deal data.

It uses an **XGBoost model** trained on marketing and sales features and is fully containerized using **Docker**.

---

## Features

- 🎯 **Feature 1**: Predict deal outcome from 10 key inputs
- 🔍 **Feature 2**: Filter deals dynamically by column + value
- ✅ Fast, lightweight UI powered by Streamlit
- 🐳 Docker-ready for team-wide deployment

---

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

