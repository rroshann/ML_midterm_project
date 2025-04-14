import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# Set up sidebar navigation
st.sidebar.title("🧭 Navigation")
choice = st.sidebar.radio("Choose a Feature:", ["Predict Deal", "Filter Dataset"])

# ---------- Feature 1: Deal Outcome Predictor ----------
if choice == "Predict Deal":
    st.title("Deal Outcome Predictor")
    st.markdown("Enter deal details to predict whether it will be **WON** or **LOST**.")

    # Numerical inputs
    feature_names = [
        '# of Agents Total',
        'Contract Term (Months)',
        '# of Agents Contracted',
        'Amount in company currency',
        'Number of Sessions',
        'Number of Form Submissions',
        'Number of Pageviews',
        'Deal_Size_Category',
        'Type',
        'Deal Type'
    ]

    numerical_inputs = []
    for feature in feature_names[:7]:
        val = st.number_input(f"{feature}", value=0.0)
        numerical_inputs.append(val)

    # Categorical dropdowns
    deal_size = st.selectbox("Deal Size Category", ['Enterprise', 'Large', 'Medium', 'Small', 'Unknown'])
    deal_type = st.selectbox("Type", ['BPO', 'Closed Lost', 'Customer', 'Former Customer', 'In Trial', 'Partner', 'Prospect', 'Suspect', 'Vendor', 'nan'])
    deal_stage = st.selectbox("Deal Type", ['Growth', 'New', 'PS', 'Renewal'])

    # Mappings from LabelEncoder
    cat_mapping = {
        'Deal_Size_Category': {'Enterprise': 0, 'Large': 1, 'Medium': 2, 'Small': 3, 'Unknown': 4},
        'Type': {'BPO': 0, 'Closed Lost': 1, 'Customer': 2, 'Former Customer': 3, 'In Trial': 4, 'Partner': 5, 'Prospect': 6, 'Suspect': 7, 'Vendor': 8, 'nan': 9},
        'Deal Type': {'Growth': 0, 'New': 1, 'PS': 2, 'Renewal': 3}
    }

    categorical_inputs = [
        cat_mapping['Deal_Size_Category'][deal_size],
        cat_mapping['Type'][deal_type],
        cat_mapping['Deal Type'][deal_stage]
    ]

    # Combine and predict
    final_input = np.array(numerical_inputs + categorical_inputs).reshape(1, -1)
    scaled_input = scaler.transform(final_input)

    if st.button("Predict Deal Outcome"):
        prediction = model.predict(scaled_input)[0]
        if prediction == 1:
            st.success("Prediction: This deal will be **WON**!")
        else:
            st.error("Prediction: This deal will be **LOST**.")

# ---------- Feature 2: Dataset Filtering ----------
elif choice == "Filter Dataset":
    st.title("Filter the Dataset by Column Value")
    st.write("")

    try:
        df = pd.read_csv("binary_df.csv")
        st.success("CSV loaded!")

        filter_column = st.selectbox("Choose a column to filter:", df.columns)
        unique_vals = df[filter_column].dropna().unique()
        selected_val = st.selectbox(f"Select a value in '{filter_column}':", unique_vals)

        filtered_df = df[df[filter_column] == selected_val]
        st.write(f"Showing rows where `{filter_column}` = **{selected_val}**:")
        st.dataframe(filtered_df)

    except Exception as e:
        st.error(f"Could not load or filter dataset: {e}") 
