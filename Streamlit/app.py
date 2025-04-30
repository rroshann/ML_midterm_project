import streamlit as st
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
import joblib

# Robust base path definition
try:
    base_dir = Path(__file__).resolve().parent
except NameError:
    base_dir = Path.cwd()

# Correct absolute paths for model and scaler
model_path = base_dir / "xgb_model.pkl"
scaler_path = base_dir / "scaler.pkl"

# Load model and scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)




# Set up sidebar
st.sidebar.title("🧭 Navigation")
choice = st.sidebar.radio("Choose a Feature:", ["Predict Deal", "Filter Dataset", "Sales Insights","High-Value Customer Segments","Cross-Segment Win Rate Comparison"])

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

    try:
        csv_path = base_dir / "binary_df.csv"
        df = pd.read_csv(csv_path)
        st.success("CSV loaded!")

        filter_column = st.selectbox("Choose a column to filter:", df.columns)
        unique_vals = df[filter_column].dropna().unique()
        selected_val = st.selectbox(f"Select a value in '{filter_column}':", unique_vals)

        filtered_df = df[df[filter_column] == selected_val]
        st.write(f"Showing rows where `{filter_column}` = **{selected_val}**:")
        st.dataframe(filtered_df)

        st.download_button("📥 Download Filtered Data", data=filtered_df.to_csv(index=False), file_name="filtered_deals.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Could not load or filter dataset: {e}")

# ---------- Feature 3: Sales Insights ----------
elif choice == "Sales Insights":
    st.title("Sales Insights Dashboard")

    try:
        csv_path = base_dir / "binary_df.csv"
        df = pd.read_csv(csv_path)

        st.subheader("Overall Deal Summary")
        total_deals = len(df)
        total_won = df['Target_Won'].sum()
        total_lost = total_deals - total_won
        win_rate = (total_won / total_deals) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Deals", total_deals)
        col2.metric("Deals Won", total_won)
        col3.metric("Deals Lost", total_lost)
        col4.metric("Win Rate", f"{win_rate:.2f}%")

        st.subheader("Top Performing Deal Types")
        top_deals = df.groupby("Deal Type")["Target_Won"].mean().sort_values(ascending=False)
        st.bar_chart(top_deals)

        st.subheader("Lead Scoring Table (sorted by win probability)")
        input_features = [
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

        # Apply encodings manually (same as training)
        cat_encoders = {
            'Deal_Size_Category': {'Enterprise': 0, 'Large': 1, 'Medium': 2, 'Small': 3, 'Unknown': 4},
            'Type': {'BPO': 0, 'Closed Lost': 1, 'Customer': 2, 'Former Customer': 3, 'In Trial': 4, 'Partner': 5, 'Prospect': 6, 'Suspect': 7, 'Vendor': 8, 'nan': 9},
            'Deal Type': {'Growth': 0, 'New': 1, 'PS': 2, 'Renewal': 3}
        }

        df_scored = df.dropna(subset=input_features).copy()
        for col, mapping in cat_encoders.items():
            df_scored[col] = df_scored[col].map(mapping)

        X_score = scaler.transform(df_scored[input_features].values)
        df_scored["Win Probability"] = model.predict_proba(X_score)[:, 1]
        st.dataframe(df_scored[["Company name", "Deal Name", "Win Probability"] + input_features].sort_values(by="Win Probability", ascending=False))

    except Exception as e:
        st.error(f"Something went wrong loading insights: {e}")
        
        
# ---------- Feature 4: High-Value Customer Segments ----------
elif choice == "High-Value Customer Segments":
    csv_path = base_dir / "binary_df.csv"
    df = pd.read_csv(csv_path)
    
    st.subheader("High-Value Customer Segments")

    segment_col = st.selectbox("Segment by:", ["Type", "Primary Industry", "Deal_Size_Category"])

    segment_summary = df.groupby(segment_col)["Target_Won"].value_counts(dropna=False).unstack(fill_value=0)
    segment_summary.columns = ['Lost (0)', 'Won (1)']
    segment_summary['Total'] = segment_summary.sum(axis=1)
    segment_summary['Win Rate (%)'] = (segment_summary['Won (1)'] / segment_summary['Total']) * 100
    segment_summary = segment_summary.sort_values(by='Win Rate (%)', ascending=False)

    st.dataframe(segment_summary.style.format({"Win Rate (%)": "{:.2f}"}))

    st.bar_chart(segment_summary["Win Rate (%)"])


# Feature 5
elif choice == "Cross-Segment Win Rate Comparison":
    csv_path = base_dir / "binary_df.csv"
    df = pd.read_csv(csv_path)
    st.subheader("Cross-Segment Win Rate Comparison")

    col1, col2 = st.columns(2)
    group_by = col1.selectbox("Group By (X-axis)", ["Type", "Primary Industry", "Deal_Size_Category", "Deal Type"])
    compare_by = col2.selectbox("Compare With (Color)", ["Deal Type", "Type", "Primary Industry", "Deal_Size_Category"])

    if group_by != compare_by:
        grouped = df.groupby([group_by, compare_by])["Target_Won"].mean().reset_index()
        pivot_df = grouped.pivot(index=group_by, columns=compare_by, values="Target_Won").fillna(0)

        st.dataframe(pivot_df.style.format("{:.2%}"))

        st.bar_chart(pivot_df)
    else:
        st.warning("Please choose two different segment types for comparison.")

