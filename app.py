import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Airline Satisfaction App", layout="wide")

st.title("✈️ Airline Passenger Satisfaction Classifier")
st.markdown("""
This app predicts whether a passenger is **Satisfied** or **Neutral/Dissatisfied**.
""")

# --- EVALUATOR HELPER: DOWNLOAD TEST DATA ---
# This button allows the evaluator to download the test dataset
# so they can upload it back to the app and verify functionality.
def get_test_data():
    # Make sure you upload 'test_data.csv' to your GitHub!
    file_path = "test_data.csv"
    if os.path.exists(file_path):
        # We read it and convert to CSV for the download button
        df_test = pd.read_csv(file_path)
        # We take a sample of 100 rows to keep it lightweight, or return the whole thing
        return df_test.head(100).to_csv(index=False).encode('utf-8')
    return None

test_csv = get_test_data()

col1, col2 = st.columns([1, 2])
with col1:
    if test_csv:
        st.download_button(
            label="⬇️ Download Test Data (for Evaluation)",
            data=test_csv,
            file_name="test_data_sample.csv",
            mime="text/csv",
            help="Click to download a sample of the test data to try out the app."
        )
    else:
        st.warning("⚠️ 'test_data.csv' not found. Please ensure it is uploaded to GitHub.")

st.markdown("---")

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("User Input")
model_name = st.sidebar.selectbox(
    "Choose Model",
    ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Random Forest", "XGBoost"]
)

uploaded_file = st.sidebar.file_uploader("Upload CSV (test.csv)", type=["csv"])

# --- MAIN LOGIC ---
if uploaded_file is not None:
    try:
        # 1. Load Data
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.dataframe(df.head())

        # 2. Preprocessing (MUST MATCH TRAINING EXACTLY)
        # Drop IDs
        df_clean = df.drop(['Unnamed: 0', 'id'], axis=1, errors='ignore')

        # Fill Missing Values (Arrival Delay)
        if 'Arrival Delay in Minutes' in df_clean.columns:
            df_clean['Arrival Delay in Minutes'] = df_clean['Arrival Delay in Minutes'].fillna(df_clean['Arrival Delay in Minutes'].mean())

        # Load Saved Encoders & Scaler
        try:
            label_encoders = joblib.load('model/label_encoders.pkl')
            scaler = joblib.load('model/scaler.pkl')
        except FileNotFoundError:
            st.error("❌ Error: Model files not found. Make sure the 'model' folder is uploaded.")
            st.stop()

        # Encode Categorical Cols
        for col, le in label_encoders.items():
            if col in df_clean.columns and col != 'satisfaction':
                # Safe transform: handles unknown categories gracefully
                df_clean[col] = df_clean[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        # Separate X and y (if y exists)
        if 'satisfaction' in df_clean.columns:
            X = df_clean.drop('satisfaction', axis=1)
            # Encode target using the specific target encoder
            y_true = label_encoders['satisfaction'].transform(df_clean['satisfaction'])
            has_truth = True
        else:
            X = df_clean
            has_truth = False

        # Scale Features (Using the saved scaler)
        X_scaled = scaler.transform(X)

        # 3. Load Model
        model_filename = f"model/{model_name.replace(' ', '_')}.pkl"
        
        try:
            model = joblib.load(model_filename)
        except FileNotFoundError:
            st.error(f"❌ Error: Could not find {model_filename}. Check your GitHub 'model' folder.")
            st.stop()

        # 4. Predict
        if st.button("Run Prediction"):
            y_pred = model.predict(X_scaled)

            # --- DISPLAY RESULTS ---
            st.divider()
            st.subheader(f"Results: {model_name}")

            if has_truth:
                # Metrics Columns
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2%}")
                m_col1.metric("F1 Score", f"{f1_score(y_true, y_pred, average='weighted'):.2f}")

                # Confusion Matrix
                with m_col2:
                    st.write("Confusion Matrix")
                    cm = confusion_matrix(y_true, y_pred)
                    fig, ax = plt.subplots(figsize=(3,3))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                    st.pyplot(fig)

                # Classification Report
                with m_col3:
                    st.write("Detailed Report")
                    report = classification_report(y_true, y_pred, output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose())

            else:
                # No ground truth, just show predictions
                df['Predicted_Satisfaction'] = y_pred
                st.success("Predictions Complete!")
                st.dataframe(df[['Predicted_Satisfaction']])

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👈 Please upload a CSV file to begin.")
