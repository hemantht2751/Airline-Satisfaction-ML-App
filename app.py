import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Airline Satisfaction AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR UI POLISH ---
st.markdown("""
    <style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        text-align: center;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #007bff;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("✈️ Airline Passenger Satisfaction AI")
st.markdown("### Intelligent Passenger Classification System")

# --- EVALUATOR HELPER: DOWNLOAD TEST DATA ---
def get_test_data():
    file_path = "test_data.csv"
    if os.path.exists(file_path):
        df_test = pd.read_csv(file_path)
        return df_test.head(100).to_csv(index=False).encode('utf-8')
    return None

test_csv = get_test_data()

# Helper button layout
col_help, _ = st.columns([1, 4])
with col_help:
    if test_csv:
        st.download_button(
            label="⬇️ Download Test CSV",
            data=test_csv,
            file_name="test_data_sample.csv",
            mime="text/csv",
            help="Download a sample file to test this app immediately."
        )

st.divider()

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configuration")
model_name = st.sidebar.selectbox(
    "Select Model",
    ["XGBoost", "Random Forest", "Logistic Regression", "Decision Tree", "KNN", "Naive Bayes"]
)

uploaded_file = st.sidebar.file_uploader("Upload Passenger Data (CSV)", type=["csv"])

# --- MAIN LOGIC ---
if uploaded_file is not None:
    try:
        # 1. Load Data
        df = pd.read_csv(uploaded_file)
        
        # 2. Preprocessing
        df_clean = df.drop(['Unnamed: 0', 'id'], axis=1, errors='ignore')
        if 'Arrival Delay in Minutes' in df_clean.columns:
            df_clean['Arrival Delay in Minutes'] = df_clean['Arrival Delay in Minutes'].fillna(df_clean['Arrival Delay in Minutes'].mean())

        # Load Artifacts
        try:
            label_encoders = joblib.load('model/label_encoders.pkl')
            scaler = joblib.load('model/scaler.pkl')
        except FileNotFoundError:
            st.error("❌ Critical Error: Model files not found in 'model/' folder.")
            st.stop()

        # Encode Categorical Cols
        for col, le in label_encoders.items():
            if col in df_clean.columns and col != 'satisfaction':
                df_clean[col] = df_clean[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        # Separate Target
        if 'satisfaction' in df_clean.columns:
            X = df_clean.drop('satisfaction', axis=1)
            y_true = label_encoders['satisfaction'].transform(df_clean['satisfaction'])
            has_truth = True
        else:
            X = df_clean
            has_truth = False

        # Scale
        X_scaled = scaler.transform(X)

        # 3. Load Model & Predict (AUTOMATIC)
        model_filename = f"model/{model_name.replace(' ', '_')}.pkl"
        model = joblib.load(model_filename)
        
        # Predictions
        y_pred = model.predict(X_scaled)
        
        # Get Probabilities for AUC (if supported)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_scaled)[:, 1]
        else:
            y_prob = y_pred

        # --- UI: METRICS SECTION ---
        if has_truth:
            st.subheader("📊 Model Performance")
            
            # Calculate all 6 metrics
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted')
            prec = precision_score(y_true, y_pred, average='weighted')
            rec = recall_score(y_true, y_pred, average='weighted')
            auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0
            mcc = matthews_corrcoef(y_true, y_pred)

            # Display Horizontally
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Accuracy", f"{acc:.1%}")
            m2.metric("F1 Score", f"{f1:.3f}")
            m3.metric("Precision", f"{prec:.3f}")
            m4.metric("Recall", f"{rec:.3f}")
            m5.metric("AUC Score", f"{auc:.3f}")
            m6.metric("MCC", f"{mcc:.3f}")
            
            st.divider()

            # --- UI: GRAPHS SECTION ---
            g1, g2 = st.columns([3, 2])
            
            with g1:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots(figsize=(8, 5)) # Bigger size
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=1, linecolor='black', ax=ax)
                plt.ylabel('Actual')
                plt.xlabel('Predicted')
                st.pyplot(fig)

            with g2:
                st.subheader("Classification Report")
                report = classification_report(y_true, y_pred, output_dict=True)
                df_report = pd.DataFrame(report).transpose()
                st.dataframe(df_report.style.format("{:.2f}"))

        # --- UI: PREDICTIONS SECTION ---
        st.divider()
        st.subheader("🔮 Prediction Results")
        
        # Convert numeric predictions back to text (Satisfied/Neutral)
        # We use the inverse_transform of the target encoder
        target_encoder = label_encoders['satisfaction']
        df['Predicted Status'] = target_encoder.inverse_transform(y_pred)
        
        # Add a visual check column
        if has_truth:
            df['Actual Status'] = df['satisfaction']
            df['Correct?'] = df['Actual Status'] == df['Predicted Status']
            cols_to_show = ['Predicted Status', 'Actual Status', 'Correct?'] + [c for c in df.columns if c not in ['Predicted Status', 'Actual Status', 'Correct?', 'satisfaction']]
        else:
            cols_to_show = ['Predicted Status'] + [c for c in df.columns if c != 'Predicted Status']

        # Show Donut Chart of Predictions
        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown("**Distribution**")
            pred_counts = df['Predicted Status'].value_counts()
            fig_pie, ax_pie = plt.subplots()
            ax_pie.pie(pred_counts, labels=pred_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
            ax_pie.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            st.pyplot(fig_pie)
            
        with c2:
            st.markdown("**Detailed Data View**")
            st.dataframe(df[cols_to_show], height=400)
            
        # Download Button for Results
        csv_results = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Predicted Results",
            data=csv_results,
            file_name="airline_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Processing Error: {e}")
        st.info("Tip: Ensure your CSV has the same columns as the training data.")

else:
    # Empty State - Fancy Placeholder
    st.info("👈 Please select a model and upload a CSV file to generate predictions automatically.")
