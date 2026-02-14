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
    layout="wide"
)

st.title("✈️ Airline Passenger Satisfaction AI")

# --- EVALUATOR HELPER ---
def get_test_data():
    file_path = "test_data.csv"
    if os.path.exists(file_path):
        df_test = pd.read_csv(file_path)
        return df_test.head(100).to_csv(index=False).encode('utf-8')
    return None

test_csv = get_test_data()
if test_csv:
    st.download_button(
        label="⬇️ Download Test Data (For Evaluators)",
        data=test_csv,
        file_name="test_data_sample.csv",
        mime="text/csv",
    )

st.divider()

# --- SIDEBAR ---
st.sidebar.header("⚙️ Control Panel")
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
        
        # --- PREVIEW SECTION ---
        with st.expander("👀 View Uploaded Data (First 5 Rows)", expanded=False):
            st.dataframe(df.head(), use_container_width=True)

        # 2. Preprocessing
        df_clean = df.drop(['Unnamed: 0', 'id'], axis=1, errors='ignore')
        if 'Arrival Delay in Minutes' in df_clean.columns:
            df_clean['Arrival Delay in Minutes'] = df_clean['Arrival Delay in Minutes'].fillna(df_clean['Arrival Delay in Minutes'].mean())

        # Load Artifacts
        try:
            label_encoders = joblib.load('model/label_encoders.pkl')
            scaler = joblib.load('model/scaler.pkl')
            model_filename = f"model/{model_name.replace(' ', '_')}.pkl"
            model = joblib.load(model_filename)
        except FileNotFoundError:
            st.error("❌ Error: Model files missing. Check 'model/' folder in GitHub.")
            st.stop()

        # Encode
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

        # Scale & Predict
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        
        # Get Probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_scaled)[:, 1]
        else:
            y_prob = y_pred

        # --- METRICS ROW ---
        if has_truth:
            st.subheader(f"📊 Performance: {model_name}")
            cols = st.columns(6)
            metrics = [
                ("Accuracy", accuracy_score(y_true, y_pred)),
                ("F1 Score", f1_score(y_true, y_pred, average='weighted')),
                ("Precision", precision_score(y_true, y_pred, average='weighted')),
                ("Recall", recall_score(y_true, y_pred, average='weighted')),
                ("AUC", roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0),
                ("MCC", matthews_corrcoef(y_true, y_pred))
            ]
            for col, (label, val) in zip(cols, metrics):
                col.metric(label, f"{val:.1%}" if label == "Accuracy" else f"{val:.3f}")

        st.divider()

        # --- VISUALS SECTION (ALWAYS VISIBLE) ---
        col_g1, col_g2 = st.columns(2)
        
        # Confusion Matrix
        with col_g1:
            if has_truth:
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots()
                # Using a dark-mode friendly colormap
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=0.5, linecolor='gray', ax=ax)
                st.pyplot(fig)
            else:
                st.info("Upload data with 'satisfaction' column to see Confusion Matrix.")
        
        # Donut Chart
        with col_g2:
            st.subheader("Prediction Distribution")
            target_encoder = label_encoders['satisfaction']
            df['Predicted Status'] = target_encoder.inverse_transform(y_pred)
            pred_counts = df['Predicted Status'].value_counts()
            
            fig_pie, ax_pie = plt.subplots()
            # Custom colors for professional look
            ax_pie.pie(pred_counts, labels=pred_counts.index, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#FF5252'])
            ax_pie.axis('equal') 
            st.pyplot(fig_pie)

        st.divider()

        # --- DETAILED DATA (BELOW VISUALS) ---
        st.subheader("📝 Detailed Analysis")
        tab1, tab2 = st.tabs(["📑 Classification Report", "🔮 Prediction Table"])

        with tab1:
            if has_truth:
                report = classification_report(y_true, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))
            else:
                st.info("Report available only when ground truth is provided.")

        with tab2:
            # Create a clean view
            if has_truth:
                df['Actual Status'] = df['satisfaction']
                view_cols = ['Predicted Status', 'Actual Status'] + [c for c in df.columns if c not in ['Predicted Status', 'Actual Status', 'satisfaction']]
            else:
                view_cols = ['Predicted Status'] + [c for c in df.columns if c != 'Predicted Status']
            
            st.dataframe(df[view_cols], use_container_width=True)
            
            # Download Results
            csv_res = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Predictions CSV", csv_res, "predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("👈 Upload your CSV file in the sidebar to start!")
