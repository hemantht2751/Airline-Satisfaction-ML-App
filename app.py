import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, precision_score, recall_score, roc_auc_score, matthews_corrcoef, roc_curve

# --- FIX FOR LARGE DATAFRAME ERROR ---
pd.set_option("styler.render.max_elements", 1_000_000)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Airline Satisfaction",
    page_icon="✈️",
    layout="wide"
)

# --- MAIN HEADINGS ---
st.markdown("### 🎓 ML Assignment 2")
st.title("✈️ Airline Passenger Satisfaction")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.title("ML Assignment 2")

# 1. How to Use (Collapsible)
with st.sidebar.expander("ℹ️ How to use this app", expanded=False):
    st.write("""
    1. **Download Test Data**: Click the button below if you need a sample file.
    2. **Select Model**: Choose an algorithm (e.g., XGBoost).
    3. **Upload Data**: Upload the CSV file to see predictions instantly.
    """)

st.sidebar.divider()

# 2. Download Helper
def get_test_data():
    file_path = "test_data.csv"
    if os.path.exists(file_path):
        df_test = pd.read_csv(file_path)
        return df_test.head(100).to_csv(index=False).encode('utf-8')
    return None

test_csv = get_test_data()
if test_csv:
    st.sidebar.download_button(
        label="⬇️ Download Test Data",
        data=test_csv,
        file_name="test_data_sample.csv",
        mime="text/csv",
    )

# 3. Model Controls
st.sidebar.subheader("⚙️ Configuration")
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
        
        # --- PREVIEW SECTION (OPEN BY DEFAULT) ---
        st.subheader("👀 Dataset Preview")
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
                # Handle new/unknown categories safely
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
        
        # Get Probabilities (Confidence Scores)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_scaled)[:, 1] # Prob of class 1
            confidence = np.max(model.predict_proba(X_scaled), axis=1)
        else:
            y_prob = y_pred
            confidence = np.ones(len(y_pred))

        # --- METRICS ROW ---
        if has_truth:
            st.divider()
            st.subheader(f"📊 Performance Metrics: {model_name}")
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

        # --- VISUAL ANALYSIS (ALWAYS VISIBLE) ---
        st.subheader("📈 Visual Analysis")
        col_g1, col_g2 = st.columns(2)
        
        # 1. Confusion Matrix (Left)
        with col_g1:
            if has_truth:
                st.write("**Confusion Matrix**")
                cm = confusion_matrix(y_true, y_pred)
                fig, ax = plt.subplots()
                # Clean, professional heatmap
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=0.5, linecolor='gray', ax=ax)
                plt.ylabel('Actual Label')
                plt.xlabel('Predicted Label')
                st.pyplot(fig)
            else:
                st.info("Upload data with 'satisfaction' column to see Confusion Matrix.")
        
        # 2. ROC Curve (Right)
        with col_g2:
            if has_truth and len(set(y_true)) > 1:
                st.write("**ROC Curve**")
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc_score(y_true, y_prob):.2f}')
                ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel('False Positive Rate')
                ax_roc.set_ylabel('True Positive Rate')
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)
            else:
                st.write("**Prediction Confidence Distribution**")
                fig_hist, ax_hist = plt.subplots()
                ax_hist.hist(confidence, bins=20, color='skyblue', edgecolor='black')
                ax_hist.set_xlabel('Model Confidence')
                ax_hist.set_ylabel('Count')
                st.pyplot(fig_hist)

        st.divider()

        # --- DETAILED ANALYSIS (TABS) ---
        st.subheader("📝 Detailed Analysis")
        tab1, tab2 = st.tabs(["📑 Classification Report", "🔮 Detailed Prediction Table"])

        with tab1:
            if has_truth:
                report = classification_report(y_true, y_pred, output_dict=True)
                st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap='Blues'))
            else:
                st.info("Report available only when ground truth is provided.")

        with tab2:
            # Enhance DataFrame
            target_encoder = label_encoders['satisfaction']
            df['Predicted Status'] = target_encoder.inverse_transform(y_pred)
            df['Confidence'] = [f"{c:.2%}" for c in confidence]
            
            if has_truth:
                df['Actual Status'] = df['satisfaction']
                df['Match'] = df['Actual Status'] == df['Predicted Status']
                
                # Filter Options
                filter_option = st.radio("Filter View:", ["All Predictions", "Show Mismatches Only (Errors)"], horizontal=True)
                
                if filter_option == "Show Mismatches Only (Errors)":
                    df_view = df[df['Match'] == False]
                else:
                    df_view = df
                
                view_cols = ['Predicted Status', 'Confidence', 'Actual Status', 'Match'] + [c for c in df.columns if c not in ['Predicted Status', 'Confidence', 'Actual Status', 'Match', 'satisfaction']]
            else:
                df_view = df
                view_cols = ['Predicted Status', 'Confidence'] + [c for c in df.columns if c not in ['Predicted Status', 'Confidence']]
            
            # Styling Helper
            def highlight_match(row):
                if 'Match' in row:
                    return ['background-color: #d4edda; color: black' if row['Match'] else 'background-color: #f8d7da; color: black'] * len(row)
                return [''] * len(row)

            # LIMIT VIEW TO TOP 1000 ROWS TO PREVENT CRASHING
            st.caption(f"Showing top 1,000 rows of {len(df_view)} total. Download full CSV below for complete results.")
            st.dataframe(df_view.head(1000)[view_cols].style.apply(highlight_match, axis=1), use_container_width=True)
            
            # Download Results (FULL DATASET)
            csv_res = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Detailed Results CSV", csv_res, "detailed_predictions.csv", "text/csv")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("👈 Upload your CSV file in the sidebar to start!")
