import streamlit as st
import pandas as pd
import joblib
from thefuzz import process
import os

# Page config
st.set_page_config(
    page_title="ML Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, colorful UI
st.markdown("""
    <style>
    /* Import a modern font */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
    }
    
    /* Overall app background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
    }
    
    /* Card style containers */
    .card {
        background: #5ed674; 
        padding: 0.25rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.8rem 1.2rem;
        font-size: 1rem;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    
    /* File uploader styling override */
    .stFileUploader {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #fafafa;
    }
    
    /* Header style */
    .header {
        text-align: center;
        padding: 1rem;
    }
    .header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        color: #2e7d32;
    }
    .header p {
        font-size: 1.2rem;
        color: #555;
    }
    
    /* Footer styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background: #2e7d32;
        color: white;
        text-align: center;
        padding: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = os.path.join('models', 'model.joblib')
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        st.stop()
    return joblib.load(model_path)

def get_model_features(model):
    """
    Extract feature names from various popular libraries:
      - scikit-learn (feature_names_in_ or feature_names_)
      - XGBoost (model.get_booster().feature_names)
      - LightGBM (model.feature_name)
      - CatBoost (model.get_feature_names() or feature_names_)
    """
    if hasattr(model, 'feature_names_in_'):
        return model.feature_names_in_
    elif hasattr(model, 'feature_names_'):
        return model.feature_names_
    elif hasattr(model, 'get_booster'):
        booster = model.get_booster()
        if hasattr(booster, 'feature_names'):
            return booster.feature_names
    elif hasattr(model, 'feature_name'):
        fnames = model.feature_name()
        if any(fname.strip() for fname in fnames):
            return fnames
    elif hasattr(model, 'get_feature_names'):
        try:
            cat_features = model.get_feature_names()
            if cat_features:
                return cat_features
        except Exception:
            pass
    elif hasattr(model, 'feature_names_'):
        return model.feature_names_
    else:
        raise AttributeError("Could not automatically extract feature names from this model.")

def get_feature_matches(excel_cols, model_features, threshold=90):
    """
    For each model feature, find potential matches in the Excel columns above a certain threshold.
    """
    feature_mapping = {}
    for model_feat in model_features:
        matches = process.extract(model_feat, excel_cols, limit=3)
        high_matches = [m for m in matches if m[1] >= threshold]
        if len(high_matches) == 1:
            feature_mapping[model_feat] = high_matches[0][0]
        elif len(high_matches) > 1:
            feature_mapping[model_feat] = None
        else:
            feature_mapping[model_feat] = None
    return feature_mapping

def main():
    # Main Header
    st.markdown(
        '<div class="header"><h1>H2 Percentage Prediction App 🚀</h1><p>Modern, Interactive, and User-Friendly!</p></div>',
        unsafe_allow_html=True
    )
    
    # Sidebar Instructions
    st.sidebar.markdown('<div class="card">', unsafe_allow_html=True)
    st.sidebar.subheader("Instructions")
    st.sidebar.markdown("""
    - **Upload** your CSV or Excel file containing your data.
    - The app will try to **automatically match** your file’s columns to the model features.
    - If multiple matches are found, **select** the correct column manually.
    - Click **Generate Predictions** to view and download the results.
    """, unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Main Content Container
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload Your Data")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['xlsx', 'xls', 'csv'])
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file:
        df = None
        try:
            df = pd.read_excel(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            try:
                df = pd.read_csv(uploaded_file)
            except Exception:
                st.error("Failed to read file. Please upload a valid CSV or Excel file.")
        
        if df is not None:
            st.success("File successfully loaded!")
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.write("Preview of your data:")
            st.dataframe(df.head())
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Converting all data in the dataframe into numbers    
            df = df.apply(pd.to_numeric, errors='coerce')
            
            # Load model and extract features
            with st.spinner("Loading model..."):
                model = load_model()
                model_features = get_model_features(model)
            
            # Feature Mapping
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Feature Mapping")
            threshold_value_mapping = st.selectbox(
                "Select matching threshold:", 
                [95, 90, 85, 80, 75, 70, 65, 60, 50],
                index=1
            )
            feature_mapping = get_feature_matches(df.columns, model_features, threshold_value_mapping)
            
            # Handle multiple matches or missing matches
            for model_feat, excel_feat in feature_mapping.items():
                if excel_feat is None:
                    matches = process.extract(model_feat, df.columns, limit=3)
                    matches = [m[0] for m in matches if m[1] >= threshold_value_mapping]
                    if matches:
                        selected = st.selectbox(f"Select column for **{model_feat}**", matches, key=model_feat)
                        feature_mapping[model_feat] = selected
                    else:
                        st.warning(f"No matching column found for feature: **{model_feat}**")
                else:
                    st.info(f"**{model_feat}** automatically matched to **{excel_feat}**")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Prediction Section
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if st.button("Generate Predictions"):
                with st.spinner("Generating predictions..."):
                    # Prepare features based on mapping
                    X = pd.DataFrame()
                    for model_feat, excel_feat in feature_mapping.items():
                        X[model_feat] = df[excel_feat]
                    
                    # Generate predictions
                    predictions = model.predict(X)
                    df['Prediction'] = predictions
                    
                    st.success("Predictions generated successfully!")
                    st.subheader("Preview Results")
                    st.dataframe(df.head())
                    
                    # Download button for CSV
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_data,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("No data loaded. Please try again.")
    
    # Footer
    st.markdown("""
    <div class="footer">
        H2 Percentage Prediction App &copy; 2025 - Made with ❤️ using Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
