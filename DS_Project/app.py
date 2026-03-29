import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Smartphone AI Analyzer", page_icon="📱", layout="wide")
st.title("📱 Smartphone Price Predictor & Recommender")
st.markdown("Welcome to the AI-powered smartphone analysis dashboard. Explore data, predict prices, and find smart recommendations.")

# ==========================================
# 1. CACHED DATA LOADING & MODEL TRAINING
# ==========================================
# @st.cache_data prevents the app from reloading the CSV and retraining the AI every time you click a button
@st.cache_data
def load_and_train():
    # Load dataset (Make sure the CSV is in the same folder as this script)
    df = pd.read_csv("C:/Users/aqilr/Downloads/DS Project/processed_data2.csv")
    
    # --- Prediction Prep ---
    pred_features_num = ['storage', 'ram', 'weight', 'display_size', 'battery']
    pred_features_cat = ['phone_brand', 'os_type']
    
    X_num = df[pred_features_num].fillna(df[pred_features_num].median())
    X_cat = df[pred_features_cat].fillna('Unknown')
    X_combined = pd.concat([X_num, X_cat], axis=1)
    
    X_encoded = pd.get_dummies(X_combined, drop_first=True)
    y = df['price_usd']
    
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    
    # --- UPGRADED: Train Model with Hyperparameter Tuning ---
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    base_model = RandomForestRegressor(random_state=42)
    
    # We use n_iter=5 (tests 5 random combinations) to keep the app loading fast
    rf_tuned = RandomizedSearchCV(estimator=base_model, param_distributions=param_grid, 
                                  n_iter=5, cv=3, random_state=42, n_jobs=-1, scoring='r2')
    
    rf_tuned.fit(X_train, y_train)
    
    # Extract the winning model to use for the rest of the app
    model = rf_tuned.best_estimator_
    # --------------------------------------------------------
    
    y_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred)
    }
    
    # --- Recommendation Prep ---
    rec_features_num = ['price_usd', 'storage', 'ram', 'battery', 'display_size']
    rec_features_cat = ['phone_brand', 'os_type']
    
    X_rec_num = df[rec_features_num].fillna(0)
    X_rec_cat = df[rec_features_cat].fillna('Unknown')
    
    scaler = StandardScaler()
    X_rec_num_scaled = scaler.fit_transform(X_rec_num)
    X_rec_cat_encoded = pd.get_dummies(X_rec_cat).values
    
    X_rec_final = np.hstack((X_rec_num_scaled, X_rec_cat_encoded))
    similarity_matrix = cosine_similarity(X_rec_final)
    
    return df, model, X_encoded, y_test, y_pred, metrics, similarity_matrix, pred_features_num

# Load everything
df, model, X_encoded, y_test, y_pred, metrics, similarity_matrix, num_cols = load_and_train()

# ==========================================
# 2. DASHBOARD TABS
# ==========================================
# Create three interactive tabs for a clean UI
tab1, tab2, tab3 = st.tabs(["📊 Market Insights & EDA", "🔮 AI Price Predictor", "🤖 Smart Recommender"])

# --- TAB 1: EDA & METRICS ---
with tab1:
    st.header("Exploratory Data Analysis & Model Performance")
    
    # Top Level Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy (R²)", f"{metrics['r2']*100:.1f}%")
    col2.metric("Average Error (MAE)", f"${metrics['mae']:.2f}")
    col3.metric("Total Phones Analyzed", len(df))
    
    st.divider()
    
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Feature Correlation (Heatmap)")
        fig_heat, ax_heat = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[num_cols + ['price_usd']].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_heat)
        st.pyplot(fig_heat)

    with colB:
        st.subheader("Actual vs Predicted Prices")
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 6))
        ax_scatter.scatter(y_test, y_pred, alpha=0.5, color='royalblue')
        ax_scatter.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax_scatter.set_xlabel("Actual Price ($)")
        ax_scatter.set_ylabel("Predicted Price ($)")
        st.pyplot(fig_scatter)

    st.subheader("Top Price Drivers (Feature Importance)")
    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': importances})
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False).head(10)
    
    fig_bar, ax_bar = plt.subplots(figsize=(10, 4))
    sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis', ax=ax_bar)
    st.pyplot(fig_bar)


# --- TAB 2: PRICE PREDICTOR ---
with tab2:
    st.header("Predict Market Value")
    st.markdown("Select a smartphone to see how the AI evaluates its price based on its hardware specifications.")
    
    selected_phone_pred = st.selectbox("Select a Smartphone:", df['phone_model'].unique(), key='pred_select')
    
    if selected_phone_pred:
        phone_idx = df[df['phone_model'] == selected_phone_pred].index[0]
        target = df.iloc[phone_idx]
        
        # Format the data exactly as the model expects it
        pred_features = ['storage', 'ram', 'weight', 'display_size', 'battery', 'phone_brand', 'os_type']
        target_df = pd.DataFrame([target[pred_features]])
        target_encoded = pd.get_dummies(target_df).reindex(columns=X_encoded.columns, fill_value=0)
        
        # Make Prediction
        predicted_price = model.predict(target_encoded)[0]
        actual_price = target['price_usd']
        diff = actual_price - predicted_price
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"{target['phone_brand'].capitalize()} {target['phone_model']}")
            st.write(f"**OS:** {target['os_type']}")
            st.write(f"**RAM:** {target['ram']} GB | **Storage:** {target['storage']} GB")
            st.write(f"**Battery:** {target['battery']} mAh | **Display:** {target['display_size']}\"")
        
        with col2:
            st.metric("Actual Retail Price", f"${actual_price:.2f}")
            st.metric("AI Predicted Value", f"${predicted_price:.2f}", delta=f"${-diff:.2f} (Over/Under Value)", delta_color="inverse")
            
            if diff > 50:
                st.warning("⚠️ The AI considers this phone slightly overpriced for its hardware specs.")
            elif diff < -50:
                st.success("✅ The AI considers this phone a great deal!")
            else:
                st.info("⚖️ This phone is priced fairly according to the market.")

# --- TAB 3: SMART RECOMMENDER ---
with tab3:
    st.header("Find Similar Smartphones")
    st.markdown("Looking for alternatives? The AI will find phones with nearly identical specs, brands, and price tiers.")
    
    selected_phone_rec = st.selectbox("I am looking for a phone similar to:", df['phone_model'].unique(), key='rec_select')
    top_n = st.slider("How many recommendations?", min_value=3, max_value=10, value=5)
    
    if st.button("Get Recommendations"):
        phone_index = df[df['phone_model'] == selected_phone_rec].index[0]
        similarity_scores = list(enumerate(similarity_matrix[phone_index]))
        sorted_similar = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
        
        st.subheader("Top Alternatives:")
        
        for i, (index, score) in enumerate(sorted_similar, 1):
            rec = df.iloc[index]
            with st.expander(f"**{i}. {rec['phone_brand'].capitalize()} {rec['phone_model']}** — (Match: {score*100:.1f}%)"):
                colA, colB = st.columns(2)
                with colA:
                    st.write(f"**Price:** ${rec['price_usd']}")
                    st.write(f"**OS:** {rec['os_type']}")
                with colB:
                    st.write(f"**RAM/Storage:** {rec['ram']}GB / {rec['storage']}GB")
                    st.write(f"**Battery:** {rec['battery']} mAh")
