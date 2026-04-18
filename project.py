import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

st.set_page_config(page_title="Influence Score Predictor", layout="wide")

st.title("📊 OTT Influence Score Prediction App")

# ==============================
# FILE UPLOAD
# ==============================
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📌 Raw Dataset")
    st.dataframe(df.head())

    # ==============================
    # CLEANING
    # ==============================
    st.subheader("🧹 Data Cleaning")

    st.write("Missing values BEFORE cleaning:")
    st.write(df.isnull().sum())

    df = df.dropna()

    st.success(f"Rows after cleaning: {df.shape[0]}")

    # ==============================
    # ENCODING
    # ==============================
    st.subheader("🔄 Encoding")

    df['Age'] = df['Age'].map({
        'Below 17': 0, '17-25': 1, '25-40': 2, 'Above 40': 3
    })

    df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

    df['WatchingFrequency'] = df['WatchingFrequency'].map({
        'Never': 0, 'Rarely': 1,
        '1–2 times a week': 2, '3–5 times a week': 3, 'Daily': 4
    })

    df['Weekly_Hours'] = df['Weekly_Hours'].map({
        'less than 3': 0, '4-7': 1, '8-15': 2, '15+ hours': 3
    })

    df['Monthly_Movies_Freq'] = df['Monthly_Movies_Freq'].map({
        'less than 2': 0, '3-5': 1, '4-7': 2, '7-10': 3, '10+ Movies': 4
    })

    df['Monthly_Series_Freq'] = df['Monthly_Series_Freq'].map({
        'less than 2': 0, '3-5': 1, '4-7': 2, '7-10': 3, '10+ series': 4
    })

    df['Binge_watch'] = df['Binge_watch'].map({
        'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Yes, very often': 3
    })

    df['Product_Bought'] = df['Product_Bought'].map({'No': 0, 'Yes': 1})

    df = pd.get_dummies(df, columns=['Fav_Genre'], drop_first=True)

    st.success("Encoding completed")

    # ==============================
    # CORRELATION
    # ==============================
    st.subheader("📈 Correlation Analysis")

    corr_with_target = df.corr()['Influence_Score'].drop('Influence_Score')
    corr_abs = corr_with_target.abs().sort_values(ascending=False)

    fig, ax = plt.subplots()
    ax.barh(corr_abs.index, corr_abs.values)
    ax.set_title("Correlation with Target")
    st.pyplot(fig)

    selected_by_corr = corr_abs[corr_abs > 0.40].index.tolist()

    st.write("Selected Features (|r| > 0.40):")
    st.write(selected_by_corr)

    # ==============================
    # VIF
    # ==============================
    st.subheader("📉 VIF Feature Selection")

    features = [
        'Career_Goals',
        'Mindset_Attitude',
        'Follow_Trends',
        'Social_Issue',
        'Lifestyle_Habits',
        'Emotional_Connection_Character',
        'Storyline_connection',
        'Fashion_Influence',
        'Trust_Information'
    ]

    dropped = []

    while True:
        X_temp = df[features]
        vif_vals = [variance_inflation_factor(X_temp.values, i)
                    for i in range(X_temp.shape[1])]

        vif_df = pd.DataFrame({
            'Feature': features,
            'VIF': vif_vals
        }).sort_values('VIF', ascending=False)

        max_vif = vif_df.iloc[0]['VIF']
        max_feat = vif_df.iloc[0]['Feature']

        if max_vif > 10:
            dropped.append(max_feat)
            features.remove(max_feat)
        else:
            break

    st.write("Dropped due to VIF:", dropped)
    st.write("Final Features:", features)

    # ==============================
    # MODEL TRAINING
    # ==============================
    st.subheader("🤖 Model Training")

    X = df[features]
    y = df['Influence_Score']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # ==============================
    # METRICS
    # ==============================
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    cv = cross_val_score(model, X, y, cv=5)

    st.subheader("📊 Model Performance")

    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{r2:.4f}")
    col2.metric("MAE", f"{mae:.4f}")
    col3.metric("RMSE", f"{rmse:.4f}")

    st.write("Cross-validation R²:", cv.mean())

    # ==============================
    # PLOTS
    # ==============================
    st.subheader("📉 Model Visualization")

    residuals = y_test - y_pred

    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, y_pred)
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.scatter(y_pred, residuals)
    ax3.axhline(y=0)
    st.pyplot(fig3)

    # ==============================
    # PREDICTION INPUT
    # ==============================
    st.subheader("🔮 Predict New Data")

    user_input = {}

    for f in features:
        user_input[f] = st.slider(f, 1, 10, 5)

    if st.button("Predict"):
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Influence Score: {prediction:.2f}")

else:
    st.info("Please upload a dataset to begin.")