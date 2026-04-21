"""
Streamlit app: Media Consumption and Influence Score — Linear Regression
-----------------------------------------------------------------------
Mirrors the cells of final_improved.ipynb in an interactive UI.

Run with:
    pip install streamlit scikit-learn statsmodels seaborn scipy pandas matplotlib
    streamlit run app.py

The app expects `response.csv` in the same folder, or upload one from the sidebar.
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page configuration and styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Influence Score Regression",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title  { font-size: 42px; font-weight: 700; color: #1F3864;
                   margin-bottom: 0; }
    .sub-title   { font-size: 18px; color: #555; margin-top: 0;
                   margin-bottom: 30px; }
    .metric-box  { background-color: #F0F4F8; padding: 18px;
                   border-radius: 10px; text-align: center; }
    .stTabs [data-baseweb="tab-list"] button
            [data-testid="stMarkdownContainer"] p {
        font-size: 16px; font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path_or_buffer):
    return pd.read_csv(path_or_buffer)


@st.cache_data(show_spinner=False)
def preprocess(df_raw: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Mirrors cells: dropna → split → encode → scale."""
    df = df_raw.dropna().reset_index(drop=True)

    x = df.drop(columns="Influence_Score")
    y = pd.DataFrame(df["Influence_Score"])

    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    # --- encoding (same logic as the notebook) ---------------------------
    encoders = {}
    obj_cols = x.select_dtypes(include=["object", "string"]).columns
    for col in obj_cols:
        if col not in ["Gender", "Fav_Genre"]:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            encoders[col] = le
        else:
            ohe = OneHotEncoder(
                drop="first", handle_unknown="ignore", sparse_output=False
            )
            X_train_enc = ohe.fit_transform(X_train[[col]])
            X_test_enc = ohe.transform(X_test[[col]])
            enc_cols = ohe.get_feature_names_out([col])

            X_train_enc = pd.DataFrame(X_train_enc, columns=enc_cols, index=X_train.index)
            X_test_enc = pd.DataFrame(X_test_enc, columns=enc_cols, index=X_test.index)

            X_train = X_train.drop(columns=col)
            X_test = X_test.drop(columns=col)
            X_train = pd.concat([X_train, X_train_enc], axis=1)
            X_test = pd.concat([X_test, X_test_enc], axis=1)
            encoders[col] = ohe

    feature_names = X_train.columns.tolist()

    # --- scaling ---------------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        "df_clean": df,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "Y_train": Y_train.reset_index(drop=True),
        "Y_test": Y_test.reset_index(drop=True),
        "feature_names": feature_names,
        "scaler": scaler,
        "encoders": encoders,
    }


@st.cache_data(show_spinner=False)
def backward_elimination(X_train_scaled, Y_train_values, feature_names, threshold=0.05):
    """Drop the feature with the highest p-value > threshold, repeat until none."""
    X_df = pd.DataFrame(X_train_scaled, columns=feature_names).reset_index(drop=True)
    y = pd.Series(np.ravel(Y_train_values)).reset_index(drop=True)
    removed = []
    cols = list(feature_names)

    while True:
        X_ols = sm.add_constant(X_df[cols])
        model = sm.OLS(y, X_ols).fit()
        p_values = model.pvalues.drop("const")
        max_p = p_values.max()
        if max_p > threshold:
            worst = p_values.idxmax()
            removed.append((worst, float(max_p)))
            cols.remove(worst)
        else:
            break
    return model, cols, removed


@st.cache_data(show_spinner=False)
def fit_lasso_cv(X_train, Y_train):
    lasso = LassoCV(alphas=None, cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_train, np.ravel(Y_train))
    return lasso


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("📊 Navigation")
st.sidebar.markdown("### Data source")
uploaded = st.sidebar.file_uploader("Upload response.csv", type=["csv"])

default_path = (
    os.path.join(os.path.dirname(__file__), "response.csv")
    if "__file__" in globals() else "response.csv"
)

df_raw = None
if uploaded is not None:
    df_raw = load_data(uploaded)
    st.sidebar.success(f"Loaded uploaded file ({df_raw.shape[0]} rows)")
elif os.path.exists(default_path):
    df_raw = load_data(default_path)
    st.sidebar.info(f"Loaded response.csv from disk ({df_raw.shape[0]} rows)")
else:
    st.sidebar.warning("Upload response.csv to continue.")

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Overview",
        "🔍 Data Exploration",
        "⚙️ Preprocessing",
        "📊 Multicollinearity (VIF)",
        "✂️ Backward Elimination (OLS)",
        "🎯 Lasso Regression",
        "🔮 Predict Influence Score",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · scikit-learn · statsmodels · seaborn")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "<p class='main-title'>Media Consumption & Influence Score</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p class='sub-title'>Interactive linear regression study of how viewing habits shape influence</p>",
    unsafe_allow_html=True,
)

if df_raw is None:
    st.info("Please upload `response.csv` from the sidebar to begin.")
    st.stop()

art = preprocess(df_raw)
df_clean      = art["df_clean"]
X_train       = art["X_train"]
X_test        = art["X_test"]
Y_train       = art["Y_train"]
Y_test        = art["Y_test"]
feature_names = art["feature_names"]

# ---------------------------------------------------------------------------
# Page: Overview
# ---------------------------------------------------------------------------
if page == "🏠 Overview":
    st.header("Project Overview")

    st.markdown(
        """
        This app predicts a viewer's **Influence Score** (1–10) based on survey
        responses covering viewing habits, genre preferences, emotional
        engagement and lifestyle variables.

        The pipeline mirrors your notebook cell-by-cell:

        1. **Load & clean** — read CSV, drop missing rows.
        2. **Split** — 80% train / 20% test, `random_state=42`.
        3. **Encode** — Label encode ordinal columns, One-Hot encode nominal.
        4. **Scale** — StandardScaler (mean 0, std 1).
        5. **Check VIF** — spot multicollinearity.
        6. **Backward elimination** — drop features whose p-value > 0.05.
        7. **Lasso-CV** — automatic feature selection with cross-validated alpha.
        8. **Predict** — interactive form for a new respondent.
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        f"<div class='metric-box'><h3>{df_raw.shape[0]}</h3><p>Raw rows</p></div>",
        unsafe_allow_html=True,
    )
    c2.markdown(
        f"<div class='metric-box'><h3>{df_clean.shape[0]}</h3><p>After dropna</p></div>",
        unsafe_allow_html=True,
    )
    c3.markdown(
        f"<div class='metric-box'><h3>{df_clean.shape[1]}</h3><p>Columns</p></div>",
        unsafe_allow_html=True,
    )
    c4.markdown(
        f"<div class='metric-box'><h3>{len(feature_names)}</h3><p>Features (encoded)</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown("### First 10 rows of the raw data")
    st.dataframe(df_raw.head(10), use_container_width=True)

# ---------------------------------------------------------------------------
# Page: Data Exploration
# ---------------------------------------------------------------------------
elif page == "🔍 Data Exploration":
    st.header("Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Summary stats", "Missing values", "Distributions", "Correlation heatmap"]
    )

    with tab1:
        st.markdown("#### `df.describe()`")
        st.dataframe(df_clean.describe().round(2), use_container_width=True)

        st.markdown("#### Column types (`df.info()`)")
        info_df = pd.DataFrame({
            "Column": df_clean.columns,
            "Dtype": df_clean.dtypes.astype(str).values,
            "Non-null count": df_clean.notna().sum().values,
        })
        st.dataframe(info_df, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("#### Missing values in the raw dataset")
        missing = df_raw.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        if len(missing) == 0:
            st.success("No missing values!")
        else:
            fig, ax = plt.subplots(figsize=(8, max(3, len(missing) * 0.3)))
            missing.plot(kind="barh", ax=ax, color="#C0392B")
            ax.set_xlabel("Number of missing values")
            ax.invert_yaxis()
            st.pyplot(fig)
            st.dataframe(missing.rename("Missing").to_frame(), use_container_width=True)

    with tab3:
        st.markdown("#### Distribution of numeric features")
        num_cols = df_clean.select_dtypes(include=np.number).columns.tolist()
        selected = st.multiselect(
            "Pick columns",
            num_cols,
            default=["Influence_Score", "Follow_Trends", "Career_Goals"],
        )
        if selected:
            n_cols = 3
            n_rows = int(np.ceil(len(selected) / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
            axes = np.array(axes).flatten()
            for ax, col in zip(axes, selected):
                sns.histplot(df_clean[col], kde=True, ax=ax, color="#2E74B5")
                ax.set_title(col, fontsize=11)
            for ax in axes[len(selected):]:
                ax.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

    with tab4:
        st.markdown("#### Correlation heatmap (numeric columns)")
        num_df = df_clean.select_dtypes(include=np.number)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm",
                    fmt=".2f", annot_kws={"size": 8}, ax=ax)
        st.pyplot(fig)

# ---------------------------------------------------------------------------
# Page: Preprocessing
# ---------------------------------------------------------------------------
elif page == "⚙️ Preprocessing":
    st.header("Preprocessing Pipeline")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Training samples", X_train.shape[0])
        st.metric("Test samples", X_test.shape[0])
    with c2:
        st.metric("Features after encoding", len(feature_names))
        st.metric("Scaler", "StandardScaler (z-score)")

    st.markdown("### Encoded feature list")
    st.dataframe(
        pd.DataFrame({"Feature": feature_names}),
        use_container_width=True, hide_index=True, height=300,
    )

    st.markdown("### Scaled training set — first 10 rows")
    st.dataframe(
        pd.DataFrame(X_train, columns=feature_names).head(10).round(3),
        use_container_width=True,
    )

    with st.expander("🔎 What each step does"):
        st.markdown(
            """
            - **dropna + reset_index** → removes rows with missing values and
              resets the index so downstream alignment is safe.
            - **train_test_split (80/20, seed 42)** → done *before* encoding to
              prevent data leakage.
            - **LabelEncoder** for ordinal columns (Age, WatchingFrequency,
              Weekly_Hours, Monthly_Movies_Freq, Monthly_Series_Freq,
              Binge_watch, Product_Bought).
            - **OneHotEncoder(drop='first')** for nominal columns
              (Gender, Fav_Genre).
            - **StandardScaler** → fits on train, transforms both train and test
              using the same mean & std — so the scaler never "sees" test data.
            """
        )

# ---------------------------------------------------------------------------
# Page: VIF / Multicollinearity
# ---------------------------------------------------------------------------
elif page == "📊 Multicollinearity (VIF)":
    st.header("Multicollinearity — Variance Inflation Factor")

    st.markdown(
        "VIF measures how much each feature's variance is inflated by its "
        "correlation with the other features. Rule of thumb:"
    )
    st.markdown(
        "- VIF < 5 → **OK**   "
        "- 5 ≤ VIF < 10 → **warning**   "
        "- VIF ≥ 10 → **severe multicollinearity**"
    )

    X_df = pd.DataFrame(X_train, columns=feature_names)
    vif_df = pd.DataFrame({
        "Feature": X_df.columns,
        "VIF": [variance_inflation_factor(X_df.values, i)
                for i in range(X_df.shape[1])],
    }).sort_values("VIF", ascending=False).reset_index(drop=True)

    def vif_color(v):
        if v >= 10:
            return "background-color: #E74C3C; color: white;"
        if v >= 5:
            return "background-color: #F39C12; color: white;"
        return "background-color: #2ECC71; color: white;"

    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(
            vif_df.style.map(vif_color, subset=["VIF"]).format({"VIF": "{:.2f}"}),
            use_container_width=True, hide_index=True, height=600,
        )
    with c2:
        fig, ax = plt.subplots(figsize=(6, max(5, len(vif_df) * 0.3)))
        bar_colors = [
            "#E74C3C" if v >= 10 else "#F39C12" if v >= 5 else "#2ECC71"
            for v in vif_df["VIF"]
        ]
        ax.barh(vif_df["Feature"], vif_df["VIF"], color=bar_colors)
        ax.axvline(5, color="orange", linestyle="--", alpha=0.7, label="VIF=5")
        ax.axvline(10, color="red", linestyle="--", alpha=0.7, label="VIF=10")
        ax.invert_yaxis()
        ax.set_xlabel("VIF")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

    max_vif = vif_df["VIF"].max()
    if max_vif < 5:
        st.success(f"Maximum VIF = {max_vif:.2f} → no multicollinearity concerns ✅")
    elif max_vif < 10:
        st.warning(f"Maximum VIF = {max_vif:.2f} → mild multicollinearity")
    else:
        st.error(f"Maximum VIF = {max_vif:.2f} → severe multicollinearity")

    st.markdown("#### Feature correlation heatmap")
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(X_df.corr(), annot=True, cmap="coolwarm",
                fmt=".2f", annot_kws={"size": 7}, ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# Page: Backward Elimination (OLS)
# ---------------------------------------------------------------------------
elif page == "✂️ Backward Elimination (OLS)":
    st.header("Backward Elimination using OLS p-values")

    threshold = st.slider(
        "p-value threshold",
        min_value=0.01, max_value=0.20, value=0.05, step=0.01,
    )

    with st.spinner("Running backward elimination…"):
        final_model, kept_cols, removed = backward_elimination(
            X_train, Y_train.values, feature_names, threshold=threshold,
        )

    c1, c2 = st.columns(2)
    c1.metric("Features kept", len(kept_cols))
    c2.metric("Features removed", len(removed))

    st.markdown("### Features removed (in order)")
    if removed:
        rem_df = pd.DataFrame(removed, columns=["Feature", "p-value at removal"])
        st.dataframe(
            rem_df.style.format({"p-value at removal": "{:.4f}"}),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No features exceeded the threshold.")

    # ---- test-set evaluation of the final OLS ----
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    X_test_ols = sm.add_constant(X_test_df[kept_cols])
    y_test_pred = final_model.predict(X_test_ols)

    st.markdown("### Test-set performance of the final OLS")
    m1, m2, m3 = st.columns(3)
    m1.metric("Test R²", f"{r2_score(Y_test, y_test_pred):.4f}")
    m2.metric("Test MSE", f"{mean_squared_error(Y_test, y_test_pred):.4f}")
    m3.metric("Test RMSE", f"{np.sqrt(mean_squared_error(Y_test, y_test_pred)):.4f}")

    st.markdown("### Full OLS summary")
    st.code(final_model.summary().as_text(), language="text")

    st.markdown("### Surviving coefficients")
    coefs = final_model.params.drop("const")
    pvals = final_model.pvalues.drop("const")
    coef_df = pd.DataFrame({
        "Feature": coefs.index,
        "Coefficient": coefs.values,
        "p-value": pvals.values,
    }).sort_values("Coefficient", key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(coef_df))))
    colors = ["#2E74B5" if v >= 0 else "#C0392B" for v in coef_df["Coefficient"]]
    ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient")
    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(
        coef_df.style.format({"Coefficient": "{:.4f}", "p-value": "{:.4f}"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown("### Actual vs Predicted on the test set")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(Y_test.values.ravel(), y_test_pred, alpha=0.6, color="#2E74B5")
    lims = [min(Y_test.values.min(), y_test_pred.min()),
            max(Y_test.values.max(), y_test_pred.max())]
    ax.plot(lims, lims, "r--", lw=1, label="perfect prediction")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.legend()
    st.pyplot(fig)

# ---------------------------------------------------------------------------
# Page: Lasso Regression
# ---------------------------------------------------------------------------
elif page == "🎯 Lasso Regression":
    st.header("Lasso Regression with Cross-Validated Alpha")

    with st.spinner("Fitting LassoCV (5-fold CV)…"):
        lasso = fit_lasso_cv(X_train, Y_train.values)

    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best alpha", f"{lasso.alpha_:.5f}")
    c2.metric("Train R²", f"{r2_score(Y_train, y_train_pred):.4f}")
    c3.metric("Test R²", f"{r2_score(Y_test, y_test_pred):.4f}")
    c4.metric("Test MSE", f"{mean_squared_error(Y_test, y_test_pred):.4f}")

    c5, c6 = st.columns(2)
    c5.metric("Train MSE", f"{mean_squared_error(Y_train, y_train_pred):.4f}")
    c6.metric("Test RMSE", f"{np.sqrt(mean_squared_error(Y_test, y_test_pred)):.4f}")

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": lasso.coef_,
    })
    kept = coef_df[coef_df["Coefficient"] != 0].sort_values(
        by="Coefficient", key=abs, ascending=False
    )
    dropped = coef_df[coef_df["Coefficient"] == 0]

    st.markdown(
        f"**Features kept by Lasso:** {len(kept)}   |   "
        f"**Features dropped:** {len(dropped)}"
    )

    tab1, tab2 = st.tabs(["Kept features", "Dropped features"])

    with tab1:
        if len(kept) > 0:
            fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * len(kept))))
            colors = ["#2E74B5" if v >= 0 else "#C0392B" for v in kept["Coefficient"]]
            ax.barh(kept["Feature"], kept["Coefficient"], color=colors)
            ax.axvline(0, color="black", lw=0.8)
            ax.invert_yaxis()
            ax.set_xlabel("Coefficient")
            plt.tight_layout()
            st.pyplot(fig)

            st.dataframe(
                kept.style.format({"Coefficient": "{:.4f}"}),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("Lasso has set all coefficients to zero. Try a smaller alpha.")

    with tab2:
        if len(dropped) > 0:
            st.dataframe(
                dropped[["Feature"]].reset_index(drop=True),
                use_container_width=True, hide_index=True,
            )
        else:
            st.success("Lasso kept every feature.")

# ---------------------------------------------------------------------------
# Page: Predict
# ---------------------------------------------------------------------------
elif page == "🔮 Predict Influence Score":
    st.header("Predict a New Viewer's Influence Score")
    st.caption(
        "Fill in the form and the trained Lasso model will predict the "
        "Influence Score on a 1-10 scale."
    )

    lasso    = fit_lasso_cv(X_train, Y_train.values)
    scaler   = art["scaler"]
    encoders = art["encoders"]

    cat_options = {
        "Age":                 sorted(df_clean["Age"].unique().tolist()),
        "Gender":              sorted(df_clean["Gender"].unique().tolist()),
        "WatchingFrequency":   sorted(df_clean["WatchingFrequency"].unique().tolist()),
        "Weekly_Hours":        sorted(df_clean["Weekly_Hours"].unique().tolist()),
        "Monthly_Movies_Freq": sorted(df_clean["Monthly_Movies_Freq"].unique().tolist()),
        "Monthly_Series_Freq": sorted(df_clean["Monthly_Series_Freq"].unique().tolist()),
        "Binge_watch":         sorted(df_clean["Binge_watch"].unique().tolist()),
        "Fav_Genre":           sorted(df_clean["Fav_Genre"].unique().tolist()),
        "Product_Bought":      sorted(df_clean["Product_Bought"].unique().tolist()),
    }

    numeric_cols = [
        "Emotional_Connection_Character", "Storyline_connection",
        "Trust_Information", "Fashion_Influence", "Social_Issue",
        "Mindset_Attitude", "Lifestyle_Habits", "Career_Goals",
        "Stess_level", "Follow_Trends",
    ]

    with st.form("predict_form"):
        st.markdown("#### Demographics & viewing habits")
        c1, c2, c3 = st.columns(3)
        with c1:
            age       = st.selectbox("Age group", cat_options["Age"])
            gender    = st.selectbox("Gender", cat_options["Gender"])
            fav_genre = st.selectbox("Favourite genre", cat_options["Fav_Genre"])
        with c2:
            wf    = st.selectbox("Watching frequency", cat_options["WatchingFrequency"])
            wh    = st.selectbox("Weekly hours", cat_options["Weekly_Hours"])
            binge = st.selectbox("Binge-watching", cat_options["Binge_watch"])
        with c3:
            mm = st.selectbox("Monthly movies", cat_options["Monthly_Movies_Freq"])
            ms = st.selectbox("Monthly series", cat_options["Monthly_Series_Freq"])
            pb = st.selectbox("Bought a product from show?", cat_options["Product_Bought"])

        st.markdown("#### Influence dimensions (1 = low, 10 = high)")
        n_inputs = {}
        rows = [numeric_cols[i:i + 5] for i in range(0, len(numeric_cols), 5)]
        for row in rows:
            cols = st.columns(len(row))
            for col, name in zip(cols, row):
                with col:
                    n_inputs[name] = st.slider(
                        name.replace("_", " "),
                        min_value=1, max_value=10,
                        value=int(df_clean[name].median()),
                    )

        submitted = st.form_submit_button("Predict Influence Score 🚀", type="primary")

    if submitted:
        row = {
            "Age": age, "Gender": gender, "WatchingFrequency": wf,
            "Weekly_Hours": wh, "Monthly_Movies_Freq": mm,
            "Monthly_Series_Freq": ms, "Binge_watch": binge,
            "Fav_Genre": fav_genre, "Product_Bought": pb,
        }
        row.update(n_inputs)
        row_df = pd.DataFrame([row])

        original_cols = df_clean.drop(columns="Influence_Score").columns.tolist()
        row_df = row_df[original_cols]

        # Apply stored encoders
        for col in original_cols:
            if col in encoders:
                enc = encoders[col]
                if isinstance(enc, LabelEncoder):
                    row_df[col] = enc.transform(row_df[col].astype(str))
                else:  # OneHotEncoder
                    enc_arr = enc.transform(row_df[[col]])
                    enc_cols = enc.get_feature_names_out([col])
                    enc_df = pd.DataFrame(enc_arr, columns=enc_cols, index=row_df.index)
                    row_df = row_df.drop(columns=col)
                    row_df = pd.concat([row_df, enc_df], axis=1)

        row_df = row_df[feature_names]
        row_scaled = scaler.transform(row_df)

        pred = float(lasso.predict(row_scaled)[0])
        pred_clipped = float(np.clip(pred, 1, 10))

        st.markdown("---")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Predicted Influence Score", f"{pred_clipped:.2f} / 10")
        with c2:
            fig, ax = plt.subplots(figsize=(8, 1.4))
            ax.barh([0], [10], color="#E0E0E0")
            ax.barh([0], [pred_clipped], color="#2E74B5")
            ax.set_xlim(0, 10)
            ax.set_yticks([])
            ax.set_xticks(range(0, 11))
            ax.set_title(f"Predicted score: {pred_clipped:.2f}")
            plt.tight_layout()
            st.pyplot(fig)

        if pred_clipped >= 7:
            st.success("🎬 Highly influenced viewer — content strongly shapes their choices.")
        elif pred_clipped >= 4:
            st.info("📺 Moderately influenced viewer.")
        else:
            st.warning("🧊 Low influence — this viewer is relatively unmoved by media.")

        with st.expander("See the encoded feature vector fed to the model"):
            st.dataframe(
                pd.DataFrame(row_scaled, columns=feature_names)
                  .T.rename(columns={0: "z-score"}),
                use_container_width=True,
            )