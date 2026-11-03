# =====================================================================
# Diamond Price Predictor - web frontend (Streamlit)
# Uses the same data, cleaning and models as diamond_price_prediction.ipynb
#
# Run in the VS Code terminal (inside the project folder):
#     python3 -m pip install streamlit
#     python3 -m streamlit run app.py
# =====================================================================

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             accuracy_score, f1_score, confusion_matrix)

st.set_page_config(page_title="Diamond Price Predictor", layout="wide")

ORDERS = {"cut": ["Fair", "Good", "Very Good", "Premium", "Ideal"],
          "color": ["J", "I", "H", "G", "F", "E", "D"],
          "clarity": ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]}
FEATURES = ["carat", "cut", "color", "clarity", "depth", "table"]
CLASSES = ["Budget", "Mid", "Premium"]
SEGMENT_NAMES = ["Small & affordable", "Medium", "Large & luxury"]
SEGMENT_COLOURS = ["#2a9d8f", "#e9c46a", "#e76f51"]


@st.cache_resource(show_spinner="Training all models (first run only, about 20 seconds)...")
def train_models():
    """Same cleaning and models as the notebook; runs once and is cached."""
    raw = pd.read_csv(Path(__file__).parent / "diamonds.csv")
    raw = raw.drop_duplicates()
    raw = raw[(raw.x > 0) & (raw.y > 0) & (raw.z > 0) & (raw.y < 30) & (raw.z < 30)]
    df = raw.copy()
    for col, order in ORDERS.items():
        df[col] = df[col].map({v: i for i, v in enumerate(order)})
    X, y_price = df[FEATURES], df["price"]
    y_class = pd.cut(df["price"], bins=[0, 1000, 5000, np.inf], labels=CLASSES)

    # ---- Price prediction (regression) ----
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X, y_price, test_size=0.2, random_state=42)
    scaler_r = StandardScaler().fit(Xr_tr)
    reg_models = {
        "Linear Regression": (LinearRegression(), True),
        "Decision Tree": (DecisionTreeRegressor(max_depth=15, min_samples_leaf=10, random_state=42), False),
        "Random Forest": (RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), False),
    }
    reg_rows = {}
    for name, (m, scaled) in reg_models.items():
        m.fit(scaler_r.transform(Xr_tr) if scaled else Xr_tr, yr_tr)
        p = m.predict(scaler_r.transform(Xr_te) if scaled else Xr_te)
        reg_rows[name] = {"R²": r2_score(yr_te, p), "MAE ($)": mean_absolute_error(yr_te, p),
                          "RMSE ($)": np.sqrt(mean_squared_error(yr_te, p))}
    reg_table = pd.DataFrame(reg_rows).T

    # ---- Price class (classification) ----
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X, y_class, test_size=0.2, random_state=42, stratify=y_class)
    scaler_c = StandardScaler().fit(Xc_tr)
    clf_models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000), True),
        "KNN (K = 21)": (KNeighborsClassifier(n_neighbors=21), True),
        "SVM (RBF kernel)": (SVC(kernel="rbf", C=10, gamma="scale"), True),
        "Decision Tree": (DecisionTreeClassifier(criterion="entropy", max_depth=10,
                                                 min_samples_leaf=10, random_state=42), False),
        "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), False),
    }
    clf_rows, clf_preds = {}, {}
    for name, (m, scaled) in clf_models.items():
        t0 = time.time()
        m.fit(scaler_c.transform(Xc_tr) if scaled else Xc_tr, yc_tr)
        train_time = time.time() - t0
        p = m.predict(scaler_c.transform(Xc_te) if scaled else Xc_te)
        clf_preds[name] = p
        clf_rows[name] = {"Accuracy": accuracy_score(yc_te, p), "F1-score": f1_score(yc_te, p, average="macro"),
                          "Train time (s)": train_time}
    clf_table = pd.DataFrame(clf_rows).T.sort_values("Accuracy", ascending=False)
    best_clf = clf_table.index[0]
    cm = confusion_matrix(yc_te, clf_preds[best_clf], labels=CLASSES)

    # ---- Market segments (K-Means on carat + price) ----
    seg_scaler = StandardScaler().fit(df[["carat", "price"]])
    km = KMeans(n_clusters=3, n_init=10, random_state=42).fit(seg_scaler.transform(df[["carat", "price"]]))
    order = np.argsort([df.loc[km.labels_ == c, "price"].mean() for c in range(3)])
    seg_of_cluster = {int(c): i for i, c in enumerate(order)}          # cluster id -> 0/1/2 (cheap -> luxury)
    raw["Segment"] = [SEGMENT_NAMES[seg_of_cluster[int(c)]] for c in km.labels_]
    seg_table = (raw.groupby("Segment").agg(Diamonds=("price", "size"), Avg_carat=("carat", "mean"),
                                            Avg_price=("price", "mean"))
                    .loc[SEGMENT_NAMES])

    importance = pd.Series(reg_models["Random Forest"][0].feature_importances_, index=FEATURES).sort_values()
    return dict(raw=raw, sample=raw.sample(3000, random_state=42), corr=df[["carat", "cut", "color", "clarity",
                "depth", "table", "price"]].corr(), reg_models=reg_models, scaler_r=scaler_r, reg_table=reg_table,
                clf_models=clf_models, scaler_c=scaler_c, clf_table=clf_table, best_clf=best_clf, cm=cm,
                km=km, seg_scaler=seg_scaler, seg_of_cluster=seg_of_cluster, seg_table=seg_table,
                importance=importance)


M = train_models()
raw, sample = M["raw"], M["sample"]

# ---------------- Header ----------------
st.title("Diamond Price Predictor")
st.caption("ML Lab Mini Project — price prediction (regression), Budget / Mid / Premium classification "
           "and K-Means market segments on 53,772 diamonds")

tab_predict, tab_data, tab_seg, tab_models = st.tabs(
    ["Predict a diamond", "Data explorer", "Market segments", "Model comparison"])

# ---------------- Tab 1: Predict ----------------
with tab_predict:
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.subheader("Diamond details")
        carat = st.slider("Carat (weight)", 0.20, 5.00, 1.00, 0.01)
        cut = st.select_slider("Cut (worst → best)", ORDERS["cut"], value="Ideal")
        color = st.select_slider("Colour (worst J → best D)", ORDERS["color"], value="G")
        clarity = st.select_slider("Clarity (worst → best)", ORDERS["clarity"], value="VS1")
        depth, table = 61.8, 57.0          # typical (median) values; they barely affect the price

    row = pd.DataFrame([{"carat": carat, "cut": ORDERS["cut"].index(cut),
                         "color": ORDERS["color"].index(color),
                         "clarity": ORDERS["clarity"].index(clarity),
                         "depth": depth, "table": table}])

    def run(models, scaler, name):
        m, scaled = models[name]
        return m.predict(scaler.transform(row) if scaled else row)[0]

    price = run(M["reg_models"], M["scaler_r"], "Random Forest")
    price_class = run(M["clf_models"], M["scaler_c"], M["best_clf"])
    cluster = int(M["km"].predict(M["seg_scaler"].transform(
        pd.DataFrame([[carat, price]], columns=["carat", "price"])))[0])
    segment = SEGMENT_NAMES[M["seg_of_cluster"][cluster]]

    with right:
        st.subheader("Prediction (updates as you change the details)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted price", f"${price:,.0f}")
        c2.metric("Price class", price_class)
        c3.metric("Market segment", segment)
        st.caption(f"Price from Random Forest · class from {M['best_clf']} (the most accurate classifier)")

        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.scatter(sample["carat"], sample["price"], s=5, alpha=0.25, color="steelblue", label="Diamonds in dataset")
        ax.scatter([carat], [price], s=220, color="crimson", marker="*", label="Your diamond", zorder=3)
        ax.set(xlabel="Carat", ylabel="Price ($)", title="Where your diamond sits")
        ax.legend()
        st.pyplot(fig)

# ---------------- Tab 2: Data explorer ----------------
with tab_data:
    st.subheader("Exploring the dataset")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Diamonds (after cleaning)", f"{len(raw):,}")
    k2.metric("Average price", f"${raw['price'].mean():,.0f}")
    k3.metric("Price range ($)", f"{raw['price'].min():,} – {raw['price'].max():,}")
    k4.metric("Median carat", f"{raw['carat'].median():.2f}")

    d1, d2 = st.columns(2)
    with d1:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.hist(raw["price"], bins=50, color="steelblue")
        ax.set(title="Distribution of price", xlabel="Price ($)", ylabel="Number of diamonds")
        st.pyplot(fig)
        st.caption("Right-skewed: most diamonds are cheap, a few are very expensive.")
    with d2:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.boxplot([raw.loc[raw["cut"] == c, "price"] for c in ORDERS["cut"]], showfliers=False)
        ax.set_xticklabels(ORDERS["cut"])
        ax.set(title="Price by cut", ylabel="Price ($)")
        st.pyplot(fig)
        med = raw.groupby("cut")[["price", "carat"]].median().loc[ORDERS["cut"]]
        st.caption(f"Ideal cut has the lowest median price (${med.loc['Ideal', 'price']:,.0f}) because Ideal "
                   f"diamonds are smaller (median {med.loc['Ideal', 'carat']:.2f} carat vs "
                   f"{med.loc['Fair', 'carat']:.2f} for Fair).")

    d3, d4 = st.columns(2)
    with d3:
        corr = M["corr"]
        fig, ax = plt.subplots(figsize=(6, 4.6))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr)))
        ax.set_yticklabels(corr.columns)
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)
        ax.set_title("Correlation matrix")
        st.pyplot(fig)
        st.caption(f"Carat has the strongest link with price (r = {corr.loc['carat', 'price']:.2f}).")
    with d4:
        fig, ax = plt.subplots(figsize=(6, 4.6))
        for i, c in enumerate(ORDERS["clarity"]):
            s = sample[sample["clarity"] == c]
            ax.scatter(s["carat"], s["price"], s=6, alpha=0.6, color=plt.cm.viridis(i / 7), label=c)
        ax.set(title="Carat vs price, coloured by clarity", xlabel="Carat", ylabel="Price ($)")
        ax.legend(title="Clarity", fontsize=7, markerscale=2)
        st.pyplot(fig)
        st.caption("For the same carat, better clarity means a higher price.")

# ---------------- Tab 3: Market segments ----------------
with tab_seg:
    st.subheader("K-Means market segments (K = 3, chosen by the Elbow Method and Silhouette Score)")
    s1, s2 = st.columns([1, 1.4], gap="large")
    with s1:
        seg = M["seg_table"].copy()
        seg["Avg_carat"] = seg["Avg_carat"].round(2)
        seg["Avg_price"] = seg["Avg_price"].map(lambda v: f"${v:,.0f}")
        seg.columns = ["Diamonds", "Average carat", "Average price"]
        st.dataframe(seg)
        st.success(f"Your diamond ({carat} carat, predicted ${price:,.0f}) is in the **{segment}** segment.")
    with s2:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for name, colour in zip(SEGMENT_NAMES, SEGMENT_COLOURS):
            s = sample[sample["Segment"] == name]
            ax.scatter(s["carat"], s["price"], s=6, alpha=0.5, color=colour, label=name)
        ax.scatter([carat], [price], s=260, color="black", marker="*", label="Your diamond", zorder=3)
        ax.set(xlabel="Carat", ylabel="Price ($)", title="Diamond segments")
        ax.legend()
        st.pyplot(fig)

# ---------------- Tab 4: Model comparison ----------------
with tab_models:
    st.subheader("All models tested on 20% unseen data")
    m1, m2 = st.columns(2, gap="large")
    with m1:
        st.markdown("**Price prediction (regression)**")
        rt = M["reg_table"].copy()
        st.dataframe(rt.style.format({"R²": "{:.4f}", "MAE ($)": "${:,.0f}", "RMSE ($)": "${:,.0f}"}))
        fig, ax = plt.subplots(figsize=(6, 2.8))
        rt["R²"].sort_values().plot(kind="barh", ax=ax, color="darkorange")
        ax.set(xlim=(0.85, 1.0), xlabel="R² (higher is better)")
        st.pyplot(fig)
    with m2:
        st.markdown("**Budget / Mid / Premium (classification)**")
        ct = M["clf_table"].copy()
        st.dataframe(ct.style.format({"Accuracy": "{:.2%}", "F1-score": "{:.4f}", "Train time (s)": "{:.2f}"}))
        fig, ax = plt.subplots(figsize=(6, 2.8))
        ct["Accuracy"].sort_values().plot(kind="barh", ax=ax, color="seagreen")
        ax.set(xlim=(0.85, 1.0), xlabel="Accuracy (higher is better)")
        st.pyplot(fig)

    m3, m4 = st.columns(2, gap="large")
    with m3:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        M["importance"].plot(kind="barh", ax=ax, color="darkorange")
        ax.set(title="What decides the price? (Random Forest)", xlabel="Importance")
        st.pyplot(fig)
    with m4:
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.imshow(M["cm"], cmap="Blues")
        ax.set_xticks(range(3))
        ax.set_xticklabels(CLASSES)
        ax.set_yticks(range(3))
        ax.set_yticklabels(CLASSES)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M['cm'][i, j]:,}", ha="center", va="center",
                        color="white" if M["cm"][i, j] > M["cm"].max() / 2 else "black")
        ax.set(title=f"Confusion matrix: {M['best_clf']}", xlabel="Predicted", ylabel="Actual")
        st.pyplot(fig)
