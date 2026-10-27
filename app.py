# SI Lab Project: Regression - Product Sales Prediction (Streamlit App)
# Name: Srushti Supradeep Bansod | Section: B3 | Batch: B2 | Roll no.: 29

import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Product Sales Prediction", page_icon="🛒", layout="wide")

# ---------------- Light theme styling ----------------
st.markdown("""
<style>
.stApp { background-color: #f4f7fc; color: #1f2937; }
[data-testid="stHeader"] { background: #f4f7fc; }
[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e3e9f3; }
h1, h2, h3, h4, label, p, span, div { color: #1f2937; }

.hero {
    background: linear-gradient(120deg, #e0ecff 0%, #f3e8ff 100%);
    border: 1px solid #d6e2f7; border-radius: 16px;
    padding: 28px 32px; margin-bottom: 18px;
}
.hero h1 { margin: 0; font-size: 2.1rem; color: #1e3a8a; }
.hero p  { margin: 6px 0 0 0; color: #475569; font-size: 1.02rem; }

.card {
    background: #ffffff; border: 1px solid #e3e9f3; border-radius: 14px;
    padding: 20px 22px; box-shadow: 0 2px 8px rgba(30, 58, 138, 0.06);
    margin-bottom: 14px;
}
.card h4 { margin: 0 0 10px 0; color: #1e3a8a; }

.result {
    background: linear-gradient(120deg, #dcfce7 0%, #d1fae5 100%);
    border: 1px solid #a7f3d0; border-radius: 16px;
    padding: 24px; text-align: center; margin-top: 6px;
}
.result .label { font-size: 0.95rem; color: #047857; letter-spacing: 1px; }
.result .value { font-size: 2.8rem; font-weight: 700; color: #065f46; }

.chip {
    display: inline-block; background: #eef2ff; color: #3730a3;
    border-radius: 999px; padding: 4px 12px; font-size: 0.85rem; margin: 2px 4px 2px 0;
}
div.stButton > button {
    background: #2563eb; color: #ffffff; border: none; border-radius: 10px;
    padding: 0.65rem 1.4rem; font-weight: 600; width: 100%;
}
div.stButton > button:hover { background: #1d4ed8; color: #ffffff; }
[data-testid="stMetric"] {
    background: #ffffff; border: 1px solid #e3e9f3; border-radius: 12px; padding: 12px 16px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_files():
    model_info = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    encoders = joblib.load("encoders.pkl")
    return model_info, scaler, encoders


# ---------------- Header ----------------
st.markdown("""
<div class="hero">
  <h1>🛒 Product Sales Prediction</h1>
  <p>Enter the details of an order and the best regression model will estimate its sales amount.</p>
</div>
""", unsafe_allow_html=True)

if not (os.path.exists("best_model.pkl") and os.path.exists("scaler.pkl") and os.path.exists("encoders.pkl")):
    st.error("Model files not found. Run the Colab notebook and place best_model.pkl, scaler.pkl and encoders.pkl next to app.py.")
    st.stop()

model_info, scaler, encoders = load_files()
model = model_info["model"]
model_name = model_info["name"]
features = model_info["features"]

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### 📘 About the Project")
    st.write("SI Lab Project: **Regression - Product Sales Prediction**")
    st.markdown("**Best model**")
    st.markdown('<span class="chip">' + model_name + '</span>', unsafe_allow_html=True)
    st.markdown("**Features used**")
    st.markdown("".join('<span class="chip">' + f + '</span>' for f in features), unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Srushti Supradeep Bansod  \nSection B3 | Batch B2 | Roll no. 29")

tab_predict, tab_perf = st.tabs(["🔮 Predict Sales", "📊 Model Performance"])

# ---------------- Tab 1: Prediction ----------------
with tab_predict:
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<div class="card"><h4>💰 Order Details</h4>', unsafe_allow_html=True)
        price = st.number_input("Price per unit", min_value=1.0, max_value=1000.0, value=100.0, step=1.0)
        discount = st.select_slider("Discount", options=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                                    value=0.0, format_func=lambda d: str(int(d * 100)) + "%")
        quantity = st.number_input("Quantity", min_value=1, max_value=10, value=1, step=1)
        delivery_days = st.number_input("Delivery time (days)", min_value=1, max_value=15, value=4, step=1)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card"><h4>👤 Customer and Product</h4>', unsafe_allow_html=True)
        customer_age = st.number_input("Customer age", min_value=18, max_value=80, value=30, step=1)
        gender = st.selectbox("Customer gender", list(encoders["customer_gender"].classes_))
        category = st.selectbox("Category", list(encoders["category"].classes_))
        region = st.selectbox("Region", list(encoders["region"].classes_))
        payment_method = st.selectbox("Payment method", list(encoders["payment_method"].classes_))
        st.markdown('</div>', unsafe_allow_html=True)

    if st.button("Predict Sales"):
        new_order = pd.DataFrame([{
            "price": price,
            "discount": discount,
            "quantity": quantity,
            "delivery_time_days": delivery_days,
            "customer_age": customer_age,
            "category": category,
            "region": region,
            "payment_method": payment_method,
            "customer_gender": gender,
        }])

        # Label encode categorical columns (same encoders as the notebook)
        for col in ["category", "region", "payment_method", "customer_gender"]:
            new_order[col] = encoders[col].transform(new_order[col])
        new_order = new_order[features]

        # Scale only if the best model needs scaled data (Ridge / Lasso)
        model_input = scaler.transform(new_order) if model_info["use_scaler"] else new_order
        prediction = max(float(model.predict(model_input)[0]), 0.0)

        st.markdown(
            '<div class="result"><div class="label">PREDICTED SALES</div>'
            '<div class="value">' + format(prediction, ",.2f") + '</div></div>',
            unsafe_allow_html=True,
        )

        # Simple breakdown of the order value
        subtotal = price * quantity
        discount_amount = subtotal * discount
        expected = subtotal - discount_amount

        st.markdown("#### Order Breakdown")
        m1, m2, m3 = st.columns(3)
        m1.metric("Subtotal (price x quantity)", format(subtotal, ",.2f"))
        m2.metric("Discount amount", "- " + format(discount_amount, ",.2f"))
        m3.metric("Calculated total", format(expected, ",.2f"))

        st.caption("The model prediction is compared with the calculated total (price x (1 - discount) x quantity). "
                   "Difference: " + format(prediction - expected, "+,.2f"))

# ---------------- Tab 2: Model performance ----------------
with tab_perf:
    if os.path.exists("model_results.csv"):
        results = pd.read_csv("model_results.csv")
        best_row = results.loc[results["R2"].idxmax()]

        c1, c2, c3 = st.columns(3)
        c1.metric("Best model", best_row["Model"])
        c2.metric("R2 score", round(best_row["R2"], 4))
        c3.metric("RMSE", round(best_row["RMSE"], 2))

        st.markdown("#### Comparison of all models")
        st.dataframe(results, hide_index=True)

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**R2 score** (higher is better)")
            st.bar_chart(results.set_index("Model")["R2"])
        with g2:
            st.markdown("**RMSE** (lower is better)")
            st.bar_chart(results.set_index("Model")["RMSE"])
    else:
        st.info("model_results.csv not found. Download it from the Colab notebook to see the comparison.")
