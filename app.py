import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="APL Logistics Dashboard", page_icon="🚚", layout="wide")
st.title("APL Logistics Delivery Performance Dashboard")
st.write("Explore delivery timing, delay risk, shipping modes, regions, markets, and customer segments.")


@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, encoding="latin1", low_memory=False)
    else:
        data = pd.read_csv("./data/raw/APL_Logistics.csv", encoding="latin1", low_memory=False)
    return data


uploaded_file = st.sidebar.file_uploader("Upload APL_Logistics.csv", type=["csv"])
try:
    df = load_data(uploaded_file)
except FileNotFoundError:
    st.error("Could not find APL_Logistics.csv. Place it in the same folder as app.py or upload it in the sidebar.")
    st.stop()
except Exception as error:
    st.error(f"Could not read the CSV file: {error}")
    st.stop()

required = [
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Late_delivery_risk",
    "Shipping Mode",
    "Order Region",
    "Order Country",
    "Market",
    "Customer Segment",
    "Delivery Status"
]
missing = [col for col in required if col not in df.columns]
if missing:
    st.error("Required columns are missing: " + ", ".join(missing))
    st.stop()

# ---------- Cleaning ----------
for col in ["Days for shipping (real)", "Days for shipment (scheduled)", "Late_delivery_risk"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.drop_duplicates()
df = df.dropna(subset=[
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Late_delivery_risk"
])
df = df[
    (df["Days for shipping (real)"] >= 0) &
    (df["Days for shipment (scheduled)"] >= 0)
].copy()

# remove extra spaces in names, for example "South of  USA "
for col in ["Order Region", "Order Country", "Market", "Shipping Mode", "Customer Segment", "Delivery Status"]:
    df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)

df["Delay Gap"] = df["Days for shipping (real)"] - df["Days for shipment (scheduled)"]
df["Delivery Timing"] = np.select(
    [df["Delay Gap"] > 0, df["Delay Gap"] < 0],
    ["Delayed", "Early"],
    default="On-time"
)

# ---------- Page selector ----------
page = st.sidebar.radio("Page", ["Dashboard", "Delay prediction"])

# ---------- Prediction page ----------
if page == "Delay prediction":
    import os
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    st.header("Predict late delivery before shipping")
    st.write("Choose a model, enter the order details and get the chance that the delivery will be late. "
             "Only information known at order time is used (no Delivery Status, real days or delay gap).")

    feature_columns = ["Type", "Shipping Mode", "Days for shipment (scheduled)", "Market", "Order Region",
                       "Customer Segment", "Category Name", "Department Name", "Order Item Quantity",
                       "Order Item Discount Rate", "Order Item Product Price", "Sales"]
    missing_ml = [col for col in feature_columns if col not in df.columns]
    if missing_ml:
        st.error("Columns needed for prediction are missing: " + ", ".join(missing_ml))
        st.stop()

    # canceled shipments are never delivered, so they are not used for the models
    model_df = df[df["Delivery Status"] != "Shipping canceled"].copy()

    MODEL_FOLDER = "models"

    @st.cache_resource(show_spinner="Loading saved models...")
    def load_saved_models():
        # models saved by the notebook in ./models/
        try:
            return {
                "columns": joblib.load(f"{MODEL_FOLDER}/feature_columns.joblib"),
                "scaler": joblib.load(f"{MODEL_FOLDER}/scaler.joblib"),
                "models": {
                    "Logistic Regression": (joblib.load(f"{MODEL_FOLDER}/logistic_regression.joblib"), True),
                    "Random Forest": (joblib.load(f"{MODEL_FOLDER}/random_forest.joblib"), False),
                },
                "results": pd.read_csv(f"{MODEL_FOLDER}/model_results.csv"),
            }
        except Exception:
            return None

    @st.cache_resource(show_spinner="Training models (only the first time)...")
    def train_models(_data, rows):
        # used only when ./models/ is missing: same setup as the notebook
        X = pd.get_dummies(_data[feature_columns], drop_first=True).astype(float)
        y = _data["Late_delivery_risk"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        scaler = StandardScaler().fit(X_train)
        models = {
            "Logistic Regression": (LogisticRegression(max_iter=1000).fit(scaler.transform(X_train), y_train), True),
            "Random Forest": (RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42,
                                                     n_jobs=-1).fit(X_train, y_train), False),
        }

        def scores(name, prob):
            pred = prob > 0.5
            return [name, accuracy_score(y_test, pred), precision_score(y_test, pred),
                    recall_score(y_test, pred), f1_score(y_test, pred), roc_auc_score(y_test, prob)]

        rows_out = []
        for name, (model, scaled) in models.items():
            prob = model.predict_proba(scaler.transform(X_test) if scaled else X_test)[:, 1]
            rows_out.append(scores(name, prob))

        # baseline: late rate of each shipping mode in the training data
        mode_rate = _data.loc[X_train.index].groupby("Shipping Mode")["Late_delivery_risk"].mean()
        rows_out.append(scores("Shipping mode rule (baseline)", _data.loc[X_test.index, "Shipping Mode"].map(mode_rate)))

        return {"columns": list(X.columns), "scaler": scaler, "models": models,
                "results": pd.DataFrame(rows_out, columns=["Model", "Accuracy", "Precision", "Recall",
                                                           "F1", "ROC_AUC"]).round(4)}

    bundle = load_saved_models()
    if bundle is None:
        st.sidebar.info("No saved models found in ./models/. Run the notebook to create them. "
                        "Training the models now instead.")
        bundle = train_models(model_df, len(model_df))
    else:
        st.sidebar.success("Using the models saved in ./models/")

    model_name = st.selectbox("Model", list(bundle["models"].keys()))

    st.subheader("Order details")
    c1, c2, c3 = st.columns(3)
    shipping_mode = c1.selectbox("Shipping mode", sorted(model_df["Shipping Mode"].unique()))
    usual_days = int(model_df[model_df["Shipping Mode"] == shipping_mode]["Days for shipment (scheduled)"].mode()[0])
    scheduled_days = c1.number_input("Scheduled shipping days", min_value=0, max_value=10, value=usual_days)
    order_type = c1.selectbox("Payment type", sorted(model_df["Type"].unique()))

    market = c2.selectbox("Market", sorted(model_df["Market"].unique()))
    region_choices = sorted(model_df[model_df["Market"] == market]["Order Region"].unique())
    region = c2.selectbox("Order region", region_choices)
    segment = c2.selectbox("Customer segment", sorted(model_df["Customer Segment"].unique()))

    category = c3.selectbox("Category", sorted(model_df["Category Name"].unique()))
    department = c3.selectbox("Department", sorted(model_df["Department Name"].unique()))
    quantity = c3.number_input("Quantity", min_value=1, max_value=int(model_df["Order Item Quantity"].max()), value=1)

    c4, c5 = st.columns(2)
    discount = c4.slider("Discount rate", 0.0, float(model_df["Order Item Discount Rate"].max()), 0.0, 0.01)
    price = c5.number_input("Product price", min_value=1.0, value=float(model_df["Order Item Product Price"].median()))
    sales = price * quantity
    st.caption(f"Sales (price x quantity) = {sales:,.2f}")

    if st.button("Predict"):
        new_row = pd.DataFrame([{
            "Type": order_type, "Shipping Mode": shipping_mode, "Days for shipment (scheduled)": scheduled_days,
            "Market": market, "Order Region": region, "Customer Segment": segment, "Category Name": category,
            "Department Name": department, "Order Item Quantity": quantity,
            "Order Item Discount Rate": discount, "Order Item Product Price": price, "Sales": sales
        }])
        # same encoding as training: missing dummy columns become 0
        new_X = pd.get_dummies(new_row).reindex(columns=bundle["columns"], fill_value=0).astype(float)

        model, scaled = bundle["models"][model_name]
        prob = float(model.predict_proba(bundle["scaler"].transform(new_X) if scaled else new_X)[0, 1])

        r1, r2 = st.columns(2)
        r1.metric("Chance of late delivery", f"{prob * 100:.1f}%")
        r2.metric("Prediction", "Late" if prob > 0.5 else "On time or early")
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption("In this data every shipping mode has a fixed scheduled number of days. "
                   "Changing the scheduled days away from the usual value gives a prediction outside what the models have seen.")

    st.subheader("How good are the models? (20% test set)")
    st.dataframe(bundle["results"].sort_values("ROC_AUC", ascending=False), width="stretch")
    st.caption("A simple rule using only the shipping mode is almost as accurate as the models. "
               "Shipping mode and scheduled days carry nearly all the signal in this dataset.")
    st.stop()

# ---------- Filters ----------
st.sidebar.header("Filters")

exclude_canceled = st.sidebar.checkbox("Exclude canceled shipments", value=True)
if exclude_canceled:
    df = df[df["Delivery Status"] != "Shipping canceled"].copy()

shipping_options = sorted(df["Shipping Mode"].unique())
region_options = sorted(df["Order Region"].unique())
market_options = sorted(df["Market"].unique())
segment_options = sorted(df["Customer Segment"].unique())

selected_shipping = st.sidebar.multiselect("Shipping mode", shipping_options, default=shipping_options)
selected_regions = st.sidebar.multiselect("Order region", region_options, default=region_options)
selected_markets = st.sidebar.multiselect("Market", market_options, default=market_options)
selected_segments = st.sidebar.multiselect("Customer segment", segment_options, default=segment_options)

filtered = df[
    df["Shipping Mode"].isin(selected_shipping) &
    df["Order Region"].isin(selected_regions) &
    df["Market"].isin(selected_markets) &
    df["Customer Segment"].isin(selected_segments)
].copy()

# date range filter only works if the file has an order date column
date_cols = [c for c in filtered.columns if "order date" in c.lower()]
if date_cols:
    filtered[date_cols[0]] = pd.to_datetime(filtered[date_cols[0]], errors="coerce")
    min_date = filtered[date_cols[0]].min().date()
    max_date = filtered[date_cols[0]].max().date()
    date_range = st.sidebar.date_input("Order date range", value=(min_date, max_date),
                                       min_value=min_date, max_value=max_date)
    if len(date_range) == 2:
        filtered = filtered[
            (filtered[date_cols[0]].dt.date >= date_range[0]) &
            (filtered[date_cols[0]].dt.date <= date_range[1])
        ]

if filtered.empty:
    st.warning("No records match these filters. Select more filter values.")
    st.stop()

overall_delayed_rate = (filtered["Delivery Timing"] == "Delayed").mean() * 100

# ---------- KPIs ----------
total = len(filtered)
on_time_rate = (filtered["Delay Gap"] <= 0).mean() * 100
delayed_rate = overall_delayed_rate
avg_gap = filtered["Delay Gap"].mean()
risk_rate = filtered["Late_delivery_risk"].mean()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Orders", f"{total:,}")
c2.metric("On-time rate", f"{on_time_rate:.2f}%")
c3.metric("Delayed rate", f"{delayed_rate:.2f}%")
c4.metric("Average delay gap", f"{avg_gap:.2f} days")
c5.metric("Late risk ratio", f"{risk_rate:.3f}")

# ---------- Overview ----------
st.subheader("Delivery performance overview")
left, right = st.columns(2)
with left:
    st.write("Delivery timing")
    timing_counts = filtered["Delivery Timing"].value_counts().reindex(
        ["On-time", "Delayed", "Early"], fill_value=0
    )
    st.bar_chart(timing_counts)
with right:
    st.write("Late delivery risk")
    risk_counts = filtered["Late_delivery_risk"].value_counts().sort_index()
    risk_counts.index = ["Risk = 0" if x == 0 else "Risk = 1" for x in risk_counts.index]
    st.bar_chart(risk_counts)

st.subheader("Delay gap distribution")
fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(filtered["Delay Gap"], bins=np.arange(-6.5, 7.5, 1), edgecolor="black")
ax.set_xlabel("Actual shipping days - scheduled shipping days")
ax.set_ylabel("Number of orders")
st.pyplot(fig)
plt.close(fig)

# ---------- Shipping mode ----------
st.subheader("Shipping mode comparison")
shipping_summary = filtered.groupby("Shipping Mode").agg(
    Orders=("Delay Gap", "size"),
    Scheduled_Days=("Days for shipment (scheduled)", "mean"),
    Actual_Days=("Days for shipping (real)", "mean"),
    Average_Delay_Gap=("Delay Gap", "mean"),
    Delayed_Rate=("Delivery Timing", lambda x: (x == "Delayed").mean() * 100)
).reset_index()
shipping_summary["SLA_Compliance"] = 100 - shipping_summary["Delayed_Rate"]
shipping_summary["Efficiency_Index"] = shipping_summary["Delayed_Rate"] / overall_delayed_rate
shipping_summary = shipping_summary.round(2).sort_values("Delayed_Rate", ascending=False)
st.dataframe(shipping_summary, width="stretch")
st.caption("Efficiency index = mode delayed rate / overall delayed rate. Above 1 means worse than average.")

left, right = st.columns(2)
with left:
    st.write("Delayed rate (%) by shipping mode")
    st.bar_chart(shipping_summary.set_index("Shipping Mode")["Delayed_Rate"])
with right:
    st.write("SLA compliance (%) by shipping mode")
    st.bar_chart(shipping_summary.set_index("Shipping Mode")["SLA_Compliance"])

st.write("Scheduled vs actual days")
st.bar_chart(shipping_summary.set_index("Shipping Mode")[["Scheduled_Days", "Actual_Days"]])

# ---------- Region and market ----------
st.subheader("Regional and market analysis")
region_summary = filtered.groupby("Order Region").agg(
    Orders=("Delay Gap", "size"),
    Average_Delay_Gap=("Delay Gap", "mean"),
    Delayed_Rate=("Delivery Timing", lambda x: (x == "Delayed").mean() * 100)
).reset_index()
region_summary["Regional_Delay_Index"] = region_summary["Delayed_Rate"] / overall_delayed_rate
region_summary = region_summary.round(2).sort_values("Delayed_Rate", ascending=False)
st.write("Regions with the highest delayed rates")
st.dataframe(region_summary, width="stretch")
st.bar_chart(region_summary.set_index("Order Region")["Delayed_Rate"])

market_summary = filtered.groupby("Market").agg(
    Orders=("Delay Gap", "size"),
    Average_Delay_Gap=("Delay Gap", "mean"),
    Delayed_Rate=("Delivery Timing", lambda x: (x == "Delayed").mean() * 100)
).reset_index()
market_summary["Regional_Delay_Index"] = market_summary["Delayed_Rate"] / overall_delayed_rate
st.write("Market performance")
st.dataframe(market_summary.round(2).sort_values("Delayed_Rate", ascending=False), width="stretch")

st.write("Heatmap: delayed rate (%) by region and shipping mode")
heat = filtered.pivot_table(index="Order Region", columns="Shipping Mode",
                            values="Late_delivery_risk", aggfunc="mean") * 100
fig, ax = plt.subplots(figsize=(7, 8))
im = ax.imshow(heat, cmap="Reds", aspect="auto")
fig.colorbar(im, ax=ax, label="Delayed rate (%)")
ax.set_xticks(range(len(heat.columns)))
ax.set_xticklabels(heat.columns, rotation=30)
ax.set_yticks(range(len(heat.index)))
ax.set_yticklabels(heat.index)
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        value = heat.iloc[i, j]
        if not np.isnan(value):
            ax.text(j, i, round(value), ha="center", va="center", fontsize=8)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.write("Countries with the highest delayed rates (at least 100 orders)")
country_summary = filtered.groupby("Order Country").agg(
    Orders=("Delay Gap", "size"),
    Average_Delay_Gap=("Delay Gap", "mean"),
    Delayed_Rate=("Delivery Timing", lambda x: (x == "Delayed").mean() * 100)
).reset_index()
country_summary = country_summary[country_summary["Orders"] >= 100].round(2)
st.dataframe(country_summary.sort_values("Delayed_Rate", ascending=False).head(10), width="stretch")

# ---------- Customer segment ----------
st.subheader("Customer segment impact")
segment_summary = filtered.groupby("Customer Segment").agg(
    Orders=("Delay Gap", "size"),
    Average_Delay_Gap=("Delay Gap", "mean"),
    Delayed_Rate=("Delivery Timing", lambda x: (x == "Delayed").mean() * 100),
    Late_Orders=("Late_delivery_risk", "sum")
).reset_index()
segment_summary["Share_of_Orders_%"] = segment_summary["Orders"] / segment_summary["Orders"].sum() * 100
segment_summary["Share_of_Late_Orders_%"] = segment_summary["Late_Orders"] / max(segment_summary["Late_Orders"].sum(), 1) * 100
st.dataframe(segment_summary.round(2), width="stretch")
st.caption("If a segment's share of late orders is higher than its share of orders, it carries extra SLA risk.")

# ---------- Delivery status ----------
st.subheader("Delivery status details")
status_summary = filtered.groupby("Delivery Status").agg(
    Orders=("Delivery Status", "size"),
    Average_Delay_Gap=("Delay Gap", "mean"),
    Late_Risk_Rate=("Late_delivery_risk", "mean")
).reset_index().round(2)
st.dataframe(status_summary, width="stretch")

# ---------- Download ----------
st.subheader("Download filtered data")
csv_data = filtered.to_csv(index=False).encode("utf-8")
st.download_button("Download filtered CSV", csv_data, "APL_Logistics_filtered.csv", "text/csv")

if not date_cols:
    st.caption("This dataset has no order date column, so the date-range filter and time trend are not shown.")
