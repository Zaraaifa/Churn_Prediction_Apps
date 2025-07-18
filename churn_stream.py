import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# Load pipeline model
model = joblib.load("Final_Model_Churn_Analysis_Project.pkl")

# Ordered features
ordered_features = [
    'NumberOfDeviceRegistered',
    'DaySinceLastOrder',
    'OrderCount',
    'OrderAmountHikeFromlastYear',
    'Tenure',
    'NumberOfAddress',
    'CityTier',
    'SatisfactionScore',
    'WarehouseToHome_Bin',
    'Cashback_Cat',
    'Complain',
    'PreferredPaymentMode'
]

# Load and predict training data
def load_and_predict_training_data():
    df_train = pd.read_csv("/Users/macm1/Desktop/VSCode/X_train_cleaned.csv")
    X_input = df_train[ordered_features].copy()
    y_pred = model.predict(X_input)
    df_train["Prediction"] = ["Churn" if p == 1 else "Not Churn" for p in y_pred]
    return df_train

df_train_with_preds = load_and_predict_training_data()

# Title
st.title("📉 Customer Churn Prediction App")

# --- Optional Dashboard Summary ---
st.markdown("---")
st.header("📊 Customer Churn Summary Dashboard")

# Load history data if available
if os.path.exists("history.csv"):
    df_history = pd.read_csv("history.csv")

    # Hitung metrik
    total_customers = len(df_history)
    churn_rate = df_history["Prediction"].value_counts(normalize=True).to_dict()
    avg_satisfaction = df_history["SatisfactionScore"].mean() if "SatisfactionScore" in df_history.columns else None
    avg_order_count = df_history["OrderCount"].mean() if "OrderCount" in df_history.columns else None

    # Layout 3 cards
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predicted Customers", f"{total_customers}")
    col2.metric("Avg. Satisfaction Score", f"{avg_satisfaction:.2f}" if avg_satisfaction else "-")
    col3.metric("Avg. Order Count", f"{avg_order_count:.2f}" if avg_order_count else "-")

    # Pie chart: Churn vs Not Churn
    st.subheader("📉 Churn Distribution")
    churn_counts = df_history["Prediction"].value_counts()
    fig1, ax1 = plt.subplots()
    ax1.pie(
        churn_counts,
        labels=churn_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#f44336", "#4caf50"]
    )
    ax1.axis("equal")
    st.pyplot(fig1)

    # Line chart: Tenure distribution (optional)
    if "Tenure" in df_history.columns:
        st.subheader("📈 Customer Tenure Distribution")
        tenure_counts = df_history["Tenure"].value_counts().sort_index()
        fig2, ax2 = plt.subplots()
        ax2.plot(tenure_counts.index, tenure_counts.values, marker="o")
        ax2.set_xlabel("Tenure (Months)")
        ax2.set_ylabel("Number of Customers")
        ax2.set_title("Customer Distribution by Tenure")
        st.pyplot(fig2)

else:
    st.info("ℹ️ No customer history available yet to generate dashboard.")


st.markdown("Fill in customer data to predict the likelihood of churn. "
            "The prediction result and history will be displayed below.")

# Input form
with st.form("predict_form"):
    st.subheader("🧾 Enter Customer Data")

    col1, col2 = st.columns(2)

    with col1:
        NumberOfDeviceRegistered = st.number_input(
            "Number of Devices Registered",
            min_value=0,
            help="How many devices are registered to the customer's account."
        )
        DaySinceLastOrder = st.number_input(
            "Day Since Last Order",
            min_value=0,
            help="Number of days since the last order was placed."
        )
        OrderCount = st.number_input(
            "Total Order Count in Months",
            min_value=0,
            help="Total number of orders placed in the last months."
        )
        OrderAmountHikeFromlastYear = st.number_input(
            "Order Amount Hike from Last Year (%)",
            min_value=0.0,
            help="Percentage increase in total order value from the previous year."
        )
        Tenure = st.number_input(
            "Customer Tenure (months)",
            min_value=0,
            help="How long (in months) the customer has been with the company."
        )
        NumberOfAddress = st.number_input(
            "Number of Saved Addresses",
            min_value=0,
            help="Total shipping addresses saved in the customer's account."
        )

    with col2:
        CityTier = st.selectbox(
            "City Tier",
            [1, 2, 3],
            help="Tier of the city the customer lives in. Tier 1 = major city, Tier 3 = smaller town."
        )
        SatisfactionScore = st.selectbox(
            "Satisfaction Score",
            [1, 2, 3, 4, 5],
            help="Customer satisfaction score (1 = very dissatisfied, 5 = very satisfied)."
        )
        WarehouseToHome_Bin = st.selectbox(
            "Distance from Warehouse to Home",
            ['Very Close (<10)', 'Close (10–14)', 'Far (15–20)', 'Very Far (>20)'],
            help="Distance from warehouse to the customer's home."
        )
        Cashback_Cat = st.selectbox(
            "Cashback Level",
            ['Low', 'Medium', 'High'],
            help="Cashback category typically received by the customer."
        )
        Complain = st.selectbox(
            "Has Complained Before?",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Whether the customer has ever submitted a complaint."
        )
        PreferredPaymentMode = st.selectbox(
            "Preferred Payment Method",
            ['Credit Card', 'Debit Card', 'UPI', 'Cash on Delivery', 'E wallet'],
            help="Most frequently used payment method by the customer."
        )

    submitted = st.form_submit_button("🚀 Predict Churn")

if submitted:
    input_data = pd.DataFrame([{
        "NumberOfDeviceRegistered": NumberOfDeviceRegistered,
        "DaySinceLastOrder": DaySinceLastOrder,
        "OrderCount": OrderCount,
        "OrderAmountHikeFromlastYear": OrderAmountHikeFromlastYear,
        "Tenure": Tenure,
        "NumberOfAddress": NumberOfAddress,
        "CityTier": CityTier,
        "SatisfactionScore": SatisfactionScore,
        "WarehouseToHome_Bin": WarehouseToHome_Bin,
        "Cashback_Cat": Cashback_Cat,
        "Complain": Complain,
        "PreferredPaymentMode": PreferredPaymentMode
    }])[ordered_features]

    # Predict
    prediction = model.predict(input_data)[0]
    pred_label = "Churn" if prediction == 1 else "Not Churn"

    # Display result with color highlight
    if prediction == 1:
        st.error("🔴 Prediction Result: **CHURN**")
    else:
        st.success("🟢 Prediction Result: **NOT CHURN**")

    # Save to history
    input_data["Prediction"] = pred_label
    if os.path.exists("history.csv"):
        history_df = pd.read_csv("history.csv")
        history_df = pd.concat([history_df, input_data], ignore_index=True)
    else:
        history_df = input_data
    history_df.to_csv("history.csv", index=False)

# Display predictions from training data
st.subheader("📊 Predictions from Training Data")
st.dataframe(df_train_with_preds)

# Display prediction history
st.subheader("📝 User Prediction History")
if os.path.exists("history.csv"):
    st.dataframe(pd.read_csv("history.csv"))
else:
    st.info("No prediction history yet.")
