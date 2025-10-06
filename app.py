import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go # Replaces matplotlib
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model, Sequential
from keras.layers import Dense, LSTM, Dropout
import datetime # Added for date calculations

st.set_page_config(page_title="Google Stock Price Prediction", page_icon="📈")

st.title("Google Stock Price Prediction App")

# File paths for datasets
train_path = "dataset/Google_Stock_Price_Train.csv"
test_path = "dataset/Google_Stock_Price_Test.csv"

@st.cache_data
def load_data():
    """Loads training and testing data from CSV files."""
    dataset_train = pd.read_csv(train_path)
    dataset_test = pd.read_csv(test_path)
    return dataset_train, dataset_test

def preprocess_data(dataset_train, dataset_test):
    """Prepares data for the LSTM model."""
    training_set = dataset_train.iloc[:, 1:2].values
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    X_train, y_train = [], []
    for i in range(60, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-60:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = sc.transform(inputs)

    X_test = []
    for i in range(60, len(inputs)):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, sc

def build_and_load_model():
    """Builds the LSTM model architecture and loads pre-trained weights."""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.load_weights('model.h5')
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# --- Main App Logic ---

# Load data and model
dataset_train, dataset_test = load_data()
X_train, y_train, X_test, sc = preprocess_data(dataset_train, dataset_test)
model = build_and_load_model()

# --- Historical Prediction and Visualization ---

# Get model predictions for the test set (Year 2017)
real_stock_price = dataset_test.iloc[:, 1:2].values
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

st.subheader("Stock Price Prediction vs Real Price (Year 2017)")

# Create Plotly figure for historical data
fig_historical = go.Figure()

# Add traces for real and predicted prices
fig_historical.add_trace(go.Scatter(
    x=pd.to_datetime(dataset_test['Date']),
    y=real_stock_price.flatten(),
    mode='lines',
    name='Real Google Stock Price',
    line=dict(color='red')
))
fig_historical.add_trace(go.Scatter(
    x=pd.to_datetime(dataset_test['Date']),
    y=predicted_stock_price.flatten(),
    mode='lines',
    name='Predicted Google Stock Price',
    line=dict(color='blue')
))

# Update layout for a clean look
fig_historical.update_layout(
    title_text="Google Stock Price: Real vs. Predicted",
    xaxis_title="Date",
    yaxis_title="Stock Price (USD)",
    legend_title="Legend"
)
st.plotly_chart(fig_historical, use_container_width=True)

st.divider()

# Metrics
st.subheader("Model Performance Metrics")

mse = np.mean((predicted_stock_price - real_stock_price) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(predicted_stock_price - real_stock_price))

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(f"**Mean Squared Error (MSE)**", f"{mse:.4f}")
with col2:
    st.metric(f"**Root Mean Squared Error (RMSE)**", f"{rmse:.4f}")
with col3:
    st.metric(f"**Mean Absolute Error (MAE)**", f"{mae:.4f}")

st.divider()

# --- Future Prediction Input and Visualization ---

st.subheader("Predict Future Stock Prices")
days_to_predict = st.number_input("Enter number of future days to predict:", min_value=1, max_value=365, value=30)

if st.button("Predict Future"):
    with st.spinner(f"Predicting stock prices for the next {days_to_predict} days..."):
        # Start prediction from the last available data point
        last_60_days = X_test[-1].reshape(1, 60, 1)
        predicted_future = []

        for _ in range(days_to_predict):
            future_pred = model.predict(last_60_days)
            predicted_future.append(future_pred[0, 0])
            # Update the sequence: drop the first day and append the prediction
            last_60_days = np.roll(last_60_days, -1, axis=1)
            last_60_days[0, -1, 0] = future_pred

        # Inverse transform the scaled predictions to get actual prices
        predicted_future = sc.inverse_transform(np.array(predicted_future).reshape(-1, 1))

        # Create future dates for the x-axis
        last_date_str = dataset_test['Date'].values[-1]
        last_date = pd.to_datetime(last_date_str)
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict)
        
        # Display the predicted data in a table
        st.write("### Predicted Prices:")
        future_df = pd.DataFrame({'Date': future_dates.date, 'Predicted Price (USD)': predicted_future.flatten()})
        st.dataframe(future_df.style.format({'Predicted Price (USD)': '{:.2f}'}))

        # Create Plotly figure for future predictions
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(
            x=future_dates,
            y=predicted_future.flatten(),
            mode='lines',
            name='Predicted Future Price',
            line=dict(color='green')
        ))
        fig_future.update_layout(
            title_text=f"Predicted Google Stock Price for Next {days_to_predict} Days",
            xaxis_title="Date",
            yaxis_title="Stock Price (USD)"
        )
        st.plotly_chart(fig_future, use_container_width=True)


# --- Footer ---
st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <h5>Created by <a href="https://github.com/gandharvk422" target="_blank">Gandharv Kulkarni</a></h5>
    <a href="https://github.com/gandharvk422" target="_blank">
        <img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
    </a>
    &nbsp;
    <a href="https://linkedin.com/in/gandharvk422" target="_blank">
        <img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
    </a>
    &nbsp;
    <a href="https://www.kaggle.com/gandharvk422" target="_blank">
        <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white" alt="Kaggle">
    </a>
</div>
""", unsafe_allow_html=True)