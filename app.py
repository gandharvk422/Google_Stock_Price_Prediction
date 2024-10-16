import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

st.set_page_config(page_title="Google Stock Price Prediction", page_icon="📈")

st.title("Google Stock Price Prediction App")

# File paths for datasets
train_path = "dataset/Google_Stock_Price_Train.csv"
test_path = "dataset/Google_Stock_Price_Test.csv"

@st.cache_data
def load_data():
    dataset_train = pd.read_csv(train_path)
    dataset_test = pd.read_csv(test_path)
    return dataset_train, dataset_test

def preprocess_data(dataset_train, dataset_test):
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
    model = load_model('model.h5')
    return model

# Load datasets
dataset_train, dataset_test = load_data()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Training Dataset")
    st.write(dataset_train)

with col2:
    st.subheader("Test Dataset")
    st.write(dataset_test)

# Preprocess data
X_train, y_train, X_test, sc = preprocess_data(dataset_train, dataset_test)

# Load the pre-trained model
model = build_and_load_model()

# Model Predictions
real_stock_price = dataset_test.iloc[:, 1:2].values
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualization of results
st.subheader("Stock Price Prediction (Year 2017)")
fig, ax = plt.subplots()
ax.plot(real_stock_price, color='red', label='Real Google Stock Price')
ax.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
ax.set_title("Google Stock Price Prediction")
ax.set_xlabel("Time")
ax.set_ylabel("Google Stock Price")
ax.legend()
st.pyplot(fig)

# Future prediction input
st.subheader("Predict Google Future Stock Prices (Year 2017)")
days_to_predict = st.number_input("Enter number of future days to predict", min_value=1, max_value=365, value=30)

# Predict future stock prices
if st.button("Predict Future Stock Price"):
    future_stock_price = X_test[-1].reshape(1, 60, 1)  # Start from last point
    predicted_future = []

    for _ in range(days_to_predict):
        future_pred = model.predict(future_stock_price)
        predicted_future.append(future_pred[0, 0])
        future_stock_price = np.roll(future_stock_price, -1, axis=1)
        future_stock_price[0, -1, 0] = future_pred

    predicted_future = np.array(predicted_future).reshape(-1, 1)
    predicted_future = sc.inverse_transform(predicted_future)

    # Prepare DataFrame for predicted future prices
    future_dates = pd.date_range(start=pd.to_datetime(dataset_test['Date'].values[-1]) + pd.Timedelta(days=1), periods=days_to_predict)
    future_prices_df = pd.DataFrame(data=predicted_future, index=future_dates, columns=["Predicted Price"])

    
    # Create a figure with two columns
    st.write(future_prices_df)

    # Visualization of future predictions
    st.subheader(f"Visualizing the Predicted Stock Prices for {days_to_predict} days")
    fig, ax = plt.subplots()
    ax.plot(future_prices_df.index, future_prices_df['Predicted Price'], color='green', label='Predicted Future Stock Price')
    ax.set_title('Predicted Google Stock Price for Next Days')
    ax.set_xlabel('Days')
    ax.set_ylabel('Google Stock Price')
    ax.legend()
    st.pyplot(fig)


# Display created by statement at the bottom
st.markdown("---")
st.markdown("##### &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Created by [Gandharv Kulkarni](https://share.streamlit.io/user/gandharvk422)")

st.markdown("&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;[![GitHub](https://img.shields.io/badge/GitHub-100000?style=the-badge&logo=github&logoColor=white&logoBackground=white)](https://github.com/gandharvk422) &emsp; [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/gandharvk422) &emsp; [![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=the-badge&logo=Kaggle&logoColor=white)](https://www.kaggle.com/gandharvk422)")