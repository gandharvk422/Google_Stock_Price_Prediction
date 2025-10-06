# Google Stock Price Prediction

Launch the web app:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stocklens.streamlit.app/)

This project uses a deep learning model, specifically a Long Short-Term Memory (LSTM) Recurrent Neural Network, to predict the stock price of Google (Alphabet Inc.). The model is trained on historical stock data and deployed as an interactive web application using Streamlit.

## 📜 Overview

The core of this project is a time-series forecasting model that analyzes the 'Open' price of Google's stock from 2012 to 2016. It learns the temporal patterns in the data to make predictions. The accompanying web application provides a user-friendly interface to:

  * Visualize the model's historical predictions against the actual stock prices for 2017.
  * Evaluate the model's accuracy through common performance metrics.
  * Forecast future stock prices for a user-defined period.

## ✨ Features

The Streamlit web application includes the following features:

  * **Historical Performance Visualization:** An interactive Plotly chart comparing the real Google stock price with the LSTM model's predictions for the year 2017.
  * **Performance Metrics:** Displays key metrics to evaluate the model's accuracy, including:
      * Mean Squared Error (MSE)
      * Root Mean Squared Error (RMSE)
      * Mean Absolute Error (MAE)
  * **Future Price Prediction:** An input field where users can enter the number of days they want to predict into the future (e.g., the next 30 days).
  * **Dynamic Forecasting Chart:** A dynamically generated chart that visualizes the predicted stock prices for the requested future period.
  * **Tabular Data:** A clean table showing the forecasted dates and the corresponding predicted prices.

## 🛠️ Tech Stack & Libraries

This project is built using the following technologies and libraries:

  * **Modeling & Data Processing:**
      * **TensorFlow & Keras:** For building and training the LSTM neural network.
      * **Scikit-learn:** Used for data preprocessing, specifically `MinMaxScaler`.
      * **NumPy & Pandas:** For efficient data manipulation and analysis.
  * **Web Application & Visualization:**
      * **Streamlit:** To create and serve the interactive web app.
      * **Plotly:** For creating rich, interactive data visualizations.
  * **Development:**
      * **Jupyter Notebook (`Notebook.ipynb`):** For initial exploration, model development, and testing.

## 🧠 Model Architecture

The prediction model is a stacked LSTM network designed to capture complex patterns in time-series data.

  * **Input Shape:** Sequences of 60 timesteps, representing the stock prices of the previous 60 days.
  * **Layers:**
    1.  **LSTM Layer 1:** 50 units, `return_sequences=True`.
    2.  `Dropout(0.2)`: To prevent overfitting.
    3.  **LSTM Layer 2:** 50 units, `return_sequences=True`.
    4.  `Dropout(0.2)`.
    5.  **LSTM Layer 3:** 50 units, `return_sequences=True`.
    6.  `Dropout(0.2)`.
    7.  **LSTM Layer 4:** 50 units.
    8.  `Dropout(0.2)`.
    9.  **Dense Output Layer:** 1 unit, which outputs the predicted stock price.
  * **Optimizer:** `Adam`
  * **Loss Function:** `Mean Squared Error`

## ⚙️ Methodology

The project follows a standard machine learning workflow for time-series forecasting:

1.  **Data Loading:** The model is trained on Google's stock data from **Jan 2012 to Dec 2016** and tested on data from **Jan 2017**.
2.  **Preprocessing:**
      * The 'Open' price is selected as the feature for prediction.
      * Data is scaled to a range of [0, 1] using `MinMaxScaler` to improve model stability and performance.
3.  **Sequence Creation:** The training data is transformed into sequences. For each prediction, the model uses the stock prices from the **previous 60 days** as input (`X_train`) to predict the price for the next day (`y_train`).
4.  **Model Training:** The LSTM model is trained on these sequences for 100 epochs, as shown in `model.py`. The trained weights are saved to `model.h5`.
5.  **Prediction & Evaluation:** The trained model is loaded in the Streamlit app to predict prices on the 2017 test set. The predictions are inverse-transformed back to their original scale and compared against the real prices.
6.  **Future Forecasting:** To predict future prices, the model takes the last 60 days of available data, predicts the next day, and then uses that prediction as part of the input for the following day's prediction in an iterative loop.

## 🚀 How to Run Locally

To run this project on your local machine, follow these steps:

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/your-username/Google_Stock_Price_Prediction.git
    cd Google_Stock_Price_Prediction
    ```

2.  **Create a Virtual Environment** (Recommended)

    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    Make sure you have all the required packages by installing them from `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit App**
    Launch the web application using the following command:

    ```bash
    streamlit run app.py
    ```

    The application should now be open and running in your web browser\!

## 📁 Project Structure

```
Google_Stock_Price_Prediction/
├── dataset/
│   ├── Google_Stock_Price_Train.csv
│   └── Google_Stock_Price_Test.csv
├── app.py                # The main Streamlit web application script
├── model.py              # Script to train the LSTM model
├── model.h5              # Pre-trained model weights
├── Notebook.ipynb        # Jupyter Notebook for development and analysis
├── requirements.txt      # Python dependencies
└── README.md             # Project README file
```

-----

*This project was created for educational and demonstration purposes and should not be used for making real financial decisions.*
