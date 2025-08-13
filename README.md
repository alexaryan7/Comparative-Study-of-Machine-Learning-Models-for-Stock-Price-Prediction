# Comparative Study of Machine Learning Models for Stock Price Prediction

This project is a Python implementation of the research paper **"Comparative Study of Machine Learning Models for Stock Price Prediction"** by Oğulcan E. Örsel and Sasha S. Yamada (arXiv:2202.03156v1).

The goal is to replicate the paper's methodology for forecasting future stock prices by applying a linear Kalman filter and various Long Short-Term Memory (LSTM) neural network architectures to historical stock data.

---

## 📜 Abstract

This work applies and compares several machine learning techniques to historical stock prices to forecast future values. The models are tested on stocks with different volatility profiles: **Microsoft (MSFT)** for low-volatility and **Tesla (TSLA)** for high-volatility. The key findings from the paper, which this implementation validates, are:

- A simple **linear Kalman filter** can predict the next-day price of low-volatility stocks with surprisingly high accuracy.
- For high-volatility stocks, more complex models like the **CNN-LSTM** and **Bidirectional LSTM** significantly outperform the Kalman filter by better capturing complex patterns and features.
- The choice of the best predictive model is highly dependent on the volatility characteristics of the time-series data.

---

## 🤖 Models Implemented

The following five models from the paper have been implemented in the Python script:

1.  **Linear Kalman Filter**: A recursive algorithm that treats short-term price movements as a random walk. It uses the local variance of the previous three days' prices to adjust its predictions.
2.  **Single Layer LSTM**: A standard LSTM network with a single hidden layer of 64 nodes.
3.  **Dual (Stacked) Layer LSTM**: An LSTM network with two consecutive hidden layers of 64 nodes each.
4.  **Bidirectional LSTM**: An LSTM network that processes the time-series data in both forward and backward directions to capture dependencies more effectively.
5.  **CNN-LSTM**: A hybrid model that uses a 1D Convolutional Neural Network (CNN) layer for feature extraction from the time-series data before feeding it into a dense layer for prediction.

---

## 🛠️ Tech Stack & Requirements

This project is written in Python 3. The following libraries are required to run the code:

-   `yfinance`: To download historical stock market data.
-   `pandas`: For data manipulation and analysis.
-   `numpy`: For numerical operations.
-   `scikit-learn`: For data scaling (`MinMaxScaler`) and performance metrics.
-   `tensorflow`: For building and training the LSTM models.
-   `matplotlib`: For visualizing the results.

---

## 🚀 Setup and Usage

Follow these steps to set up your environment and run the experiments.

### 1. Install Dependencies

First, ensure you have Python 3 installed. Then, install the required libraries using pip:

```bash
pip install yfinance pandas numpy scikit-learn tensorflow matplotlib
```

### 2. Run the Script

Save the provided Python code as a file (e.g., `stock_predictor.py`) and run it from your terminal:

```bash
python stock_predictor.py
```

The script will automatically:
1.  Download the historical data for TSLA and MSFT.
2.  Preprocess the data.
3.  Build, train, and evaluate all five models for each stock.
4.  Print the performance metrics (RMSE, MAE, and R²) for each model in a summary table.
5.  Display plots for the best-performing models for each stock type.

---

## 🔬 Methodology

The experiment follows the methodology outlined in the paper:

-   **Dataset**: Historical daily closing prices for `TSLA` and `MSFT` are fetched from Yahoo Finance.
-   **Time Period**: January 1, 2011, to January 1, 2021.
-   **Data Split**: The data is split into a **75% training set** (approx. 7.5 years) and a **25% testing set** (approx. 2.5 years).
-   **Lookback Period**: For the LSTM models, a lookback window of **3 days** is used to predict the next day's price.
-   **Evaluation Metrics**: The models are evaluated using three standard regression metrics:
    -   **Root Mean Square Error (RMSE)**: The square root of the average of squared differences between prediction and actual observation. Lower is better.
    -   **Mean Absolute Error (MAE)**: The average of the absolute differences between prediction and actual observation. Lower is better.
    -   **R-squared (R²)**: The proportion of the variance in the dependent variable that is predictable from the independent variable(s). Closer to 1.0 is better.

---

## 📊 Expected Results

The script will output performance tables and plots. The results should closely align with the findings in the paper.

### Performance Tables

**Table 1: Performance on TSLA (High-Volatility)**
| Model | RMSE | MAE | R² |
| :--- | :--- | :--- | :--- |
| Kalman Filter | High | High | Low |
| Single Layer LSTM | Low | Low | High |
| Dual Layer LSTM | Mid | Mid | Mid |
| Bidirectional LSTM| Low | Low | High |
| **CNN-LSTM** | **Lowest**| **Lowest**| **Highest**|

**Table 2: Performance on MSFT (Low-Volatility)**
| Model | RMSE | MAE | R² |
| :--- | :--- | :--- | :--- |
| **Kalman Filter** | **Low** | **Low** | **High** |
| Single Layer LSTM | Low | Low | High |
| Dual Layer LSTM | High | High | Low |
| **Bidirectional LSTM**| **Low** | **Low** | **High** |
| CNN-LSTM | Low | Low | High |

### Result Plots

The script will generate plots visualizing the model's predictions against the actual stock prices.

**CNN-LSTM Prediction for TSLA:**
![CNN-LSTM fit for TSLA stock](https://github.com/alexaryan7/Comparative-Study-of-Machine-Learning-Models-for-Stock-Price-Prediction/blob/main/CNN-LSTM%20on%20TSLA.png)

**Bidirectional LSTM Prediction for MSFT:**
![Bidirectional LSTM fit for MSFT stock](https://i.imgur.com/Y6fH7vO.png)

---

## 📄 Citation

For more details on the methodology and analysis, please refer to the original paper:

> Örsel, O. E., & Yamada, S. S. (2022). *Comparative Study of Machine Learning Models for Stock Price Prediction*. arXiv preprint arXiv:2202.03156.
> [https://arxiv.org/abs/2202.03156](https://arxiv.org/abs/2202.03156)
