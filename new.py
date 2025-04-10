


import certifi
print(certifi.where())


import os
import certifi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from alpha_vantage.timeseries import TimeSeries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Fix SSL certificate issue
os.environ['SSL_CERT_FILE'] = certifi.where()

# Alpha Vantage API Key
API_KEY = "SQA4D550151BL3GJ"  # Replace with your API Key

# Function to Fetch Large Stock Data
def get_stock_data(symbol, interval="daily"):                                   # symbol: the stock ticker symbol, e.g., "AAPL" for Apple, "GOOG" for Google.
                                                                                # interval: how frequently you want the stock data (default is "daily").                      
    ts = TimeSeries(key=API_KEY, output_format="pandas")                        # TimeSeries is a class provided by the alpha_vantage library.
                                                                                # output_format="pandas" means the data will be automatically returned in a pandas DataFrame (instead of JSON).
    if interval == "daily":
        data, meta_data = ts.get_daily(symbol=symbol, outputsize="full")
    else:
        raise ValueError("Invalid interval")
    
    data.columns = ["Open", "High", "Low", "Close", "Volume"]
    data = data.iloc[::-1]  # Reverse order to ascending
    data.index = pd.to_datetime(data.index)
    return data.head(5000)  # Limit to 5000 rows

# Fetch Data
symbol = "AAPL"
df = get_stock_data(symbol)

# 1️⃣ Stock Price Analysis
df["SMA_50"] = df["Close"].rolling(window=50).mean()
df["SMA_200"] = df["Close"].rolling(window=200).mean()
plt.figure(figsize=(12, 5))
plt.plot(df["Close"], label="Close Price")
plt.plot(df["SMA_50"], label="50-Day SMA", linestyle="dashed")
plt.plot(df["SMA_200"], label="200-Day SMA", linestyle="dashed")
plt.title(f"{symbol} Stock Price & Moving Averages")
plt.legend()
plt.show()

# 2️⃣ Trading Volume Insights
plt.figure(figsize=(12, 5))
plt.plot(df["Volume"], label="Trading Volume", color="purple")
plt.title(f"{symbol} Trading Volume Over Time")
plt.legend()
plt.show()

# 3️⃣ Stock Price Prediction (Linear Regression)
df["Target"] = df["Close"].shift(-5)  # Predict 5 days ahead
df.dropna(inplace=True)
X = df[["Open", "High", "Low", "Close"]]
y = df["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
plt.figure(figsize=(12, 5))
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted", linestyle="dashed")
plt.title("Stock Price Prediction")
plt.legend()
plt.show()

# 4️⃣ Market Correlation Analysis
symbols = ["AAPL", "MSFT", "GOOGL"]
stock_data = {}
for sym in symbols:
    stock_data[sym] = get_stock_data(sym)["Close"]
df_corr = pd.DataFrame(stock_data)
sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm")
plt.title("Stock Correlation Matrix")
plt.show()

# 5️⃣ Stock Performance Comparison
returns = df_corr.pct_change().mean() * 252  # Annualized Returns
volatility = df_corr.pct_change().std() * np.sqrt(252)  # Annualized Volatility
performance_df = pd.DataFrame({"Returns": returns, "Volatility": volatility})
print(performance_df)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=volatility, y=returns, hue=returns.index, s=100)
plt.xlabel("Volatility (Risk)")
plt.ylabel("Annualized Returns")
plt.title("Stock Performance (Risk vs. Return)")
plt.legend()
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Create a Seaborn Lineplot for Closing Prices
plt.figure(figsize=(12, 6))
sns.lineplot(data=df, x=df.index, y="Close", color="blue")
plt.title(f"{symbol} Closing Price Over Time", fontsize=16)
plt.xlabel("Date", fontsize=14)
plt.ylabel("Close Price", fontsize=14)
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# Calculate daily returns
df['Daily Return'] = df['Close'].pct_change()

# Seaborn KDE plot
plt.figure(figsize=(12,6))
sns.kdeplot(df['Daily Return'].dropna(), fill=True, color="green", alpha=0.6)
plt.title(f"{symbol} Daily Return Distribution", fontsize=16)
plt.xlabel("Daily Return", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.grid(True)
plt.show()



















# import os
# import certifi
# os.environ["SSL_CERT_FILE"] = certifi.where()



# import os
# import certifi
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.graph_objects as go
# import plotly.express as px
# from alpha_vantage.timeseries import TimeSeries
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# import requests

# url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&outputsize=full&apikey=VYIATSQKQZ41EHOX&datatype=json"
# response = requests.get(url, verify=False)




# # Fix SSL certificate issue
# os.environ['SSL_CERT_FILE'] = certifi.where()

# # Alpha Vantage API Key
# API_KEY = "VYIATSQKQZ41EHOX"  # Replace with your API Key

# # Function to Fetch Large Stock Data
# def get_stock_data(symbol, interval="daily"):
#     ts = TimeSeries(key=API_KEY, output_format="pandas")
#     if interval == "daily":
#         data, meta_data = ts.get_daily(symbol=symbol, outputsize="full")
#     else:
#         raise ValueError("Invalid interval")
    
#     data.columns = ["Open", "High", "Low", "Close", "Volume"]
#     data = data.iloc[::-1]  # Reverse order to ascending
#     data.index = pd.to_datetime(data.index)
#     return data.head(5000)  # Limit to 5000 rows

# # Fetch Data
# symbol = "AAPL"
# df = get_stock_data(symbol)

# # 1️⃣ Stock Price Analysis
# df["SMA_50"] = df["Close"].rolling(window=50).mean()
# df["SMA_200"] = df["Close"].rolling(window=200).mean()
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name='Close Price'))
# fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], mode='lines', name='50-Day SMA', line=dict(dash='dash')))
# fig.add_trace(go.Scatter(x=df.index, y=df["SMA_200"], mode='lines', name='200-Day SMA', line=dict(dash='dot')))
# fig.update_layout(title=f"{symbol} Stock Price & Moving Averages", xaxis_title='Date', yaxis_title='Price', template='plotly_dark')
# fig.show()

# # 2️⃣ Trading Volume Insights
# fig = px.line(df, x=df.index, y="Volume", title=f"{symbol} Trading Volume Over Time", labels={'Volume': 'Trading Volume'})
# fig.update_layout(template='plotly_dark')
# fig.show()

# # 3️⃣ Stock Price Prediction (Linear Regression)
# df["Target"] = df["Close"].shift(-5)  # Predict 5 days ahead
# df.dropna(inplace=True)
# X = df[["Open", "High", "Low", "Close"]]
# y = df["Target"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
# print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
# fig = go.Figure()
# fig.add_trace(go.Scatter(y=y_test.values, mode='lines', name='Actual'))
# fig.add_trace(go.Scatter(y=y_pred, mode='lines', name='Predicted', line=dict(dash='dash')))
# fig.update_layout(title="Stock Price Prediction", xaxis_title='Days', yaxis_title='Price', template='plotly_dark')
# fig.show()

# # 4️⃣ Market Correlation Analysis
# symbols = ["AAPL", "MSFT", "GOOGL"]
# stock_data = {}
# for sym in symbols:
#     stock_data[sym] = get_stock_data(sym)["Close"]
# df_corr = pd.DataFrame(stock_data)
# fig = px.imshow(df_corr.corr(), text_auto=True, title="Stock Correlation Matrix", color_continuous_scale='RdBu')
# fig.show()

# # 5️⃣ Stock Performance Comparison
# returns = df_corr.pct_change().mean() * 252  # Annualized Returns
# volatility = df_corr.pct_change().std() * np.sqrt(252)  # Annualized Volatility
# performance_df = pd.DataFrame({"Returns": returns, "Volatility": volatility})
# fig = px.scatter(performance_df, x="Volatility", y="Returns", text=performance_df.index, size=[10]*len(performance_df), title="Stock Performance (Risk vs. Return)", labels={'Volatility': 'Risk', 'Returns': 'Annualized Return'})
# fig.update_traces(textposition='top center')
# fig.show()









# import os
# import certifi
# import numpy as np
# import pandas as pd
# import plotly.graph_objects as go
# import plotly.express as px
# from alpha_vantage.timeseries import TimeSeries
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error

# # Fix SSL certificate issue
# os.environ["SSL_CERT_FILE"] = certifi.where()

# # Alpha Vantage API Key
# API_KEY = "VYIATSQKQZ41EHOX"  # Replace with your actual API Key

# # Function to Fetch Stock Data
# def get_stock_data(symbol):
#     ts = TimeSeries(key=API_KEY, output_format="pandas")
#     data, _ = ts.get_daily(symbol=symbol, outputsize="full")
#     data.columns = ["Open", "High", "Low", "Close", "Volume"]
#     data = data.iloc[::-1]  # Reverse to ascending order
#     data.index = pd.to_datetime(data.index)
#     return data.head(5000)  # Limit to 5000 rows

# # Fetch Data
# symbol = "AAPL"
# df = get_stock_data(symbol)

# # Stock Price Analysis with Moving Averages
# df["SMA_50"] = df["Close"].rolling(window=50).mean()
# df["SMA_200"] = df["Close"].rolling(window=200).mean()

# fig = go.Figure()
# fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode='lines', name='Close Price'))
# fig.add_trace(go.Scatter(x=df.index, y=df["SMA_50"], mode='lines', name='50-Day SMA', line=dict(dash='dash')))
# fig.add_trace(go.Scatter(x=df.index, y=df["SMA_200"], mode='lines', name='200-Day SMA', line=dict(dash='dot')))
# fig.update_layout(title=f"{symbol} Stock Price & Moving Averages", template='plotly_dark')
# fig.show()

# # Trading Volume Insights
# fig = px.line(df, x=df.index, y="Volume", title=f"{symbol} Trading Volume Over Time")
# fig.update_layout(template='plotly_dark')
# fig.show()

# # Stock Price Prediction (Linear Regression)
# df["Target"] = df["Close"].shift(-5)  # Predict 5 days ahead
# df.dropna(inplace=True)

# X = df[["Open", "High", "Low", "Close"]]
# y = df["Target"]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
# print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# fig = go.Figure()
# fig.add_trace(go.Scatter(y=y_test.values, mode='lines', name='Actual'))
# fig.add_trace(go.Scatter(y=y_pred, mode='lines', name='Predicted', line=dict(dash='dash')))
# fig.update_layout(title="Stock Price Prediction", template='plotly_dark')
# fig.show()

# # Market Correlation Analysis
# symbols = ["AAPL", "MSFT", "GOOGL"]
# stock_data = {sym: get_stock_data(sym)["Close"] for sym in symbols}
# df_corr = pd.DataFrame(stock_data)

# fig = px.imshow(df_corr.corr(), text_auto=True, title="Stock Correlation Matrix", color_continuous_scale='RdBu')
# fig.show()

# # Stock Performance Comparison
# returns = df_corr.pct_change(fill_method=None).mean() * 252  # Annualized Returns
# volatility = df_corr.pct_change(fill_method=None).std() * np.sqrt(252)  # Annualized Volatility

# performance_df = pd.DataFrame({"Returns": returns, "Volatility": volatility})

# fig = px.scatter(performance_df, x="Volatility", y="Returns", text=performance_df.index, size=[10]*len(performance_df),
#                  title="Stock Performance (Risk vs. Return)")
# fig.update_traces(textposition='top center')
# fig.show()
