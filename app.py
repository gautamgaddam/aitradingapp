import streamlit as st
from datetime import date
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import ta
import pandas_ta as pta
from scipy.stats import linregress  # Import linregress
from streamlit_option_menu import option_menu
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import hashlib






# Set wide mode as default layout
st.set_page_config(layout="wide")
base="light"
primaryColor="#4682b4"

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="e-Trade",
        options=["Login","Markets" ,"Stock Screener", "Technical Analysis","Watch List","Stock Price Forecasting"],
        menu_icon="list",
        default_index=0,  # Default to Interactive charts
    )


## page 1-------------------------------------------------------------------------------------------------------------------
# Function to download data and calculate moving averages
def get_stock_data(ticker_symbol, start_date, end_date):
    data = yf.download(ticker_symbol, start=start_date, end=end_date)
    data['MA_15'] = data['Close'].rolling(window=15).mean()
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data.dropna(inplace=True)
    return data

# Function to create Plotly figure
def create_figure(data, indicators, title):
    fig = go.Figure()
    if 'Close' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))

    if 'MA_15' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=data['MA_15'], mode='lines', name='15-day MA'))

    if 'MA_50' in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=data['MA_50'], mode='lines', name='50-day MA'))

    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price',
                      xaxis_rangeslider_visible=True,
                      plot_bgcolor='dark grey',
                      paper_bgcolor='white',
                      font=dict(color='black'),
                      hovermode='x',
                      xaxis=dict(rangeselector=dict(buttons=list([
                          dict(count=1, label="1m", step="month", stepmode="backward"),
                          dict(count=6, label="6m", step="month", stepmode="backward"),
                          dict(count=1, label="YTD", step="year", stepmode="todate"),
                          dict(count=1, label="1y", step="year", stepmode="backward"),
                          dict(step="all")
                      ])),
                                 rangeslider=dict(visible=True),
                                 type='date'),
                      yaxis=dict(fixedrange=False),
                      updatemenus=[dict(type="buttons",
                                        buttons=[dict(label="Reset Zoom",
                                                      method="relayout",
                                                      args=[{"xaxis.range": [None, None],
                                                             "yaxis.range": [None, None]}])])])
    return fig

# Main application logic
if selected == "Markets":
    st.sidebar.subheader("Markets")
    submenu = st.sidebar.radio(
        "Select Option",
        ["Equities", "Commodities", "Currencies", "Cryptocurrencies"]
    )

    # Create two columns
    col1, col2 = st.columns(2)

    # Set up the start and end date inputs
    with col1:
        START = st.date_input('Start Date', pd.to_datetime("2015-01-01"))

    with col2:
        END = st.date_input('End Date', pd.to_datetime("today"))

    if submenu == "Equities":
        st.header("Equity Markets")

        # Download data
        data_nyse = get_stock_data("^NYA", START, END)
        data_bse = get_stock_data("^BSESN", START, END)

        # Multi-select option to select indicators
        indicators = st.multiselect("Select Indicators", ['Close', 'MA_15', 'MA_50'], default=['Close'])

        # Create figures
        fig_nyse = create_figure(data_nyse, indicators, 'NYSE Price')
        fig_bse = create_figure(data_bse, indicators, 'BSE Price')

        # Display plots in two columns
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(fig_nyse)
            
            st.subheader(" Insights (NYSE):")
            st.divider()
            if data_nyse['MA_15'].iloc[-1] < data_nyse['MA_50'].iloc[-1]:
                st.markdown("* Market sentement is  **Bearish**")
            elif data_nyse['MA_15'].iloc[-1] > data_nyse['MA_50'].iloc[-1]:
                st.markdown("* Market sentement is **Bullish**")
            
        with col2:
            st.plotly_chart(fig_bse)
            st.subheader("Insights (SENSEX):")
            st.divider()
            
            if data_bse['MA_15'].iloc[-1] < data_bse['MA_50'].iloc[-1]:
                st.markdown("* Market sentement is **Bearish**")
            elif data_bse['MA_15'].iloc[-1] > data_bse['MA_50'].iloc[-1]:
                st.markdown("* Market sentiment is **Bullish**")

    elif submenu == "Commodities":
        st.header("Commodities")

        # Define the list of tickers
        tickers = ["GC=F", "CL=F", "NG=F", "SI=F", "HG=F"]

        # Multi-select option to select tickers
        selected_tickers = st.multiselect("Select stock tickers to visualize", tickers, default=["GC=F", "CL=F"])

        # Multi-select option to select indicators
        indicators = st.multiselect("Select Indicators", ['Close', 'MA_15', 'MA_50'], default=['Close'])

        # Ensure at least one ticker is selected
        if not selected_tickers:
            st.warning("Please select at least one ticker.")
        else:
            # Download data and create figures for selected tickers
            columns = st.columns(len(selected_tickers))
            for ticker, col in zip(selected_tickers, columns):
                data = get_stock_data(ticker, START, END)
                fig = create_figure(data, indicators, f'{ticker} Price')
                col.plotly_chart(fig)

    elif submenu == "Currencies":
        st.header("Currencies")

        # Define the list of tickers
        tickers = ["EURUSD=X", "GBPUSD=X", "CNYUSD=X", "INRUSD=X"]

        # Multi-select option to select currency pairs
        selected_tickers = st.multiselect("Select currency pairs to visualize", tickers, default=["INRUSD=X", "CNYUSD=X"])

        # Multi-select option to select indicators
        indicators = st.multiselect("Select Indicators", ['Close', 'MA_15', 'MA_50'], default=['Close'])

        # Ensure at least one currency pair is selected
        if not selected_tickers:
            st.warning("Please select at least one currency pair.")
        else:
            # Download data and create figures for selected currency pairs
            columns = st.columns(len(selected_tickers))
            for ticker, col in zip(selected_tickers, columns):
                data = get_stock_data(ticker, START, END)
                fig = create_figure(data, indicators, f'{ticker} Price')
                col.plotly_chart(fig)

    elif submenu == "Cryptocurrencies":
        st.header("Cryptocurrencies")

        # Define the list of tickers
        tickers = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]

        # Multi-select option to select cryptocurrencies
        selected_tickers = st.multiselect("Select cryptocurrencies to visualize", tickers, default=["BTC-USD", "ETH-USD"])

        # Multi-select option to select indicators
        indicators = st.multiselect("Select Indicators", ['Close', 'MA_15', 'MA_50'], default=['Close'])

        # Ensure at least one cryptocurrency is selected
        if not selected_tickers:
            st.warning("Please select at least one cryptocurrency.")
        else:
            # Download data and create figures for selected cryptocurrencies
            columns = st.columns(len(selected_tickers))
            for ticker, col in zip(selected_tickers, columns):
                data = get_stock_data(ticker, START, END)
                fig = create_figure(data, indicators, f'{ticker} Price')
                col.plotly_chart(fig)

## Page 2---------------------------------------------------------------------------------------------------------------------------------------------------------------------


# Function to fetch and process stock data
def get_stock_data(ticker_symbols, start_date, end_date):
    try:
        stock_data = {}
        for ticker_symbol in ticker_symbols:
            df = yf.download(ticker_symbol, start=start_date, end=end_date)
            print(f"Downloaded data for {ticker_symbol}: Shape = {df.shape}")
            df.interpolate(method='linear', inplace=True)
            df = calculate_indicators(df)
            df.dropna(inplace=True)
            print(f"Processed data for {ticker_symbol}: Shape = {df.shape}")
            stock_data[ticker_symbol] = df
        combined_df = pd.concat(stock_data.values(), axis=1)
        combined_df.columns = ['_'.join([ticker, col]).strip() for ticker, df in stock_data.items() for col in df.columns]
        return combined_df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

# Function to calculate technical indicators
def calculate_indicators(df):
    # Calculate Moving Averages
    df['5_MA'] = df['Close'].rolling(window=5).mean()
    df['20_MA'] = df['Close'].rolling(window=20).mean()
    df['50_MA'] = df['Close'].rolling(window=50).mean()
    
    # Calculate MACD
    df['MACD'] = ta.trend.MACD(df['Close']).macd()
    df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Calculate ADX
    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
    
    # Calculate Parabolic SAR
    psar = pta.psar(df['High'], df['Low'], df['Close'])
    df['Parabolic_SAR'] = psar['PSARl_0.02_0.2']
    
    # Calculate RSI
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    
    # Calculate Volume Moving Average (20 days)
    df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()

    # Calculate Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'])
    df['Bollinger_High'] = bollinger.bollinger_hband()
    df['Bollinger_Low'] = bollinger.bollinger_lband()
    df['Bollinger_Middle'] = bollinger.bollinger_mavg()
    
    return df

# Function to query the stocks
def query_stocks(df, conditions, tickers):
    results = []
    for ticker in tickers:
        condition_met = True
        for condition in conditions:
            col1, op, col2 = condition
            col1 = f"{ticker}_{col1}"
            col2 = f"{ticker}_{col2}"
            if col1 not in df.columns or col2 not in df.columns:
                condition_met = False
                break
            if op == '>':
                if not (df[col1] > df[col2]).iloc[-1]:
                    condition_met = False
                    break
            elif op == '<':
                if not (df[col1] < df[col2]).iloc[-1]:
                    condition_met = False
                    break
            elif op == '>=':
                if not (df[col1] >= df[col2]).iloc[-1]:
                    condition_met = False
                    break
            elif op == '<=':
                if not (df[col1] <= df[col2]).iloc[-1]:
                    condition_met = False
                    break
        if condition_met:
            row = {
                'Ticker': ticker,
                'MACD': df[f"{ticker}_MACD"].iloc[-1],
                'MACD_Signal': df[f"{ticker}_MACD_Signal"].iloc[-1],
                'RSI': df[f"{ticker}_RSI"].iloc[-1],
                'ADX': df[f"{ticker}_ADX"].iloc[-1],
                'Close': df[f"{ticker}_Close"].iloc[-1],
                '5_MA': df[f"{ticker}_5_MA"].iloc[-1],
                '20_MA': df[f"{ticker}_20_MA"].iloc[-1],
                'Bollinger_High': df[f"{ticker}_Bollinger_High"].iloc[-1],
                'Bollinger_Low': df[f"{ticker}_Bollinger_Low"].iloc[-1],
                'Bollinger_Middle': df[f"{ticker}_Bollinger_Middle"].iloc[-1],
                'Parabolic_SAR': df[f"{ticker}_Parabolic_SAR"].iloc[-1],
                'Volume': df[f"{ticker}_Volume"].iloc[-1],
                'Volume_MA_20': df[f"{ticker}_Volume_MA_20"].iloc[-1]
            }
            results.append(row)
    return pd.DataFrame(results)

# Function to create Plotly figure
def create_figure(data, indicators, title):
    fig = go.Figure()
    for indicator in indicators:
        fig.add_trace(go.Scatter(x=data.index, y=data[indicator], mode='lines', name=indicator))
    
    fig.add_trace(go.Bar(x=data.index, y=data[f"{selected_stock}_MACD_Histogram"], name='MACD_Histogram', marker_color='gray'))
    fig.update_layout(title=title, xaxis_title='Date', yaxis_title='Price', xaxis_rangeslider_visible=True)
    return fig

# Check if "Stock Scrrener" is selected
if selected == "Stock Screener":
    st.sidebar.subheader("Screens")

    submenu = st.sidebar.radio(
        "Select Option",
        ["LargeCap-1", "LargeCap-2","LargeCap-3","MidCap", "SmallCap"]
        
    )
    # Define ticker symbols for different market caps
    largecap3_tickers = ["ITC.NS", "JBCHEPHARM.BO", "JWL.BO", "JYOTHYLAB.BO", "KPRMILL.NS", "KAJARIACER.NS", "KEI.BO", "LTTS.NS", "LTIM.NS", "MANKIND.NS", "MARICO.NS", "METROBRAND.BO", "MOTILALOFS.NS", "MPHASIS.NS", "MUTHOOTFIN.NS", "NH.NS", "NAVINFLUOR.NS", "NAM-INDIA.BO", "NMDC.NS", "OFSS.NS", "PGHH.NS", "PIIND.NS", "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS", "PFIZER.NS", "PIDILITIND.NS", "POLYMED.NS", "POLYCAB.NS", "RRKABEL.NS", "RVNL.NS", "RATNAMANI.NS", "RITES.NS", "SANOFI.NS", "SCHAEFFLER.NS", "SKFINDIA.NS", "SOLARINDS.NS", "SONACOMS.NS", "SUMICHEM.NS", "SUNTV.NS", "SUNDRMFAST.NS", "SUPREMEIND.BO", "TATAELXSI.NS", "TATATECH.NS", "TCS.NS", "TECHM.NS", "TIMKEN.NS", "TITAN.NS", "TRITURBINE.NS", "TIINDIA.NS", "UNITDSPR.BO", "VGUARD.NS", "MANYAVAR.NS", "VINATIORGA.NS", "WIPRO.NS","ZYDUSLIFE.NS"]
    largecap2_tickers = ["CUMMINSIND.NS", "CYIENT.NS", "DATAPATTNS.NS", "DEEPAKNTR.NS", "DIVISLAB.NS", "LALPATHLAB.NS", "RDY", "ELGIEQUIP.NS", "EMAMILTD.NS", "FIVESTAR.BO", "GRINFRA.NS", "GILLETTE.NS", "GLAXO.NS", "GODFRYPHLP.NS", "GRINDWELL.NS", "HAVELLS.NS", "HCLTECH.NS", "HAL.BO", "HONAUT.BO", "IRCTC.NS", "ISEC.BO", "INFY.NS", "IPCALAB.BO"]
   
    largecap1_tickers = ["ABBOTINDIA.NS", "ADANIPOWER.NS", "AFFLE.BO", "AIAENG.BO", "AJANTPHARM.BO", "APLLTD.BO", "ALKEM.BO", "ARE&M.NS", "ANANDRATHI.BO", "APARINDS.BO", "ASIANPAINT.NS", "ASTRAL.NS", "ASTRAZEN.NS", "BAJFINANCE.NS", "BASF.NS", "BAYERCROP.BO", "BERGEPAINT.BO", "BDL.NS", "BEL.NS", "BSOFT.BO", "CDSL.NS", "CAMS.NS", "CARBORUNIV.BO", "CASTROLIND.NS", "CHAMBLFERT.BO", "COALINDIA.NS", "COFORGE.BO", "COLPAL.NS", "CONCORDBIO.BO", "COROMANDEL.BO", "CREDITACC.BO" ]
    
    
    smallcap_tickers = ["TAPARIA.BO", "LKPFIN.BO", "EQUITAS.NS"]
    
    
    midcap_tickers = ["PNCINFRA.NS","INDIASHLTR.NS","RAYMOND.NS","KAMAHOLD.BO","BENGALASM.BO","CHOICEIN.NS","GRAVITA.NS","HGINFRA.NS","JKPAPER.NS","MTARTECH.NS","HAPPSTMNDS.NS","SARDAEN.NS","WELENT.NS","LTFOODS.NS","GESHIP.NS","SHRIPISTON.NS","SHAREINDIA.NS","CYIENTDLM.NS","VTL.NS","EASEMYTRIP.NS","LLOYDSME.NS","ROUTE.NS","VAIBHAVGBL.NS","GOKEX.NS","USHAMART.NS","EIDPARRY.NS","KIRLOSBROS.NS","MANINFRA.NS","CMSINFO.NS","RALLIS.NS","GHCL.NS","NEULANDLAB.NS","SPLPETRO.NS","MARKSANS.NS","NAVINFLUOR.NS","ELECON.NS","TANLA.NS","KFINTECH.NS","TIPSINDLTD.NS","ACI.NS","SURYAROSNI.NS","GPIL.NS","GMDCLTD.NS","MAHSEAMLES.NS","TDPOWERSYS.NS","TECHNOE.NS","JLHL.NS"]
    

    # Create two columns
    col1, col2 = st.columns(2)

    # Set up the start and end date inputs
    with col1:
        START = st.date_input('Start Date', pd.to_datetime("2015-01-01"))

    with col2:
        END = st.date_input('End Date', pd.to_datetime("today"))

    if submenu == "LargeCap-1":
            st.header("LargeCap-1")
            tickers = largecap1_tickers

    if submenu == "LargeCap-2":
            st.header("LargeCap-2")
            tickers = largecap2_tickers

    if submenu == "LargeCap-3":
            st.header("LargeCap-3")
            tickers = largecap3_tickers

    if submenu == "MidCap":
            st.header("MidCap")
            tickers = midcap_tickers
    if submenu == "SmallCap":
            st.header("SmallCap")
            tickers = smallcap_tickers


    # Fetch data and calculate indicators for each stock
    stock_data = get_stock_data(tickers, START, END)

    # Define first set of conditions
    first_conditions = [
        ('MACD', '>', 'MACD_Signal'),
        ('Volume', '>', 'Volume_MA_20'),
    ]

    # Query stocks based on the first set of conditions
    first_query_df = query_stocks(stock_data, first_conditions, tickers)

    # Display the final results
    st.write("Stocks in an uptrend with high volume:")
    st.dataframe(first_query_df.round(2))
    # Generate insights
    recommendation_df=first_query_df[(first_query_df['RSI']<70)&(first_query_df['ADX']>20)]
    st.subheader("Stock Recommendation:")
    st.dataframe(recommendation_df['Ticker'])
    
    # Dropdown for stock selection
    st.subheader("Analysis:")
    selected_stock = st.selectbox("Select Stock for Analysis", first_query_df['Ticker'].tolist())

    # If a stock is selected, plot its data with the selected indicators
    if selected_stock:
        selected_stock_data = stock_data[[col for col in stock_data.columns if selected_stock in col]]
        indicators = st.multiselect(
            "Select Indicators",
            ['Close', '5_MA', '20_MA', '50_MA', 'MACD', 'MACD_Signal', 'RSI', 'ADX', 'Bollinger_High', 'Bollinger_Low', 'Bollinger_Middle'],
            default=['Close']
        )
        timeframe = st.radio(
            "Select Timeframe",
            ['15 days', '30 days', '90 days', '180 days', '1 year', 'All'],
            
            index=4,
            horizontal=True
        )

        if timeframe == '15 days':
            selected_stock_data = selected_stock_data[-15:]
        elif timeframe == '30 days':
            selected_stock_data = selected_stock_data[-30:]
        elif timeframe == '90 days':
            selected_stock_data = selected_stock_data[-90:]
        elif timeframe == '180 days':
            selected_stock_data = selected_stock_data[-180:]
        elif timeframe == '1 year':
            selected_stock_data = selected_stock_data[-365:]

        fig = create_figure(selected_stock_data, [f'{selected_stock}_{ind}' for ind in indicators], f'{selected_stock} Trend Analysis')
        st.plotly_chart(fig)

## Page 3---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Check if "Technical Analysis" is selected
if selected == "Technical Analysis":
    st.sidebar.subheader("Interactive Charts")
    submenu = st.sidebar.radio(
        "Select Option",
        ["Trend Analysis", "Volume Analysis", "Support & Resistance Levels"]
    )

    # Create three columns
    col1, col2, col3 = st.columns(3)

    # Set up the start and end date inputs
    with col1:
        # List of stock symbols
        stock_symbols = ["ABBOTINDIA.NS", "ADANIPOWER.NS", "AFFLE.BO", "AIAENG.BO", "AJANTPHARM.BO", "APLLTD.BO", "ALKEM.BO", "ARE&M.NS", "ANANDRATHI.BO", "APARINDS.BO", "ASIANPAINT.NS", "ASTRAL.NS", "ASTRAZEN.NS", "BAJFINANCE.NS", "BASF.NS", "BAYERCROP.BO", "BERGEPAINT.BO", "BDL.NS", "BEL.NS", "BSOFT.BO", "CDSL.NS", "CAMS.NS", "CARBORUNIV.BO", "CASTROLIND.NS", "CHAMBLFERT.BO", "COALINDIA.NS", "COFORGE.BO", "COLPAL.NS", "CONCORDBIO.BO", "COROMANDEL.BO", "CREDITACC.BO", "CUMMINSIND.NS", "CYIENT.NS", "DATAPATTNS.NS", "DEEPAKNTR.NS", "DIVISLAB.NS", "LALPATHLAB.NS", "RDY", "ELGIEQUIP.NS", "EMAMILTD.NS", "FIVESTAR.BO", "GRINFRA.NS", "GILLETTE.NS", "GLAXO.NS", "GODFRYPHLP.NS", "GRINDWELL.NS", "HAVELLS.NS", "HCLTECH.NS", "HAL.BO", "HONAUT.BO", "IRCTC.NS", "ISEC.BO", "INFY.NS", "IPCALAB.BO", "ITC.NS", "JBCHEPHARM.BO", "JWL.BO", "JYOTHYLAB.BO", "KPRMILL.NS", "KAJARIACER.NS", "KEI.BO", "LTTS.NS", "LTIM.NS", "MANKIND.NS", "MARICO.NS", "METROBRAND.BO", "MOTILALOFS.NS", "MPHASIS.NS", "MUTHOOTFIN.NS", "NH.NS", "NAVINFLUOR.NS", "NAM-INDIA.BO", "NMDC.NS", "OFSS.NS", "PGHH.NS", "PIIND.NS", "PAGEIND.NS", "PERSISTENT.NS", "PETRONET.NS", "PFIZER.NS", "PIDILITIND.NS", "POLYMED.NS", "POLYCAB.NS", "RRKABEL.NS", "RVNL.NS", "RATNAMANI.NS", "RITES.NS", "SANOFI.NS", "SCHAEFFLER.NS", "SKFINDIA.NS", "SOLARINDS.NS", "SONACOMS.NS", "SUMICHEM.NS", "SUNTV.NS", "SUNDRMFAST.NS", "SUPREMEIND.BO", "TATAELXSI.NS", "TATATECH.NS", "TCS.NS", "TECHM.NS", "TIMKEN.NS", "TITAN.NS", "TRITURBINE.NS", "TIINDIA.NS", "UNITDSPR.BO", "VGUARD.NS", "MANYAVAR.NS", "VINATIORGA.NS", "WIPRO.NS","ZYDUSLIFE.NS"]
        #ticker = st.text_input("Enter Stock symbol", '^BSESN')
        # Auto-suggestion using selectbox
        ticker = st.selectbox("Enter Stock symbol", stock_symbols)
        st.write(f"You selected: {ticker}")
    with col2:
        START = st.date_input('Start Date', pd.to_datetime("2015-01-01"))
    with col3:
        END = st.date_input('End Date', pd.to_datetime("today"))
    
    # Load data
    def load_data(ticker):
        df = yf.download(ticker, START, END)
        df.reset_index(inplace=True)
        return df

    # Handle null values
    def interpolate_dataframe(df):
        if df.isnull().values.any():
            df = df.interpolate()
        return df

    # Load data for the given ticker
    df = load_data(ticker)

    if df.empty:
        st.write("No data available for the provided ticker.")
    else:
        df = interpolate_dataframe(df)

        # Ensure enough data points for the calculations
        if len(df) > 200:  # 200 is the maximum window size used in calculations
            # Calculate Moving Averages
            df['15_MA'] = df['Close'].rolling(window=15).mean()
            df['20_MA'] = df['Close'].rolling(window=20).mean()
            df['50_MA'] = df['Close'].rolling(window=50).mean()
            df['200_MA'] = df['Close'].rolling(window=200).mean()
            
            # Calculate MACD
            df['MACD'] = ta.trend.MACD(df['Close']).macd()
            df['MACD_Signal'] = ta.trend.MACD(df['Close']).macd_signal()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Calculate ADX
            df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
            
            # Calculate Parabolic SAR using 'pandas_ta' library
            psar = pta.psar(df['High'], df['Low'], df['Close'])
            df['Parabolic_SAR'] = psar['PSARl_0.02_0.2']
            
            # Calculate RSI
            df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
            
            # Calculate Volume Moving Average (20 days)
            df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
            
            # Calculate On-Balance Volume (OBV)
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
            
            # Calculate Volume Oscillator (20-day EMA and 50-day EMA)
            df['Volume_EMA_20'] = ta.trend.EMAIndicator(df['Volume'], window=20).ema_indicator()
            df['Volume_EMA_50'] = ta.trend.EMAIndicator(df['Volume'], window=50).ema_indicator()
            df['Volume_Oscillator'] = df['Volume_EMA_20'] - df['Volume_EMA_50']
            
            # Calculate Chaikin Money Flow (20 days)
            df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume']).chaikin_money_flow()

            # Identify Horizontal Support and Resistance
            def find_support_resistance(df, window=20):
                df['Support'] = df['Low'].rolling(window, center=True).min()
                df['Resistance'] = df['High'].rolling(window, center=True).max()
                return df

            df = find_support_resistance(df)

            # Draw Trendlines
            def calculate_trendline(df, kind='support'):
                if kind == 'support':
                    prices = df['Low']
                elif kind == 'resistance':
                    prices = df['High']
                else:
                    raise ValueError("kind must be either 'support' or 'resistance'")

                indices = np.arange(len(prices))
                slope, intercept, _, _, _ = linregress(indices, prices)
                trendline = slope * indices + intercept
                return trendline

            df['Support_Trendline'] = calculate_trendline(df, kind='support')
            df['Resistance_Trendline'] = calculate_trendline(df, kind='resistance')

            # Calculate Fibonacci Retracement Levels
            def fibonacci_retracement_levels(high, low):
                diff = high - low
                levels = {
                    'Level_0': high,
                    'Level_0.236': high - 0.236 * diff,
                    'Level_0.382': high - 0.382 * diff,
                    'Level_0.5': high - 0.5 * diff,
                    'Level_0.618': high - 0.618 * diff,
                    'Level_1': low
                }
                return levels

            recent_high = df['High'].max()
            recent_low = df['Low'].min()
            fib_levels = fibonacci_retracement_levels(recent_high, recent_low)

            # Calculate Pivot Points
            def pivot_points(df):
                df['Pivot'] = (df['High'].shift(1) + df['Low'].shift(1) + df['Close'].shift(1)) / 3
                df['R1'] = 2 * df['Pivot'] - df['Low'].shift(1)
                df['S1'] = 2 * df['Pivot'] - df['High'].shift(1)
                df['R2'] = df['Pivot'] + (df['High'].shift(1) - df['Low'].shift(1))
                df['S2'] = df['Pivot'] - (df['High'].shift(1) - df['Low'].shift(1))
                return df

            df = pivot_points(df)

            # Calculate Bollinger Bands
            bollinger = ta.volatility.BollingerBands(df['Close'])
            df['Bollinger_High'] = bollinger.bollinger_hband()
            df['Bollinger_Low'] = bollinger.bollinger_lband()
            df['Bollinger_Middle'] = bollinger.bollinger_mavg()  # Middle band is typically the SMA

            # Generate buy/sell signals
            df['Buy_Signal'] = (df['MACD'] > df['MACD_Signal']) & (df['ADX'] > 20)
            df['Sell_Signal'] = (df['Close'] < df['15_MA'])

            # Create a new column 'Signal' based on 'Buy_Signal' and 'Sell_Signal' conditions
            def generate_signal(row):
                if row['Buy_Signal']:
                    return 'Buy'
                elif row['Sell_Signal']:
                    return 'Sell'
                else:
                    return 'Hold'

            df['Signal'] = df.apply(generate_signal, axis=1)
            
            if submenu == "Trend Analysis":
                st.header("Trend Analysis")

                indicators = st.multiselect(
                    "Select Indicators",
                    ['Close', '20_MA', '50_MA', '200_MA', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'RSI','Signal','ADX','Parabolic_SAR','Bollinger_High','Bollinger_Low','Bollinger_Middle'],
                    default=['Close','Signal']
                )
                timeframe = st.radio(
                    "Select Timeframe",
                    ['15 days', '30 days', '90 days', '180 days', '1 year', 'All'],
                    index=4,
                    horizontal=True
                )

                if timeframe == '15 days':
                    df = df[-15:]
                elif timeframe == '30 days':
                    df = df[-30:]
                elif timeframe == '90 days':
                    df = df[-90:]
                elif timeframe == '180 days':
                    df = df[-180:]
                elif timeframe == '1 year':
                    df = df[-365:]

                fig = go.Figure()
                colors = {'Close': 'blue', '20_MA': 'orange', '50_MA': 'green', '200_MA': 'red', 'MACD': 'purple', 'MACD_Signal': 'brown', 'RSI': 'pink','Signal': 'black','ADX': 'magenta','Parabolic_SAR': 'yellow','Bollinger_High': 'black','Bollinger_Low': 'cyan','Bollinger_Middle': 'grey'}

                for indicator in indicators:
                    if indicator == 'Signal':
                        # Plot buy and sell signals
                        buy_signals = df[df['Signal'] == 'Buy']
                        sell_signals = df[df['Signal'] == 'Sell']
                        fig.add_trace(go.Scatter(x=buy_signals['Date'], y=buy_signals['Close'], mode='markers', name='Buy Signal', marker=dict(color='green', symbol='triangle-up')))
                        fig.add_trace(go.Scatter(x=sell_signals['Date'], y=sell_signals['Close'], mode='markers', name='Sell Signal', marker=dict(color='red', symbol='triangle-down')))
                    elif indicator == 'MACD_Histogram':
                        fig.add_trace(go.Bar(x=df['Date'], y=df[indicator], name=indicator, marker_color='gray'))
                    else:
                        fig.add_trace(go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator, line=dict(color=colors.get(indicator, 'black'))))

                st.plotly_chart(fig)
                
                # Generate insights
                st.subheader("Insights:")
                if 'MACD' in indicators and 'MACD_Signal' in indicators:
                    last_macd = df['MACD'].iloc[-1]
                    last_macd_signal = df['MACD_Signal'].iloc[-1]
                    if last_macd > last_macd_signal:
                        st.markdown("MACD is above the MACD Signal line - Bullish Signal")
                    else:
                        st.markdown("MACD is below the MACD Signal line - Bearish Signal")

                if 'RSI' in indicators:
                    last_rsi = df['RSI'].iloc[-1]
                    if last_rsi > 70:
                        st.markdown("RSI is above 70 - Overbought")
                    elif last_rsi < 30:
                        st.markdown("RSI is below 30 - Oversold")

                if 'ADX' in indicators:
                    last_adx = df['ADX'].iloc[-1]
                    if last_adx > 20:
                        st.markdown("ADX is above 20 - Strong Trend")
                    else:
                        st.markdown("ADX is below 20 - Weak Trend")

                if 'Parabolic_SAR' in indicators:
                    last_close = df['Close'].iloc[-1]
                    last_psar = df['Parabolic_SAR'].iloc[-1]
                    if last_close > last_psar:
                        st.markdown("Price is above Parabolic SAR - Bullish Signal")
                    else:
                        st.markdown("Price is below Parabolic SAR - Bearish Signal")

                if 'Bollinger_High' in indicators and 'Bollinger_Low' in indicators:
                    last_close = df['Close'].iloc[-1]
                    last_boll_high = df['Bollinger_High'].iloc[-1]
                    last_boll_low = df['Bollinger_Low'].iloc[-1]
                    if last_close > last_boll_high:
                        st.markdown("Price is above the upper Bollinger Band - Potentially Overbought")
                    elif last_close < last_boll_low:
                        st.markdown("Price is below the lower Bollinger Band - Potentially Oversold")

            elif submenu == "Volume Analysis":
                st.header("Volume Analysis")
                volume_indicators = st.multiselect(
                    "Select Volume Indicators",
                    ['Volume', 'Volume_MA_20', 'OBV', 'Volume_Oscillator', 'CMF'],
                    default=['Volume']
                )
                volume_timeframe = st.radio(
                    "Select Timeframe",
                    ['15 days', '30 days', '90 days', '180 days', '1 year', 'All'],
                    index=4,
                    horizontal=True
                )

                if volume_timeframe == '15 days':
                    df = df[-15:]
                elif volume_timeframe == '30 days':
                    df = df[-30:]
                elif volume_timeframe == '90 days':
                    df = df[-90:]
                elif volume_timeframe == '180 days':
                    df = df[-180:]
                elif volume_timeframe == '1 year':
                    df = df[-365:]

                fig = go.Figure()
                for indicator in volume_indicators:
                    fig.add_trace(go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator))

                st.plotly_chart(fig)

                # Generate insights
                st.subheader("Insights:")
                if 'Volume' in volume_indicators and 'Volume_MA_20' in volume_indicators:
                    last_volume = df['Volume'].iloc[-1]
                    last_volume_ma_20 = df['Volume_MA_20'].iloc[-1]
                    if last_volume > last_volume_ma_20:
                        st.markdown("Current volume is above the 20-day average - Increased buying/selling interest")
                    else:
                        st.markdown("Current volume is below the 20-day average - Decreased buying/selling interest")

                if 'OBV' in volume_indicators:
                    last_obv = df['OBV'].iloc[-1]
                    if last_obv > df['OBV'].iloc[-2]:
                        st.markdown("OBV is increasing - Accumulation phase (buying pressure)")
                    else:
                        st.markdown("OBV is decreasing - Distribution phase (selling pressure)")

                if 'CMF' in volume_indicators:
                    last_cmf = df['CMF'].iloc[-1]
                    if last_cmf > 0:
                        st.markdown("CMF is positive - Buying pressure")
                    else:
                        st.markdown("CMF is negative - Selling pressure")

            elif submenu == "Support & Resistance Levels":
                st.header("Support & Resistance Levels")
                sr_indicators = st.multiselect(
                    "Select Indicators",
                    ['Close', '20_MA', '50_MA', '200_MA', 'Support', 'Resistance', 'Support_Trendline', 'Resistance_Trendline', 'Pivot', 'R1', 'S1', 'R2', 'S2'],
                    default=['Close']
                )
                sr_timeframe = st.radio(
                    "Select Timeframe",
                    ['15 days', '30 days', '90 days', '180 days', '1 year', 'All'],
                    index=4,
                    horizontal=True
                )

                if sr_timeframe == '15 days':
                    df = df[-15:]
                elif sr_timeframe == '30 days':
                    df = df[-30:]
                elif sr_timeframe == '90 days':
                    df = df[-90:]
                elif sr_timeframe == '180 days':
                    df = df[-180:]
                elif sr_timeframe == '1 year':
                    df = df[-365:]

                fig = go.Figure()
                for indicator in sr_indicators:
                    fig.add_trace(go.Scatter(x=df['Date'], y=df[indicator], mode='lines', name=indicator))

                st.plotly_chart(fig)

                # Generate insights
                st.subheader("Insights:")
                if 'Support' in sr_indicators and 'Resistance' in sr_indicators:
                    last_close = df['Close'].iloc[-1]
                    last_support = df['Support'].iloc[-1]
                    last_resistance = df['Resistance'].iloc[-1]
                    if last_close > last_resistance:
                        st.markdown("Price is above the resistance level - Potential breakout")
                    elif last_close < last_support:
                        st.markdown("Price is below the support level - Potential breakdown")

                if 'Pivot' in sr_indicators:
                    last_close = df['Close'].iloc[-1]
                    last_pivot = df['Pivot'].iloc[-1]
                    if last_close > last_pivot:
                        st.markdown("Price is above the pivot point - Bullish sentiment")
                    else:
                        st.markdown("Price is below the pivot point - Bearish sentiment")

        else:
            st.write("Not enough data points for technical analysis.")

## page4 ---------------------------------------------------------------------------------------

# Initialize session state
if 'watchlists' not in st.session_state:
    st.session_state['watchlists'] = {f"Watchlist {i}": [] for i in range(1, 11)}

# Function to fetch stock data
def get_stock_data(ticker):
    df = yf.download(ticker, period='1y')
    df['2_MA'] = df['Close'].rolling(window=2).mean()
    df['15_MA'] = df['Close'].rolling(window=15).mean()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
    return df[['Close', '2_MA', '15_MA', 'RSI', 'ADX']].dropna()

# Function to update watchlist
def update_watchlist(watchlist_name, ticker):
    if ticker not in st.session_state['watchlists'][watchlist_name]:
        st.session_state['watchlists'][watchlist_name].append(ticker)



# Watch List Page

# Initialize session state
if 'watchlists' not in st.session_state:
    st.session_state['watchlists'] = {f"Watchlist {i}": [] for i in range(1, 11)}

# Function to fetch stock data
def get_stock_data(ticker):
    try:
        df = yf.download(ticker, period='1y')
        if df.empty:
            st.warning(f"No data found for {ticker}.")
            return pd.DataFrame()  # Return an empty DataFrame
        df['2_MA'] = df['Close'].rolling(window=2).mean()
        df['15_MA'] = df['Close'].rolling(window=15).mean()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
        return df[['Close', '2_MA', '15_MA', 'RSI', 'ADX']].dropna()
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame

# Function to update watchlist
def update_watchlist(watchlist_name, ticker):
    if ticker not in st.session_state['watchlists'][watchlist_name]:
        if len(st.session_state['watchlists'][watchlist_name]) < 10:
            st.session_state['watchlists'][watchlist_name].append(ticker)
        else:
            st.warning(f"{watchlist_name} already has 10 stocks. Remove a stock before adding a new one.")
    else:
        st.warning(f"{ticker} is already in {watchlist_name}.")



# Watch List Page

# Initialize session state for watchlists
if 'watchlists' not in st.session_state:
    st.session_state['watchlists'] = {f"Watchlist {i}": [] for i in range(1, 11)}

# Function to fetch stock data
def get_stock_data(ticker):
    try:
        df = yf.download(ticker, period='1y')
        if df.empty:
            st.warning(f"No data found for {ticker}.")
            return pd.DataFrame()  # Return an empty DataFrame
        df['2_MA'] = df['Close'].rolling(window=2).mean()
        df['15_MA'] = df['Close'].rolling(window=15).mean()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
        return df[['Close', '2_MA', '15_MA', 'RSI', 'ADX']].iloc[-1]
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return pd.Series(dtype='float64')  # Return an empty Series

# Function to update watchlist
def update_watchlist(watchlist_name, ticker):
    if ticker not in st.session_state['watchlists'][watchlist_name]:
        if len(st.session_state['watchlists'][watchlist_name]) < 10:
            st.session_state['watchlists'][watchlist_name].append(ticker)
        else:
            st.warning(f"{watchlist_name} already has 10 stocks. Remove a stock before adding a new one.")
    else:
        st.warning(f"{ticker} is already in {watchlist_name}.")

# Function to remove a ticker from watchlist
def remove_from_watchlist(watchlist_name, ticker):
    if ticker in st.session_state['watchlists'][watchlist_name]:
        st.session_state['watchlists'][watchlist_name].remove(ticker)
        st.success(f"Ticker {ticker} removed from {watchlist_name}!")

# Watch List Page
if selected == "Watch List":
    st.sidebar.header("Watchlist Manager")
    selected_watchlist = st.sidebar.radio("Select Watchlist", list(st.session_state['watchlists'].keys()))

    # Sidebar - Add ticker to selected watchlist
    st.sidebar.subheader("Add Ticker")
    ticker_input = st.sidebar.text_input("Ticker Symbol (e.g., AAPL)")

    if st.sidebar.button("Add Ticker"):
        update_watchlist(selected_watchlist, ticker_input.upper())
        st.sidebar.success(f"Ticker {ticker_input.upper()} added to {selected_watchlist}!")

    # Main section - Display watchlist and stock data
    st.header(f"{selected_watchlist}")
    watchlist_tickers = st.session_state['watchlists'][selected_watchlist]

    if watchlist_tickers:
        # Fetch data for all tickers in the watchlist
        watchlist_data = {ticker: get_stock_data(ticker) for ticker in watchlist_tickers}
        
        # Convert the dictionary of series to a DataFrame
        watchlist_df = pd.DataFrame(watchlist_data).T  # Transpose to have tickers as rows
        st.write("Watchlist Data:")
        st.dataframe(watchlist_df)
        
        # Provide option to remove tickers
        st.subheader("Remove Ticker")
        ticker_to_remove = st.selectbox("Select Ticker to Remove", watchlist_tickers)
        if st.button("Remove Ticker"):
            remove_from_watchlist(selected_watchlist, ticker_to_remove)
            st.experimental_rerun()  # Refresh the app to reflect changes
    else:
        st.write("No tickers in this watchlist.")

    # Footer - Show all watchlists and their tickers
    st.sidebar.subheader("All Watchlists")
    for watchlist, tickers in st.session_state['watchlists'].items():
        st.sidebar.write(f"{watchlist}: {', '.join(tickers) if tickers else 'No tickers'}")


# Main application logic
if selected == "Stock Price Forecasting":

    # Step 2: Search box for stock ticker
    ticker = st.text_input('Enter Stock Ticker', 'AAPL')

    # Step 3: Slider for selecting the date range
    years = st.slider('Select number of years of historical data', 1, 5, 2)

    # Calculate the start and end date based on the slider
    end_date = datetime.today().date()
    start_date = end_date - timedelta(days=years*365)

    # Fetch historical data from Yahoo Finance
    @st.cache_data
    def load_data(ticker, start_date, end_date):
        data = yf.download(ticker, start=start_date, end=end_date)
        data.reset_index(inplace=True)
        return data

    data_load_state = st.text('Loading data...')
    data = load_data(ticker, start_date, end_date)
    data_load_state.text('Loading data...done!')

    # Display the raw data
    st.subheader('Raw data')
    st.write(data.tail())

    # Prepare the data for regression
    data['Date'] = pd.to_datetime(data['Date'])
    data['Date_ordinal'] = data['Date'].map(datetime.toordinal)

    # Features and target
    X = data[['Date_ordinal']]
    y = data['Close']



    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Initialize and train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict the next 14 days
    future_dates = [end_date + timedelta(days=i) for i in range(1, 15)]
    future_dates_ordinal = [date.toordinal() for date in future_dates]
    future_dates_df = pd.DataFrame(future_dates_ordinal, columns=['Date_ordinal'])

    # Make predictions
    predictions = model.predict(future_dates_df)

    # Combine historical and forecast data
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Close': predictions
    })

    combined_df = pd.concat([data[['Date', 'Close']], forecast_df])

    # Step 6: Display the forecast
    st.subheader('Forecast data')
    st.write(forecast_df)

    # Plot the forecast
    st.subheader('Forecast plot')

    fig = go.Figure()

    # Add historical data
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Historical Close Prices'))

    # Add forecasted data
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Close'], mode='lines', name='Forecasted Close Prices'))

    fig.update_layout(
        title='Stock Price Forecast',
        xaxis_title='Date',
        yaxis_title='Close Price',
        legend=dict(x=0, y=1)
    )

    st.plotly_chart(fig)