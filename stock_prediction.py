import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib as mpl
from pandas.plotting import scatter_matrix
import math
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def read_finance_data(stock, start, end):
    '''
    Reading Yahoo Finance Data
    Args :
        stock : Stock for which prediction required
        start : Start date of hisorical data
        end : End date of historical data
    Returns :
        df : Data frame with historical data
    '''
    df = web.DataReader(stock, 'yahoo', start, end)
    print(df.tail())
    return df

def explore_rate(df):
    '''
    Exploring Rolling Mean and Return Rate of Stocks
    Args :
        df : Data frame with historical data
    Returns :
        mavg : Moving average
        close_px : Close_px
    '''
    close_px = df['Adj Close']
    mavg = close_px.rolling(window=100).mean()
    mavg.tail(10)
    return(mavg, close_px)

def plot_ma(mavg, close_px):
    '''
    Plot Moving Average with our Stocks Price Chart
    Args :
        mavg : Moving average
        close_px : Close_px
    Returns :
        None
    '''
    # Adjusting the size of matplotlib
    mpl.rc('figure', figsize=(8, 7))
    mpl.__version__

    # Adjusting the style of matplotlib
    style.use('ggplot')

    close_px.plot(label='AAPL')
    mavg.plot(label='mavg')
    plt.legend()

    # Return deviation
    rets = close_px / close_px.shift(1) - 1
    rets.plot(label='return')

def analyze_comp_stcoks(competitors, start, end):
    '''
    Analyzing Competitors Stocks
    Args :
        competitors : Competitors stocks array
        start : Start time of historical data
        end : End time of historical data
    Returns :
        None
    '''
    dfcomp = web.DataReader(competitors,'yahoo',start=start,end=end)['Adj Close']
    print(dfcomp.tail(10))

    # Correlation Analysis
    retscomp = dfcomp.pct_change()
    corr = retscomp.corr()
    print(corr)

    # Lets plot APPLE and GE with ScatterPlot to view their return distributions
    plt.scatter(retscomp.AAPL, retscomp.GE)
    plt.xlabel('Returns AAPL')
    plt.ylabel('Returns GE')

    # Plotting the scatter_matrix to visualize possible correlations among competing stocks**"""
    scatter_matrix(retscomp, diagonal='kde', figsize=(10, 10));

    #Heatmap
    plt.imshow(corr, cmap='hot', interpolation='none')
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns)
    plt.yticks(range(len(corr)), corr.columns);

    # Stocks Return Rate and Risk
    plt.scatter(retscomp.mean(), retscomp.std())
    plt.xlabel('Expected returns')
    plt.ylabel('Risk')
    for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):
        plt.annotate(
            label,
            xy = (x, y), xytext = (20, -20),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

def feature_eng(df):
    '''
    Feature Engineering
    Args :
        df : Data frame of historical data
    Returns :
        dfreg : Engineered data frame
    '''
    dfreg = df.loc[:,['Adj Close','Volume']]
    dfreg['HL_PCT'] = (df['High'] - df['Low']) / df['Close'] * 100.0
    dfreg['PCT_change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
    print(dfreg.tail(10))
    return dfreg

def pre_processing(dfreg):
    '''
    Pre-processing and cross validation
    Args :
        dfreg : Engineered data frame
    Returns : 
        X_train, X_test, y_train, y_test, X_lately : Training and testing data
    '''
    # Drop missing values
    dfreg.fillna(value=-99999, inplace=True)

    # We want to separate 1 percent of the data to forecast
    forecast_out = int(math.ceil(0.01 * len(dfreg)))

    # Separating the label here, we want to predict the AdjClose
    forecast_col = 'Adj Close'
    dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)
    X = np.array(dfreg.drop(['label'], 1))

    # Scale the X so that everyone can have the same distribution for linear regression
    X = preprocessing.scale(X)

    # Finally We want to find Data Series of late X and early X (train) for 
    # model generation and evaluation
    X_lately = X[-forecast_out:]
    X = X[:-forecast_out]

    # Separate label and identify it as y
    y = np.array(dfreg['label'])
    y = y[:-forecast_out]

    # Test train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return (X_train, X_test, y_train, y_test, X_lately)

def generate_model(X_train, y_train):
    '''
    Model generation
    Args :
        X_train, y_train : Training data
    Returns :
        clfreg, clfpoly2, clfpoly3, clfknn : Different regression classifiers
    '''
    # Linear Regression
    clfreg = LinearRegression(n_jobs=-1)
    clfreg.fit(X_train, y_train)

    # Quadratic Regression 2
    clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
    clfpoly2.fit(X_train, y_train)

    # Quadratic Regression 3
    clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
    clfpoly3.fit(X_train, y_train)

    # KNN Regression
    clfknn = KNeighborsRegressor(n_neighbors=2)
    clfknn.fit(X_train, y_train)

    return(clfreg, clfpoly2, clfpoly3, clfknn)

def model_evaluation(X_test, y_test, clfreg, clfpoly2, clfpoly3, clfknn):
    '''
    Model Evaluation
    Args : 
        X_test, y_test : Test data
        clfreg, clfpoly2, clfpoly3, clfknn : Different regression classifiers
    Returns :
        confidencereg, confidencepoly2, confidencepoly3, confidenceknn : Confidence scores
    '''
    confidencereg = clfreg.score(X_test, y_test)
    confidencepoly2 = clfpoly2.score(X_test,y_test)
    confidencepoly3 = clfpoly3.score(X_test,y_test)
    confidenceknn = clfknn.score(X_test, y_test)
    return(confidencereg, confidencepoly2, confidencepoly3, confidenceknn)

def sanity_testing(reg_method, dfreg, X_lately):
    '''
    Sanity Testing
    Args :
        reg_method : Regression method
        dfreg : Data frame for regression
        X_lately : Testing data
    Returns :
        None
    '''
    forecast_set = reg_method.predict(X_lately)
    dfreg['Forecast'] = np.nan

    # Plotting the Prediction
    last_date = dfreg.iloc[-1].name
    last_unix = last_date
    next_unix = last_unix + datetime.timedelta(days=1)

    for i in forecast_set:
        next_date = next_unix
        next_unix += datetime.timedelta(days=1)
        dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]
    dfreg['Adj Close'].tail(500).plot()
    dfreg['Forecast'].tail(500).plot()
    plt.legend(loc=4)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.show()

def stock_predict(my_stock):
    '''
    Predicting stock price
    Args : 
        my_stock : Stock for which you want to predict
    Returns :
        None
    '''
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime(2019, 8, 31)
    df = read_finance_data("AAPL", start, end)
    (mavg, close_px) = explore_rate(df)
    plot_ma(mavg, close_px)
    competitors = ['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT']
    analyze_comp_stcoks(competitors, start, end)
    dfreg = feature_eng(df)
    (X_train, X_test, y_train, y_test, X_lately) = pre_processing(dfreg)
    (clfreg, clfpoly2, clfpoly3, clfknn) = generate_model(X_train, y_train)
    (confidencereg, confidencepoly2, confidencepoly3, confidenceknn) = model_evaluation(X_test, y_test, clfreg, clfpoly2, clfpoly3, clfknn)
    print('Confidence Score using Linear Regression : ' , confidencereg)
    print('Confidence Score using Quadratic Regression 2 : ' , confidencepoly2)
    print('Confidence Score using Quadratic Regression 3 : ' , confidencepoly3)
    print('Confidence Score using KNN Regression : ' , confidenceknn)

    sanity_testing(clfreg, dfreg, X_lately)
    #sanity_testing(clfpoly2, dfreg, X_lately)
    #sanity_testing(clfpoly3, dfreg, X_lately)
    #sanity_testing(clfknn, dfreg, X_lately)

my_stock = "APPL"
stock_predict(my_stock)
