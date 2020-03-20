#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 14:20:07 2020

@author: Kirankumar
"""

import pandas_datareader.data as web
import datetime
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib import style

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns


from flask import Flask, render_template, request


stocks=[]
tech_stock=['AAPL','GOOG','MSFT','PYPL','AMD']
green_stock=['SEDG','JKS','SPWR','CWCO','SJW']
retail_stock=['WMT','AMZN','TGT','BABA','CVS']



bond='AGG'





start_date= datetime.datetime(2014,1,1)
end_date=datetime.datetime(2019,1,1)
stocks_percentage= 0.0
bond_percentage= 0.0
final_df_max_sharpie= []
final_df_min_volatility=[]
final_df=[]
num_simulation= 100
predicted_days =365

bond_data=web.DataReader(bond,data_source='yahoo',start=start_date,end=end_date)['Adj Close']
bond_data.sort_index(inplace=True)
bond_mu = expected_returns.mean_historical_return(bond_data)


def risk_calculator(age,plan,salary,saving,family):
    risk_score = (float(age)+float(plan)+float(salary)+float(saving)+float(family))
    
    return risk_score

def convert(ticker): 
    return (ticker.split()) 
  
def remove_values_from_list(the_list, val):
   return [value for value in the_list if value != val]

def stocks_add_remove(add_tickers,remove_tickers):
 dup_stocks=np.concatenate((stocks,add_tickers))
 added_stocks= list(set(dup_stocks))
 for i in remove_tickers:
    added_stocks= remove_values_from_list(added_stocks,i)
 return added_stocks



def max_sharpie_ratio(stocks_in_portfolio):
    
     stock_data = web.DataReader(stocks_in_portfolio,data_source='yahoo',start=start_date,end=end_date)['Adj Close']
     stock_data.sort_index(inplace=True)
     
     mu = expected_returns.mean_historical_return(stock_data)
     S = risk_models.sample_cov(stock_data)
     lower_bound=0.30/len(stocks_in_portfolio)

     # Optimise for maximal Sharpe ratio with no contraints
     ef = EfficientFrontier(mu,S,weight_bounds=(lower_bound,1))
     #Need to change the risk free rate 
     raw_weights = ef.max_sharpe(risk_free_rate=0.02)
     cleaned_weights = ef.clean_weights()
     cleaned_weights_df=pd.DataFrame.from_dict(cleaned_weights, orient='index')
     #remove weights with 0% 
     cleaned_weights_df=cleaned_weights_df.loc[(cleaned_weights_df!=0).any(1)]
     #print("Portfolio having maximal sharpie ratio and with no contraints\n" )
    # print(cleaned_weights)
     final_return= ef.portfolio_performance(verbose=True)
     index=['Expected Annual Return','Expected Annual Volatility','Sharpe Ratio']
     final_return_df = pd.DataFrame(final_return,index=index)
     final_df=pd.concat([cleaned_weights_df,final_return_df])
     return final_df
 
    
def min_volatility(stocks_in_portfolio):
     stock_data = web.DataReader(stocks_in_portfolio,data_source='yahoo',start=start_date,end=end_date)['Adj Close']
     stock_data.sort_index(inplace=True)
     
     mu = expected_returns.mean_historical_return(stock_data)
     S = risk_models.sample_cov(stock_data)
     lower_bound=0.30/len(stocks_in_portfolio)

     # Optimise for maximal Sharpe ratio with no contraints
     ef = EfficientFrontier(mu,S,weight_bounds=(lower_bound,1))
     #Need to change the risk free rate 
     raw_weights = ef.min_volatility()
     cleaned_weights = ef.clean_weights()
     cleaned_weights_df=pd.DataFrame.from_dict(cleaned_weights, orient='index')
     #remove weights with 0% 
     cleaned_weights_df=cleaned_weights_df.loc[(cleaned_weights_df!=0).any(1)]
     #print("Portfolio having maximal sharpie ratio and with no contraints\n" )
    # print(cleaned_weights)
     final_return= ef.portfolio_performance(verbose=True)
     index=['Expected Annual Return','Expected Annual Volatility','Sharpe Ratio']
     final_return_df = pd.DataFrame(final_return,index=index)
     final_df=pd.concat([cleaned_weights_df,final_return_df])
     return final_df

def calculate_return_volatility(tickers,weights):
    ticker_data = web.DataReader(tickers,data_source='yahoo',start=start_date,end=end_date)['Adj Close']
    ticker_data.sort_index(inplace=True)
    #convert daily stock prices into daily returns
    returns = ticker_data.pct_change()

    #calculate mean daily return and covariance of daily returns
    mean_daily_returns = returns.mean() 
    cov_matrix = returns.cov()
    portfolio_return = np.sum(mean_daily_returns * weights) * 252
    #check the folrmula aagain later
    portfolio_std_dev = np.sqrt(np.dot(weights,np.dot(cov_matrix * 252, weights))) 
    return portfolio_return,portfolio_std_dev


def backtesting(tickers,weights):
    start_date= datetime.datetime(2019,1,1)
    end_date=datetime.datetime(2020,1,1)
    ticker_data = web.DataReader(tickers,data_source='yahoo',start=start_date,end=end_date)['Adj Close']
    ticker_data.sort_index(inplace=True)
    #convert daily stock prices into daily returns
    returns = ticker_data.pct_change()
    
    #print (returns)
    port_val = returns * weights
    port_val['Portfolio Value'] = port_val.sum(axis=1)
   # print (port_val)
   
   
   
    benchmark_ticker=['^GSPC']
    benchmark_data=web.DataReader(benchmark_ticker,data_source='yahoo',start=start_date,end=end_date)['Adj Close']
    benchmark_data.sort_index(inplace=True)
    benchmark_returns=benchmark_data.pct_change()
    benchmark_cumulative_ret = (benchmark_returns + 1).cumprod()
    #print(benchmark_returns)
    prices = port_val['Portfolio Value']
    portfolio_cumulative_ret = (prices + 1).cumprod()
    benchmark_returns=pd.concat([benchmark_cumulative_ret,portfolio_cumulative_ret],axis=1)
    #print(benchmark_returns)
    #benchmark_returns.to_csv("df.csv")
    
    fig = plt.figure()
    style.use('bmh')
    
    title = "Backtesting "
    plt.plot(benchmark_returns['^GSPC'],label='S&P 500')
    plt.plot(benchmark_returns['Portfolio Value'],label='Portfolio')
    fig.suptitle(title,fontsize=18, fontweight='bold')
    plt.xlabel('Month')
    plt.ylabel('Returns')
    plt.legend()
    plt.grid(True,color='grey')
    
    date_string = time.strftime("%Y-%m-%d-%H:%M:%S")
        
    plt.savefig('static/benchmark'+date_string+'.png')
    png_name='static/benchmark'+date_string+'.png'
    return png_name
    
def add_row(df,column_1,column_2):
    to_append = [column_1,column_2]
    df_length = len(df)
    df.loc[df_length] = to_append
    return df

def sort_average(df):
        df=df.sort_values(by=99,axis=1,ascending=False)
        
       # del df['Unnamed: 0']
        #print (df)
        l = [i for i in range(num_simulation)]
        df.columns=l
        #print (df)
        x=0
        df_2=pd.DataFrame()
        average_by=int(num_simulation/10)
        
        while x<num_simulation:
            df_1=df.iloc[:,x:x+average_by]
            df_mean=df_1.mean(axis=1)
            df_2=pd.concat([df_2, df_mean], axis=1)
            #print(df_2)
            x=x+10
        return df_2

class monte_carlo:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        
        

    def get_portfolio(self, symbols, weights):
       
        
        #Get Price Data
        #df = web.get_data_yahoo(symbols,start_date, end_date,interval='m')['Adj Close']
        df=web.DataReader(symbols,'yahoo',start_date,end_date)['Adj Close']
        #Percent Change
        returns = df.pct_change()
        returns += 1
        
        #Define dollar amount in each asset
        port_val = returns * weights
        port_val['Portfolio Value'] = port_val.sum(axis=1)
        
        #Portfolio Dollar Values
        prices = port_val['Portfolio Value']
        
        #Portfolio Returns
        returns = port_val['Portfolio Value'].pct_change()
        returns = returns.replace([np.inf, -np.inf], np.nan)
                
        self.returns = returns
        self.prices = prices
    

    
    
    
    def brownian_motion(self, num_simulations, predicted_days):
        returns = self.returns
        prices = self.prices

        last_price = prices[-1]

        #Note we are assuming drift here
        simulation_df = pd.DataFrame()
        
        #Create Each Simulation as a Column in df
        for x in range(num_simulations):
            
            #Inputs
            count = 0
            avg_daily_ret = returns.mean()
            variance = returns.var()
            
            daily_vol = returns.std()
            daily_drift = avg_daily_ret - (variance/2)
            drift = daily_drift - 0.5 * daily_vol ** 2
            
            #Append Start Value    
            prices = []
            
            shock = drift + daily_vol * np.random.normal()
            last_price * math.exp(shock)
            prices.append(last_price)
            
            for i in range(predicted_days):
                if count == predicted_days:
                    break
                shock = drift + daily_vol * np.random.normal()
                price = prices[count] * math.exp(shock)
                prices.append(price)
                
        
                count += 1
            simulation_df[x] = prices
            
        self.simulation_df = simulation_df
        self.predicted_days = predicted_days
           

   
    
    def line_graph(self):
        prices = self.prices
        predicted_days = self.predicted_days
        simulation_df = self.simulation_df
        
        last_price = prices[-1]
        fig = plt.figure()
        style.use('bmh')
        
        #print(prices, predicted_days,simulation_df)
        simulation_df_sorted_averaged= sort_average(simulation_df)
        #print(simulation_df_sorted_averaged)
      
     
    
        title = "Monte Carlo Simulation: " + str(predicted_days) + " Days"
        plt.plot(simulation_df_sorted_averaged)
        fig.suptitle(title,fontsize=18, fontweight='bold')
        plt.xlabel('Day')
        plt.ylabel('Price ($USD)')
        plt.grid(True,color='grey')
        plt.axhline(y=last_price, color='r', linestyle='-')
        date_string = time.strftime("%Y-%m-%d-%H:%M:%S")
        
        plt.savefig('static/mcs'+date_string+'.png')
        png_name='static/mcs'+date_string+'.png'
        return png_name
        

        

    
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about/')
def about():
    return render_template('about.html')


@app.route('/form', methods=["POST"])
def form():
   # plans=request.form['plan']
    return render_template('form.html')


@app.route('/risk analysis', methods=["POST"])
def risk_analysis():
    global stocks_percentage
    global bond_percentage
    ages=request.form['age']
    plans=request.form['plan']
    salarys=request.form['salary']
    saving=request.form['savings']
    familys=request.form['family']
    risk_score = risk_calculator(ages,plans,salarys,saving,familys) 
    stocks_percentage= risk_score * 20
    bond_percentage= 100 - (risk_score * 20)
    
    return render_template('risk analysis.html',risk=risk_score,stocks=stocks_percentage,bonds=bond_percentage)

@app.route('/sector selection',methods=["POST"])
def sector_selection():
    return render_template('sector selection.html')

    
@app.route('/stock selection', methods=["POST"])
def stock_selection():
     
        global stocks
        stocks=[]
        sector=request.form.getlist('sector')
       # print(sector) 
        for i in sector :
            if i=='tech':
                stocks+=tech_stock
            if i=='green':
                stocks+=green_stock
            if i=='retail':
                stocks+=retail_stock
      
        #print (stocks)
            
        return render_template('stock selection.html')

@app.route('/stock weight', methods=["POST"])
def stock_weight():
    
     must_ticker= request.form['must_ticker']
     reject_ticker= request.form['reject_ticker']
     converted_must_tickers=[]
     converted_reject_tickers=[]
     #converting from string to list
     converted_must_tickers= convert(must_ticker)
     converted_reject_tickers= convert(reject_ticker)
     #print (converted_must_tickers,converted_reject_tickers)
     #addind and removing stocks from the main stocks
     stocks_in_portfolio = stocks_add_remove(converted_must_tickers,converted_reject_tickers)
     #print (stocks_in_portfolio)
     global final_df_max_sharpie 
     global final_df_min_volatility
     
     final_df_max_sharpie=max_sharpie_ratio(stocks_in_portfolio)
     final_df_min_volatility=min_volatility(stocks_in_portfolio)
    
     return render_template('stock weight.html',tables_1=[final_df_max_sharpie.to_html(classes='data')],tables_2=[final_df_min_volatility.to_html(classes='data')])
 
@app.route('/portfolio', methods=["POST"])
def portfolio():

    
    global final_df_max_sharpie
    global final_df_min_volatility
    global stocks_percentage
    global bond_percentage 
    global start_date ,end_date
    global final_df
    
    weights = request.form['weights']
    #if weights==0:
        #print (weights)
        #final_df=final_df_max_sharpie
       # print (final_df)
   # elif weights==1:
       # print (weights)
        #final_df=final_df_min_volatility
        #print (final_df)
    
    #print (final_df.tail())
    final_df=final_df_min_volatility
    if weights=='0':
        final_df=final_df_max_sharpie
    final_df = final_df * (float(stocks_percentage)/100)
    #print (final_df.tail())
    #removing sharpie ratio 
    final_df = final_df.drop("Sharpe Ratio", axis=0)
    final_df = final_df.drop("Expected Annual Return", axis=0)
    final_df = final_df.drop("Expected Annual Volatility", axis=0)
    #print(final_df.tail())
    final_df=final_df.reset_index()
    final_df=final_df.rename(columns={'index':'Ticker Symbol',0:'Weights'})
    #print(final_df.tail())
    
    final_df=add_row(final_df,bond,float(bond_percentage)/100)
    
    #print(final_df.tail())
    final_portfolio_tickers=final_df['Ticker Symbol'].tolist()
    final_portfolio_weights=final_df['Weights'].tolist()
    Portfolio_return,Portfolio_volatility=calculate_return_volatility(final_portfolio_tickers,final_portfolio_weights)
    
    final_df_1=add_row(final_df,"Annual Return",Portfolio_return)
    final_df_2=add_row(final_df_1,"Annual Volatility",Portfolio_volatility)
    #print (final_df.tail())
  
    
    sim = monte_carlo(start_date, end_date)
    sim.get_portfolio(final_portfolio_tickers,final_portfolio_weights)
    sim.brownian_motion(num_simulation,predicted_days)
    mc_png_name=sim.line_graph()
    
    
    backtesting_png_name= backtesting(final_portfolio_tickers,final_portfolio_weights) 
    
    return render_template('portfolio.html',tables=[final_df_2.to_html(classes='data')],mc_image=mc_png_name,backtesting_image=backtesting_png_name,stocks=stocks_percentage,bonds=bond_percentage)
    
if __name__ == '__main__':
    app.run(debug=True)
    
