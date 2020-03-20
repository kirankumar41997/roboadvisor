# roboadvisor
To run the roboadvisor, first download the whole folder 'Robo Advisor'. Then open Roboadvisor.py file and run it. 
Once the program is running, go to the your browser such as chrome and in the url, type "http://localhost:5000/". 
This will open the home page of the roboadvisor. 



Executive Summary: 

For the purpose of this team project in our Fintech class, we decided to create a Robo Advisor. 
A Robo Advisor is a tool used to provide automated financial guidance and services. The goal of our Robo Advisor is to determine a client´s risk aversion using a simple set of questions and assigning the client a risk score. Based on that risk score we offer the client a personalized, risk-adjusted portfolio which will be optimized by either maximizing the Sharpe Ratio or minimizing the Volatility. 

We build our Robo Advisor using mainly Python and HTML. The Robo Advisor utilizes a webpage for gathering the necessary inputs and displaying the according outputs. 

For testing purposes, to create our base-portfolio, we have pre-selected a set of stocks from the S&P 500 which can be changed and altered at any given time. The client´s personalized stock portfolio was then built by using the base-portfolio and adding or removing stocks at the client´s preference. This final selection of stocks was then optimized using historical data from yahoo finance and finding the optimal Sharpe Ratio on python.  

The final step on building the client´s personal and risk-adjusted portfolio was combining the optimized personal stock portfolio with our low risk investment alternative, the AGG Bond ETF. With the client’s risk score we determine the exact weightings of each asset class in the final portfolio. 

For this final portfolio we compute average return and risk. We display our results for the customer with data about each of the personalized portfolios position, forecasting of possible performance outcomes and a back testing to simulate how the portfolio would have performed compared to a benchmark in the past. 

This paper includes the following sections to explain the different steps of our project in more depth:
1)	Determination of the client’s risk profile
2)	Building the personalized stock portfolio
3)	Using historical data to optimize the personalized stock portfolio
4)	Combining the personalized stock portfolio with the low risk alternative investment (AGG)
5)	Displaying the outcome for the customer

Furthermore, displays of the Robo-Advisor, codes, and references are displayed in the appendix. 








Our Robo Advisor: 

1.	Determination of the client´s risk profile 

The first part of this project focuses on understanding the risk aversion profile of our client. In order to do so, we ask basic questions to our customers in order to analyze and calculate their risk aversion. The questions will be related to the client’s age, pre-tax salary, savings, family status and goal of their investment. We are aware that we could ask an infinite number of questions to get a deep understanding of our client’s risk profile. But for the sake of simplicity we have focused on a set of questions which in our opinion have a clear impact on a person’s risk adversity. 
The first step for our client is to answer the following questions that can be found on our web page. The client is asked to select one answer per question.
  

 




Using the grading system that assigns a specific score to each possible answer in each question we are now able to calculate a customer’s risk score. The result will lead us to build the portfolio in relation to what kind of personal risk profile they have. 
In order to assign grades to the answers of our client, we have made several assumptions:
•	The younger our client, the lower the risk aversion
•	The higher the client’s salary, the lower the risk aversion
•	The higher the personal savings, the lower the risk aversion
•	Also, if our client doesn’t have a family (married or children), the less risk averse he is. We assume that by answering yes, the client has another party to worry and provide for than him- or herself. 
o	(If a client was institutional, this question could be changed to ask if the client has any employees to provide for.)
•	Finally, the reason for investing is going from adding another income stream, to building a retirement plan which is the most risk averse to the least.

As a result, our customer will receive a final score between 1.35 (high risk aversion) and 5.00 (the least risk averse). In our system it is not possible to have a risk score of 0 because we assume that any individual looking to invest in order to make returns is already displaying a certain level of risk acceptance. 

After answering the different questions and being assigned a risk score, the customer will then receive his final result: “According to your personal risk score of … your portfolio should contain … % of stocks and …% of bonds.” Further explanations to the exact calculations will follow in step four. 


2.	Building the personalized stock portfolio

For building the personalized stock portfolio, we give the client the opportunity to choose positions that should be explicitly in- or excluded. 

Furthermore, we offer the client the possibility to choose between different sectors to invest in. In this case we have chosen Tech, Green and Retail as exemplifying sectors to choose from. For each sector we have a pre-picked set of stocks included into our base portfolio. The chosen stocks combined with our sector specific baseline portfolio will then represent the personalized stock portfolio. 

The selection of sectors and stocks is merely for testing and displaying purposes to determine the functionality of our Robo Advisor. 


3.	Using historical data to optimize the personalized stock portfolio

After building the personalized stock portfolio, the next step is to optimize the weightings of the assets included in the portfolio. 

We decided, that each included position of the personalized stock portfolio will have a minimum weight in order to be included. The minimum weight we decided upon is 0,3 divided by the number of stocks. So, if a portfolio would have 3 stocks, each of them would have a minimum weight of 10% in the portfolio. This is to make sure a customer’s preference for choosing a stock is respected and could be changed any time to achieve a more appropriate weighting. 


Now, with that rule as a constraint we are using historical data to optimize the portfolio using for two aspects: 
A)	We optimize the weights of the stocks to get the highest return for a given amount of risk.
B)	We optimize the portfolio to minimize the standard deviation.

The customer gets to see the different results as a comparison and to pick the preferred strategy. 


4.	Combining the personalized stock portfolio with the low risk alternative investment (AGG) to build the final portfolio

In order to find out to percentage of bonds and stocks of our customer’s portfolio, we have come up with a two steps formula (figure 3). We here assume that in the real world every asset is assigned an individual risk score and that helps to balance out every portfolio according to the risks of the individual included assets. For simplicity we use the 0 and 5 system. We make the assumption that bonds have an assigned risk score of 0 and stocks have the highest risk classification of 5. For a more sophisticated version, any asset class and any given security could have an individual assigned risk classification between 0 and 5.







As shown in figure 3, the percentage of stocks is calculated by multiplying the risk aversion score (calculated by the gradings from its answers) by 20/100. In this case, we are able to find the percentage of bonds by discounting this percentage of stocks from 1. Because the final portfolio will be equal to X% of bonds + X% of stocks = 100%.


5.	Displaying the outcome for the customer

This is the final part of our Robo-Advisor. After the final portfolio is created using the personalized stock portfolio, the AGG, and the risk formula from step 4, the final results of the portfolio are displayed. 

Here all the individual positions of the portfolio are shown in a table with the assigned weightings. The general formula for computing those weights is as follows: 



All positions now add up to 100% and that is the final, risk adjusted and personalized portfolio. 

Another part of the outcome is a one year forecast for our customer. We are running a Monte-Carlo simulation for the final portfolio using the calculated Standard Deviation and Average Return. We then average the top 10%, the following 10% and so forth to the bottom 10% of the simulated returns. Displaying the average returns for every 10% of the simulations and comparing it with the average return of all simulations helps us to visualize possible outcomes and their likelihood for our client. 

Finally, we used back testing to check how the final portfolio performed compared to a benchmark with real life returns of the past. For the benchmark we use the S&P500. We run the simulation for the last year and graph the performance of the benchmark against the portfolio. In our example it is clear, that the portfolio is less volatile, even though the benchmark generated higher returns. So, the initial goal of reducing risk by adding the AGG bonds to the portfolio was successful. 

Conclusion:

We have built a functioning Robo-Advisor that is able to take customer input and create a personalized output in form of a risk adjusted and optimized investment portfolio. 

We believe, that our Robo Advisor can be improved in many ways. The next steps would be to include:

-	Sortino Ratio to show downturn risks
-	Assign individual risk classifications to each asset
-	Include more question to acquire a more sufficient customer risk score
-	Include more sectors
-	Be more specific about a client´s investment strategy

All of these improvements could be made but would have been too demanding for the given timeframe and scope of our project. We enjoyed building this prototype and we have learned a lot along the way. 




