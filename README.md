# Solution for DengAI Driven Data Competition (Machine Learing / Data Analytics)

## Introduction

The main goal was to predict the number of total Dengue cases reported in each week given set of features including daily climate data, satellite precipitation measurements, climate Forecast System Reanalysis measurements, normalized difference
vegetation index etc. 

## Data prepossessing

There were some issues with provided data samples and to overcome those, here are the techniques used.

1. **Selecting best attributes** – In the training dataset there were about 20 features and to get the best outcomes it is necessary to select the best features that describe the label. Reason for that is when considering correlations we found that some attributes have very low importance. Weka tool is used for selecting best correlated features.

Following are the best attributes selected.
* Reanalysis_specific_humidity_g_per_kg
* reanalysis_dew_point_temp_k
* station_avg_temp_c
* station_min_temp_c

2. **Data divided into two cities.** - This was a very important step as it has increased the accuracy by a considerable amount. After analysing the training data labels for two cities we found that there is a significant difference. So we split data into two cities and customized our models according to the data to get better results.


3. **Filling missing values** - There were many missing values in the dataset. For treating the missing values we used a sliding window of size 10 in corresponding column by making the missing value as centre and fill the missing value by the average of window.

4. **Generated new attributes by time shifting** - Another important observation had was timely dependence of features. Here we wanted to check upto which extent current weeks prediction is affected by the features of previous weeks. In order to do that, by shifting data up to 4 months we created a new set of attributes. With that we were able to gear up the accuracy by a considerable amount.

## Methodology

### XGBoost Algorithm

XGBoost is an optimized distributed gradient boosting library. It implements several machine learning algorithms under a Gradient boosting framework. We selected the boosted version of the Random Forest algorithm to solve DengAI problem. The process of XGBoost can be discussed under 3 steps.
 
1. **Gradient Boosting (GB)**

Gradient Boosting trains an ensemble of simple models while Stochastic Gradient Descent(SGD) trains a single complex model. GB doesn’t rely on a fixed architecture. In fact, the whole point of gradient boosting is to find the function which best approximates the data. The mathematical representation of it can be present like:

![Gradient-Boosting-Mathematical](https://raw.githubusercontent.com/ngimhana/DengAI_Driven_Data-Competition-Machine_Learing_Data_Analytics/master/diagrams/equation.png)


The only thing that has changed compared to SGD, in addition to finding the best parameters P, it also required to find the best function F. Initially it finds the best function F by taking lots of simple functions and adding them together.

2. **Gradient Boosted Trees**


In this step, it is considered in a special case, where the simple model is a tree. The challenging part of building a decision tree is to decide how to split a current leaf. That means deciding the next level of expansion. For instance, in the below image which is taken from XGBoost official documentation, how could someone add another layer to the (age > 15) leaf? The greedy way to do this is to consider every possible split on the remaining features (in this case both gender and occupation) and calculate the new loss for each split. Then picks the tree which most reduces loss. 

![Gradient-Boosting-Mathematical](https://raw.githubusercontent.com/ngimhana/DengAI_Driven_Data-Competition-Machine_Learing_Data_Analytics/master/diagrams/figure-2.png)



XGBoost, boosting this process and for that, it uses 2 features, number of leaf nodes and their weights.   

3. **Extreme Gradient Boosting**
 
In this step is boosting the tree selection process. XGBoost is one of the fastest implementations of gradient boosted trees. XGBoost uses several Hyperparameters to reduce overfitting the model. It tackles the major inefficiencies in constructing gradient boosted trees. 
XGBoost introduces several hyperparameters to tune which are designed to limit overfitting. Here are some of the Most useful hyperparameters.

* n_estimators = Number of trees to fit.
* max_depth = Maximum tree depth for base learners
* learning_rate = Boosting learning rate
* reg_alpha = L1 regularization term on weights

## Results

![Gradient-Boosting-Mathematical](https://raw.githubusercontent.com/ngimhana/DengAI_Driven_Data-Competition-Machine_Learing_Data_Analytics/master/diagrams/final_Results.png)