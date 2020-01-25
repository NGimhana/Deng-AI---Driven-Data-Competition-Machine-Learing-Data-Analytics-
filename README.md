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


3. **Filling missing values** - This was another issue with the data we had. There were many missing values in the dataset. For treating the missing values we used a sliding window of size 10 in corresponding column by making the missing value as centre and fill the missing value by the average of window.

4. **Generated new attributes by time shifting** - Another important observation we had was timely dependence of features. Here we wanted to check upto which extent current weeks prediction is affected by the features of previous weeks. In order to do that, by shifting data up to 4 months we created a new set of attributes. With that we were able to gear up the accuracy by a considerable amount.