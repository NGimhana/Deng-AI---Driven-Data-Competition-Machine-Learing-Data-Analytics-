import xgboost as xgb
import pandas as pd
import numpy as np
import os

dirname = os.path.dirname(__file__)
dengi_test_x_data_file = os.path.join(dirname, 'x_y/four_month_before/original_data/test_x_iq_time_shifted.csv')
dengi_train_x_data_file = os.path.join(dirname, 'x_y/four_month_before/original_data/train_x_iq_time_shifted.csv')
dengi_test_y_data_file = os.path.join(dirname, 'x_y/four_month_before/original_data/test_y_iq.csv')
dengi_train_y_data_file = os.path.join(dirname, 'x_y/four_month_before/original_data/train_y_iq.csv')


dengi_test_x_data = pd.read_csv(dengi_test_x_data_file, header=0)
dengi_train_x_data = pd.read_csv(dengi_train_x_data_file,header=0)
dengi_test_y_data = pd.read_csv(dengi_test_y_data_file,header=0)
dengi_train_y_data = pd.read_csv(dengi_train_y_data_file,header=0)


dengi_test_x_dataframe = pd.DataFrame(dengi_test_x_data)
dengi_train_x_dataframe = pd.DataFrame(dengi_train_x_data)
dengi_test_y_dataframe = pd.DataFrame(dengi_test_y_data)
dengi_train_y_dataframe = pd.DataFrame(dengi_train_y_data)


X ,y =dengi_train_x_dataframe , dengi_train_y_data


data_dmatrix = xgb.DMatrix(data=X,label=y)

X_train, X_test, y_train, y_test = dengi_train_x_dataframe , dengi_test_x_dataframe ,\
                                   dengi_train_y_dataframe,dengi_test_y_dataframe

# max_depth 7->25.411 8->23.9447 done
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.01,
                max_depth = 5, alpha = 13, n_estimators = 10)

xg_reg.fit(X_train,y_train)

preds = xg_reg.predict(X_test)

actualValues = ((y_test.values))

# Model Accuracy Comparison (Mean Squared Error) =>2.2709
# print(np.sqrt((preds-actualValues)**2).mean())


submission = pd.DataFrame({'total_CAses': preds})

submission_file = os.path.join(dirname, "x_y/four_month_before/submissions/iq_model_test.csv")

submission.to_csv(submission_file, index=False)
