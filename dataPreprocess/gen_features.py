from sklearn.model_selection import train_test_split
from dataPreprocess.dataClean import load_dump
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from Model.model_dump import model_dump
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor,plot_importance,plot_metric,early_stopping
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


fea_pth = '../features_add.pic'
label_pth = '../label.pic'
growth_pth = '../price_growth_label.pic'


fea = load_dump(fea_pth)
label = load_dump(label_pth)
growth = load_dump(growth_pth)



all_pth = '../all.pic'

print(fea.shape)

all = pd.concat([fea,label],axis=1)

print(all.shape)

X_train,X_test,Y_train,Y_test = train_test_split(all,growth,test_size=0.1,random_state=45)

lgbm_param = {'boosting_type':'goss','objective':'regression','n_estimators':100,'learning_rate':0.02,'reg_lambda':0.9,'n_jobs':2}


lgb = LGBMRegressor(**lgbm_param)
lgb.fit(X_train,Y_train)





# print('best iter %d .'%(lgb.best_iteration_))
#
# y_pre = lgb.predict(X_test,num_iteration=lgb.best_iteration_)
#
# lgb_rmse = math.sqrt(mean_squared_error(Y_test,y_pre))
# print('rg rmse:  ',lgb_rmse)
# rg = RidgeCV(alphas=[0.9,0.8,0.7],fit_intercept=True,cv=5,normalize=True)
#
# print(' trainning ...')
# rg.fit(X_train,Y_train)
#
# print('trainning  over ...')
# rg_pre = rg.predict(X_test)
# rg_rmse = math.sqrt(mean_squared_error(Y_test,rg_pre))
# print('rg rmse:  ',rg_rmse)
#
# rg_model_pth = 'rg_model.pkl'
#
# model_dump(rg,rg_model_pth)