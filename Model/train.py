from sklearn.model_selection import train_test_split
from dataPreprocess.dataClean import load_dump
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Ridge
from Model.model_dump import model_dump
from sklearn.metrics import mean_squared_error
import math


fea_pth = 'features.pic'
label_pth = 'label.pic'

fea = load_dump(fea_pth)
label = load_dump(label_pth)


print('....split ....')
X_train,X_test,Y_train,Y_test = train_test_split(fea,label,test_size=0.1,random_state=45)


print(X_train.shape)
print(X_test.shape)



rf_param = {'n_estimators':100,"min_samples_split":4,"oob_score":True}
rf = RandomForestRegressor(**rf_param)

rf.fit(X_train,Y_train.values.ravel())

print(rf.oob_score_)

rf_pre = rf.predict(X_test)

rf_rmse = math.sqrt(mean_squared_error(Y_test,rf_pre))
rf_model_pth = 'rf_model.pkl'
print('rf rmse:  ',rf_rmse)
model_dump(rf,rf_model_pth)

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