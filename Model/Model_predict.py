
import pandas as pd
from Model.model_dump import model_load


def map_x(x):
    if x< 0.0:
        return 0.0
    else:
        return x

eval_pth = '~/Desktop/xbz/download/tap4fun_data/tap_fun_test.csv'
rg_model_pth = 'rg_model.pkl'

df_test = pd.read_csv(eval_pth)

#print(df_test.head())

print('loading data completed ...')

eval_id = df_test['user_id']

df_test.drop(['user_id','register_time'],axis=1,inplace=True)

rg = model_load(rg_model_pth)


pre = rg.predict(df_test)
print('predict complete...')


results = eval_id

print(type(results))

results = pd.DataFrame(results)

#results['prediction_pay_price'] = pre

print(results.shape)
print('len',len(pre))


results.insert(1,'prediction_pay_price',pre)



print(results.shape)
print(results.head())


print('mean:',results['prediction_pay_price'].mean())

results['prediction_pay_price'] = results['prediction_pay_price'].apply(map_x)
results['prediction_pay_price'] = results['prediction_pay_price'].apply(lambda x:x*1.5)

print(results.head())

print('mean:',results['prediction_pay_price'].mean())
results_pth =  '~/Desktop/xbz/download/tap4fun_data/tap_fun_test_pre_2.csv'

results.to_csv(results_pth,index=False)