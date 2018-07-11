
import pandas as pd
import numpy as np
import pickle,gzip


data_pth = '~/Desktop/xbz/download/tap4fun_data/tap_fun_train.csv'

def dump_data(obj,pth):
    print('dumping model ...')
    zipfw = gzip.open(pth,'wb')
    pickle.dump(obj,zipfw)
    zipfw.close()

def load_dump(pth):
    print('loading dump ... : '+pth)
    zipfr = gzip.open(pth,'rb')
    obj = pickle.load(zipfr)
    zipfr.close()
    return obj


def load_data(pth):
    print('loading data ...')
    df = pd.read_csv(pth)
    return df



# df = load_data(pth)
#
# print(df.shape)
#
#
# label = df[['prediction_pay_price']]
#
# print(label)
#
# df.drop(['prediction_pay_price','user_id','register_time'],axis=1,inplace=True)
# print(df.shape)
#
# features_dump_pth = "../features.pic"
# label_dump_pth = '../label.pic'
# dump_data(df,features_dump_pth)
# dump_data(label,label_dump_pth)
# #obj = load_dump(features_dump_pth)
#
# #print(obj.shape)
#
# label = load_dump(label_dump_pth)
#
# print(label.shape)
