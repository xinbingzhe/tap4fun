from sklearn.model_selection import train_test_split
from dataPreprocess.dataClean import load_dump
from dataPreprocess.dataClean import dump_data
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

