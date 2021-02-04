# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm, preprocessing
from sklearn import metrics
import sys
import time
import pickle
from sklearn.utils import shuffle

from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import BernoulliRBM


Train_FilePath= "./sample_dataset_for_train.csv"
Test_FilePath= "./sample_dataset_for_test.csv"
Modle_Save_FilePath='./model.pickle'
Test_Res_FilePath= "./predict_res.txt"

start = time.perf_counter()


XunLianDiZhi_List=[]
file = open(Train_FilePath, 'r', encoding='UTF-8')
for line in file:
    XunLianDiZhi_List.append(line.strip())
file.close()


CeShiDiZhi_List=[]
file = open(Test_FilePath, 'r', encoding='UTF-8')
for line in file:
    CeShiDiZhi_List.append(line.strip())
file.close()


data = pd.read_csv(open(XunLianDiZhi_List[0],'r', encoding='UTF-8'), low_memory=False)
data_test = pd.read_csv(open(CeShiDiZhi_List[0],'r', encoding='UTF-8'), low_memory=False)


for i in range(1,len(XunLianDiZhi_List)):
    data=data.append(pd.read_csv(open(XunLianDiZhi_List[i],'r', encoding='UTF-8'), low_memory=False),ignore_index=True)

for i in range(1,len(CeShiDiZhi_List)):
    data_test=data_test.append(pd.read_csv(open(CeShiDiZhi_List[i],'r', encoding='UTF-8'), low_memory=False),ignore_index=True)

data = shuffle(data)
pd.set_option('display.max_columns', None)


data.drop("ID",axis=1,inplace=True)
data_test.drop("ID",axis=1,inplace=True)


data['Class']=data['Class'].map({'DR':0,'FSC&BAR':1,'HBC':2,'MIN-FSC':3,'MIN-HBC':4,'NR':5,'PAII&PAR&BAR':6,'PAII&WI':7,'PAR&BAR':8,'PA':9})
data_test['Class']=data_test['Class'].map({'DR':0,'FSC&BAR':1,'HBC':2,'MIN-FSC':3,'MIN-HBC':4,'NR':5,'PAII&PAR&BAR':6,'PAII&WI':7,'PAR&BAR':8,'PA':9})


features_remain = data.columns[1:]
features_remain_test = data_test.columns[1:]
print(features_remain)
print('-'*100)

train_X = data[features_remain]
train_y=data['Class']
test_X= data_test[features_remain_test]
test_y =data_test['Class']

scaler=preprocessing.MinMaxScaler()
train_X=scaler.fit_transform(train_X)
test_X=scaler.transform(test_X)


FenLeiQi="ExtraTreeClassifier"

model=None
if FenLeiQi=="ExtraTreesClassifier":
    model=ExtraTreesClassifier(n_estimators=100, random_state=0)
elif FenLeiQi=="DecisionTreeClassifier":
    model=DecisionTreeClassifier(random_state=0)
elif FenLeiQi=="ExtraTreeClassifier":
    model=ExtraTreeClassifier(random_state=0)

model.fit(train_X,train_y)

prediction=model.predict(test_X)
file_Verify=open(Test_Res_FilePath, 'w')
for i in range(len(prediction)):
    file_Verify.write(str(prediction[i])+' \n')
file_Verify.close()


# with open(Modle_Save_FilePath, 'wb') as f:
#     pickle.dump(model, f)
end = time.perf_counter()
print('Time(seconds): ',end-start)
print('Classifier: ', FenLeiQi)
print('Accuracy: ', metrics.accuracy_score(prediction,test_y))

