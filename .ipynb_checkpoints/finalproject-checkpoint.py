import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='minority')
data = pd.read_csv("fetal_health-1.csv")
df = data.head()
print(data['fetal_health'].value_counts())
tem = data['fetal_health'].value_counts()
isbalanced=True
if((( tem[2] + tem[3] ) / sum(tem)) < 0.3 or ((tem[2]+tem[3])/sum(tem))>0.7):
    isbalanced=False
else:
    isbalanced=True
if isbalanced==False:
    x,y=oversample.fit_resample(data.drop('fetal_health',axis=1),data['fetal_health'])
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=21)
else:
    x_train, x_test, y_train, y_test = train_test_split(data.drop('fetal_health',axis=1),data['fetal_health'],test_size=0.3,random_state=21)
cor=data.corr()
print((cor['fetal_health']**2).sort_values(ascending=False))
data=data.drop(['baseline value','histogram_tendency','severe_decelerations','mean_value_of_short_term_variability','fetal_movement','histogram_width','histogram_min','light_decelerations','histogram_max','histogram_number_of_peaks','histogram_number_of_zeroes'],axis=1)
print(data.head())
x_train=x_train.drop(['baseline value','histogram_tendency','severe_decelerations','mean_value_of_short_term_variability','fetal_movement','histogram_width','histogram_min','light_decelerations','histogram_max','histogram_number_of_peaks','histogram_number_of_zeroes'],axis=1)
x_test=x_test.drop(['baseline value','histogram_tendency','severe_decelerations','mean_value_of_short_term_variability','fetal_movement','histogram_width','histogram_min','light_decelerations','histogram_max','histogram_number_of_peaks','histogram_number_of_zeroes'],axis=1)
model1=DecisionTreeClassifier(criterion="gini",random_state=100)
model1.fit(x_train,y_train)
model2=LogisticRegression()
model2.fit(x_train,y_train)
pred_prob1 = model1.predict(x_test)
pred_prob2 = model2.predict(x_test)
# group those suspect with Pathological
temp_true1=np.where(y_test>1,0,1)
temp_pred1=np.where(pred_prob1>1,0,1)
c_1=confusion_matrix(temp_true1, temp_pred1)
print(c_1)
c_1.ravel()
fpr1, tpr1, thresh1 = roc_curve(temp_true1, temp_pred1, pos_label=1)
auc_score1 = roc_auc_score(temp_true1, temp_pred1)
print(auc_score1)
temp_true2=np.where(y_test>1,0,1)
temp_pred2=np.where(pred_prob2>1,0,1)
c_2=confusion_matrix(temp_true2, temp_pred2)
print(c_2)
c_2.ravel()
fpr2, tpr2, thresh2 = roc_curve(temp_true2, temp_pred2, pos_label=1)

