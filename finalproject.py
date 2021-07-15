#Analysis of the data is unbalanced since 77.8 of data has an output of 1.
#Code:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
oversample = RandomOverSampler(sampling_strategy='minority')
data = pd.read_csv("fetal_health-1.csv")
df=data.head()
print(data['fetal_health'].value_counts())
tem=data['fetal_health'].value_counts()
isbalanced=True
if(((tem[2]+tem[3])/sum(tem))<0.3 or ((tem[2]+tem[3])/sum(tem))>0.7):
    isbalanced=False
else:
    isbalanced=True
if(isbalanced==False):
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
pred_y1 = model1.predict(x_test)
pred_y2 = model2.predict(x_test)
pred_prob1=model1.predict_proba(x_test)
pred_prob2=model2.predict_proba(x_test)
# group those suspect with Pathological
c_1=confusion_matrix(y_test, pred_y1)
print(c_1)
print(classification_report(y_test, pred_y1))
c_2=confusion_matrix(y_test, pred_y2)
print(c_2)
print(classification_report(y_test, pred_y2))
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,0], pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:,0], pos_label=1)
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Decision Tree Classifier')
plt.plot(fpr2, tpr2, linestyle='--',color='green', label='Logistic Regression')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show()
auc_score1 = roc_auc_score(y_test, pred_prob1,multi_class="ovo")
auc_score2 = roc_auc_score(y_test, pred_prob2,multi_class="ovo")
print("Logistic Regression AUC: ", auc_score1)
print("Decision Tree Classifier AUC", auc_score2)
color=['maroon','black','green','cyan','magenta','yellow','purple','pink','grey','bisque','orange','brown','tan','darkblue','aquamarine']
kmeans=KMeans(n_clusters=5)
data['clusters']=kmeans.fit_predict(data.drop('fetal_health',axis=1),data['fetal_health'])
print("number of clusters equals 5: ")
print(data)
x_i=3
y_i=4
print(data.iloc[:,1].name)
for k in range(5):
    tem=data[data['clusters']==k]
    plt.scatter(tem.iloc[:,x_i], tem.iloc[:,y_i],c=color[k])
plt.scatter(kmeans.cluster_centers_[:, x_i], kmeans.cluster_centers_[:, y_i], s=300, c='red')
plt.xlabel(data.iloc[:,x_i].name)
plt.ylabel(data.iloc[:,y_i].name)
plt.show()
kmeans=KMeans(n_clusters=10)
data['clusters']=kmeans.fit_predict(data.drop('fetal_health',axis=1),data['fetal_health'])
print("number of clusters equals 10: ")
print(data)
for k in range(10):
    tem=data[data['clusters']==k]
    plt.scatter(tem.iloc[:,x_i], tem.iloc[:,y_i],c=color[k])
plt.scatter(kmeans.cluster_centers_[:, x_i], kmeans.cluster_centers_[:, y_i], s=300, c='red')
plt.xlabel(data.iloc[:,x_i].name)
plt.ylabel(data.iloc[:,y_i].name)
plt.show()
kmeans=KMeans(n_clusters=15)
data['clusters']=kmeans.fit_predict(data.drop('fetal_health',axis=1),data['fetal_health'])
print("number of clusters equals 15: ")
print(data)
for k in range(15):
    tem=data[data['clusters']==k]
    plt.scatter(tem.iloc[:,x_i], tem.iloc[:,y_i],c=color[k])
plt.scatter(kmeans.cluster_centers_[:, x_i], kmeans.cluster_centers_[:, y_i], s=300, c='red')
plt.xlabel(data.iloc[:,x_i].name)
plt.ylabel(data.iloc[:,y_i].name)
plt.show()
