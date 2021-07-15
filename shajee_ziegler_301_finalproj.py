import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from itertools import cycle
from sklearn.metrics import roc_curve,precision_recall_curve,average_precision_score
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy import stats
from sklearn.cluster import KMeans
import seaborn as sns
oversample = RandomOverSampler(sampling_strategy='not majority')
data = pd.read_csv("fetal_health-1.csv")
df=data.head()
print(data['fetal_health'].value_counts())
tem=data['fetal_health'].value_counts()
isbalanced=True
if(((tem[2]+tem[3])/sum(tem))<0.3 or ((tem[2]+tem[3])/sum(tem))>0.7):
    isbalanced=False
else:
    isbalanced=True
sns.countplot(x="fetal_health",data=data)
plt.show()
if(isbalanced==False):
    x,y=oversample.fit_resample(data.drop('fetal_health',axis=1),data['fetal_health'])
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=21)
    new_data=pd.DataFrame(x)
    new_data['fetal_health']=y
    print(new_data['fetal_health'].value_counts())
    sns.countplot(x="fetal_health",data=new_data)
    plt.show()
else:
    x_train, x_test, y_train, y_test = train_test_split(data.drop('fetal_health',axis=1),data['fetal_health'],test_size=0.3,random_state=21)
    new_data=data
cor=new_data.corr()
print((cor['fetal_health']**2).sort_values(ascending=False))
for i in ('prolongued_decelerations','abnormal_short_term_variability','percentage_of_time_with_abnormal_long_term_variability','accelerations','histogram_mode','histogram_mean','mean_value_of_long_term_variability','histogram_variance','histogram_median','uterine_contractions'):
    pearson_coef,p_value = stats.pearsonr(new_data[i], new_data['fetal_health'])
    print("For ",i,"correlation coefficient is ",pearson_coef," and p-value is ",p_value)
new_data=new_data.drop(['baseline value','histogram_tendency','severe_decelerations','mean_value_of_short_term_variability','fetal_movement','histogram_width','histogram_min','light_decelerations','histogram_max','histogram_number_of_peaks','histogram_number_of_zeroes'],axis=1)
print(new_data.head())
x_train=x_train.drop(['light_decelerations','fetal_movement','uterine_contractions','severe_decelerations','histogram_min','histogram_max','mean_value_of_short_term_variability','histogram_width','histogram_number_of_peaks','histogram_number_of_zeroes','baseline value'],axis=1)
x_test=x_test.drop(['light_decelerations','fetal_movement','uterine_contractions','severe_decelerations','histogram_min','histogram_max','mean_value_of_short_term_variability','histogram_width','histogram_number_of_peaks','histogram_number_of_zeroes','baseline value'],axis=1)
model1=DecisionTreeClassifier(criterion="gini",random_state=21)
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

#%%
#precision recall curves
# binarize the testing data from each mode for use in multi-class precision-recall curves.
# Prediction probabilities are already in the binarized form so we can use pred_prob1 and
# pred_prob2 without modification

Y_test = label_binarize(y_test, classes=[1,2,3])
n_classes = Y_test.shape[1]

# uncomment to see what the binarized data looks like for further clarification
#print(Y_test)

# decision tree model
precision_y1 = dict()
recall_y1 = dict()
average_precision_y1 = dict()
for i in range(n_classes):
   precision_y1[i], recall_y1[i], _ = precision_recall_curve(Y_test[:, i],
                                                       pred_prob1[:,i])
   average_precision_y1[i] = average_precision_score(Y_test[:, i], pred_prob1[:,i])
precision_y1["micro"], recall_y1["micro"], _ = precision_recall_curve(Y_test.ravel(),
   pred_prob1.ravel())
average_precision_y1["micro"] = average_precision_score(Y_test, pred_prob1,
                                                    average="micro")
print('Average precision score for Decision Tree model, micro-averaged over all classes: {0:0.2f}'
     .format(average_precision_y1["micro"]))

# Logistic Regression Model
precision_y2 = dict()
recall_y2 = dict()
average_precision_y2 = dict()
for i in range(n_classes):
   precision_y2[i], recall_y2[i], _ = precision_recall_curve(Y_test[:, i],
                                                       pred_prob2[:,i])
   average_precision_y2[i] = average_precision_score(Y_test[:, i], pred_prob2[:,i])
precision_y2["micro"], recall_y2["micro"], _ = precision_recall_curve(Y_test.ravel(),
   pred_prob2.ravel())
average_precision_y2["micro"] = average_precision_score(Y_test, pred_prob2,
                                                    average="micro")
print('Average precision score for Logistic Regression model, micro-averaged over all classes: {0:0.2f}'
     .format(average_precision_y2["micro"]))
# Model 1 & 2 avg precision score micro-averaged
plt.figure(2)
plt.step(recall_y1['micro'], precision_y1['micro'], where='post', color='orange',label='Decision Tree Classifier')
plt.step(recall_y2['micro'], precision_y2['micro'], where='post', color='green',label='Logistic Regression')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0,1.05])
plt.xlim([0.0,1.0])
plt.legend(loc='best')
plt.title("Average precision score for both models, micro-averaged over all classes")
plt.savefig("avg_precision",dpi=300)
plt.show()

#%%
# Plot Precision-Recall curve for the Decision Tree model for each class and iso-f1 curves
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
plt.figure(3,figsize=(7,8))
f_scores = np.linspace(0.2,0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
   x = np.linspace(0.01, 1)
   y = f_score * x / (2 *  x - f_score)
   l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
   plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall_y1["micro"], precision_y1["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'''.format(average_precision_y1["micro"]))

for i, color in zip(range(n_classes), colors):
   l, = plt.plot(recall_y1[i], precision_y1[i], color=color, lw=2)
   lines.append(l)
   labels.append('Precision-recall for class {0} (area = {1:0.2f})'''.format(i+1, average_precision_y1[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0,1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

plt.show()

#%%
# Plot Precision-Recall curve for the Decision Tree model for each class and iso-f1 curves
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
plt.figure(3,figsize=(7,8))
f_scores = np.linspace(0.2,0.8, num=4)
lines = []
labels = []
for f_score in f_scores:
   x = np.linspace(0.01, 1)
   y = f_score * x / (2 *  x - f_score)
   l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
   plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

lines.append(l)
labels.append('iso-f1 curves')
l, = plt.plot(recall_y1["micro"], precision_y1["micro"], color='gold', lw=2)
lines.append(l)
labels.append('micro-average Precision-recall (area = {0:0.2f})'''.format(average_precision_y1["micro"]))

for i, color in zip(range(n_classes), colors):
   l, = plt.plot(recall_y1[i], precision_y1[i], color=color, lw=2)
   lines.append(l)
   labels.append('Precision-recall for class {0} (area = {1:0.2f})'''.format(i+1, average_precision_y1[i]))

fig = plt.gcf()
fig.subplots_adjust(bottom=0.25)
plt.xlim([0.0,1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Extension of Precision-Recall curve to multi-class')
plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

plt.show()

#%%
auc_score1 = roc_auc_score(y_test, pred_prob1,multi_class="ovo")
auc_score2 = roc_auc_score(y_test, pred_prob2,multi_class="ovo")
print("Decision Tree Classifier AUC: ", auc_score1)
print("LogisticRegression AUC", auc_score2)
color=['maroon','black','green','cyan','magenta','yellow','purple','pink','grey','bisque','orange','brown','tan','darkblue','aquamarine']
kmeans=KMeans(n_clusters=5,random_state=0)
new_data['clusters']=kmeans.fit_predict(new_data.drop('fetal_health',axis=1),new_data['fetal_health'])
print("number of clusters equals 5: ")
print(new_data)
x_i=3
y_i=4
print(new_data.iloc[:,1].name)
for k in range(5):
    tem=new_data[new_data['clusters']==k]
    plt.scatter(tem.iloc[:,x_i], tem.iloc[:,y_i],c=color[k])
plt.scatter(kmeans.cluster_centers_[:, x_i], kmeans.cluster_centers_[:, y_i], s=300, c='red')
plt.xlabel(new_data.iloc[:,x_i].name)
plt.ylabel(new_data.iloc[:,y_i].name)
plt.show()
kmeans=KMeans(n_clusters=10,random_state=0)
new_data['clusters']=kmeans.fit_predict(new_data.drop('fetal_health',axis=1),new_data['fetal_health'])
print("number of clusters equals 10: ")
print(new_data)
for k in range(10):
    tem=new_data[new_data['clusters']==k]
    plt.scatter(tem.iloc[:,x_i], tem.iloc[:,y_i],c=color[k])
plt.scatter(kmeans.cluster_centers_[:, x_i], kmeans.cluster_centers_[:, y_i], s=300, c='red')
plt.xlabel(new_data.iloc[:,x_i].name)
plt.ylabel(new_data.iloc[:,y_i].name)
plt.show()
kmeans=KMeans(n_clusters=15,random_state=0)
new_data['clusters']=kmeans.fit_predict(new_data.drop('fetal_health',axis=1),new_data['fetal_health'])
print("number of clusters equals 15: ")
print(new_data)
for k in range(15):
    tem=new_data[new_data['clusters']==k]
    plt.scatter(tem.iloc[:,x_i], tem.iloc[:,y_i],c=color[k])
plt.scatter(kmeans.cluster_centers_[:, x_i], kmeans.cluster_centers_[:, y_i], s=300, c='red')
plt.xlabel(new_data.iloc[:,x_i].name)
plt.ylabel(new_data.iloc[:,y_i].name)
plt.show()
