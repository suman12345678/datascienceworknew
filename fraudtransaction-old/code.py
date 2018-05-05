#import libraries
#=================
#dataset from kaggle PS_20174392719_1491204439457_log
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import Imputer 
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler 
from sklearn import metrics 
from sklearn.cross_validation import train_test_split 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix,roc_curve, auc, log_loss
from sklearn.utils import resample
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import KFold
#conda install -c anaconda py-xgboost=0.60
#
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC 
from sklearn.tree import DecisionTreeClassifier



#collecting data
#================
os.chdir("C:\\Users\\suman\\Desktop\\datasciencework\\fraudtransaction")
dataset = pd.read_csv('PS_20174392719_1491204439457_log.csv')
dataset.head(5)
dataset.columns




#eda of data
#============
#total row
len(dataset.index)#6362620 rows
dataset['isFraud'].value_counts()#6354407(not isFraud) and 8213(isFraud) so we need oversampling of isFraud. this command will exclude NaN
#total 


#as the data is imbalanced need oversampling but this is giving poor result
#dataset_majority=dataset[dataset.isFraud==0]
#dataset_minority=dataset[dataset.isFraud==1]
#df_minority_upsample=resample(dataset_minority,replace=True,n_samples=6354407,random_state=123)#random state for reproduiable
#df_upsampled=pd.concat([dataset_majority,df_minority_upsample])
#df_upsampled['isFraud'].value_counts()#now its 6354407 and 6354407
#dataset=df_upsampled

#check number of NaN value in each column
missing_value_count = dataset.isnull().sum()
missing_values_count[0:10]


#do bivariant analysis of type payment and fraud(catagory vs catagory)
dataset['type'].value_counts()
pd.crosstab(index=dataset["type"],columns=dataset["isFraud"])   # 2 way table   
#all fraue are for type=CASH_OUT.TRANSFER
pd.crosstab(index=dataset["step"],columns=dataset["isFraud"])   # 2 way table   
#in some steps fraud is more

#boxplot 'newbalanceDest', 'isFraud'
fig, ax = plt.subplots(figsize=(10,10))
l=['amount']
dataset.boxplot(l,'isFraud',ax)
plt.show()
#not much relation
#relation 'nameOrig', 'isFraud'
df2 = dataset.groupby(["isFraud", "nameOrig"]).size().reset_index(name='Count')
df2=df2[df2['Count'] >1]
#mostly those are not having any pattern
#relation 'nameDest', 'isFraud'
df2 = dataset.groupby(["nameDest", "isFraud"]).size().reset_index(name='Count')
df2=df2[df2['Count'] >70]
#there are some repetative same value of nameDest. some nameDest has a lot of obs but no fraud

fig, ax = plt.subplots(figsize=(10,10))
l=['amount']
dataset.boxplot(l,'isFraud',ax)
plt.show()

fig, ax = plt.subplots(figsize=(10,10))
l=['oldbalanceOrg']
dataset.boxplot(l,'isFraud',ax)
plt.show()

fig, ax = plt.subplots(figsize=(10,10))
l=['newbalanceOrig']
dataset.boxplot(l,'isFraud',ax)
plt.show()

fig, ax = plt.subplots(figsize=(10,10))
l=['oldbalanceDest']
dataset.boxplot(l,'isFraud',ax)
plt.show()


fig, ax = plt.subplots(figsize=(10,10))
l=['newbalanceDest']
dataset.boxplot(l,'isFraud',ax)
plt.show()


#data preprocessing
#===================================
#there is no NaN value so just change one and then impute one
dataset.ix[100,'isFlaggedFraud']=np.NaN
#check which row has NaN
dataset[dataset['isFlaggedFraud'].isnull()]
#impute mean value in the nan place

#take important columns split train test
dataset_pred=dataset[['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig','oldbalanceDest', 'newbalanceDest', 'isFraud','isFlaggedFraud']]
X = dataset_pred.loc[:,dataset_pred.columns != 'isFraud'].values 
y = dataset_pred.iloc[:, 7].values 

#impute mean value 
imputer = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0) 
imputer = imputer.fit(X[:,7:8]) 
X[:,7:8] = imputer.transform(X[:,7:8]) 

#see cor plot of numeric variables not working
#sns.set(style="ticks", color_codes=True)
#g = sns.pairplot(dataset_pred,hue="isFraud")


#use encoder
labelencoder_X_1 = LabelEncoder() 
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1]) 
#labelencoder_X_2 = LabelEncoder() 
#X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2]) 
#create dummy variable for type 5 type as there are catagory not working
onehotencoder = OneHotEncoder(categorical_features=[1]) #apply in column type as there are more than 2 catagory
X = onehotencoder.fit_transform(X).toarray() 




# splitting train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 
#check traing data spread fraud and nonfraud are equally sprea in train and test
np.count_nonzero(y_test==1)
np.count_nonzero(y_test==0)
np.count_nonzero(y_train==1)
np.count_nonzero(y_train==0)



#feature scalling


#sc_X = StandardScaler() 
#X_train = sc_X.fit_transform(X_train) 
#X_test = sc_X.transform(X_test) 
#sc_y = StandardScaler() 
#y_train = sc_y.fit_transform(y_train)




#model building with logistic
#========================================
# build model Logistic regression
classifier = LogisticRegression(class_weight={0:0.1,1:.9},random_state = 0) 
classifier.fit(X_train, y_train) 



#try grid search
grid = {
         'C': np.power(5.0, np.arange(-2, 2)),
          'penalty' : ['l2'] # no support of l1 penalty
         , 'solver': ['liblinear']#'newton-cg',
    }
classifier = LogisticRegression(class_weight={0:0.1,1:.9},random_state = 0)
gs = GridSearchCV(classifier, param_grid=grid, scoring='roc_auc', cv=5)
gs1=gs.fit(X_train, y_train)

print ('gs.best_score_:', gs1.best_score_)

# build model Logistic regression using abobe C parameter
classifier = LogisticRegression(class_weight={0:0.1,1:.9},C=1, random_state = 0) 
classifier.fit(X_train, y_train)

 
classifier.fit(X_train, y_train) 

#accuracy check
#=========================================
y_pred = classifier.predict(X_test) 
mat=pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
#positive is fraud, negative is not fraud
tp=mat.iloc[1,1]
tn=mat.iloc[0,0]
fp=mat.iloc[0,1]
fn=mat.iloc[1,0]
precision=tp/(tp+fp)
recall=tp/(tp+fn)
fscore=2*precision*recall/(precision+recall)
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show() 
logloss=log_loss(y_test, y_pred)

#area under curve = .78 for logistic regression class weight .1,.9, penalty L2
#fscore=.21 best  1 and worst 0  for logistic regression class weight .1,.9, penalty L2
#logloss .19

#without upsampled
#Predicted        0     1      
#True                             
#0          1269657(TN) 1226(FP)  
#1              953(FN) 688(TP)  1270345/1272524 = 99% but FP FN are huge we have to reduce type2 error 

#after balancing of data 
#Predicted        0        1     
#True                                
#0          1155862   113783
#1           128950  1143168  
#with classweight {0:0.1,1:.9}
#Predicted        0     1    
#True                             
#0          1264761  6122  
#1              713   928
#try grid search with different parameter





#try SVM but its very slow so left it without executing code

classifier = SVC(kernel = 'linear', random_state = 0) 
classifier.fit(X_train, y_train) 
# Predicting the Test set results 
y_pred = classifier.predict(X_test) 

mat=pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
#positive is fraud, negative is not fraud
tp=mat.iloc[1,1]
tn=mat.iloc[0,0]
fp=mat.iloc[0,1]
fn=mat.iloc[1,0]
precision=tp/(tp+fp)
recall=tp/(tp+fn)
fscore=2*precision*recall/(precision+recall)
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show() 
logloss=log_loss(y_test, y_pred)
#area under curve = .78 for logistic regression class weight .1,.9, penalty L2
#fscore=.21 best  1 and worst 0  for logistic regression class weight .1,.9, penalty L2
#logloss .19

#without upsampled
#Predicted        0     1      
#True                             
#0          1269657(TN) 1226(FP)  
#1              953(FN) 688(TP)  1270345/1272524 = 99% but FP FN are huge we have to reduce type2 






#try XGBoost package is not installed connection error



 
classifier = XGBClassifier() 
classifier.fit(X_train, y_train) 

# Predicting the Test set results 
y_pred = classifier.predict(X_test) 

mat=pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
#positive is fraud, negative is not fraud
tp=mat.iloc[1,1]
tn=mat.iloc[0,0]
fp=mat.iloc[0,1]
fn=mat.iloc[1,0]
precision=tp/(tp+fp)
recall=tp/(tp+fn)
fscore=2*precision*recall/(precision+recall)
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show() 
logloss=log_loss(y_test, y_pred)
#Predicted        0     1      All
#True                             
#0          1270872    11  1270883
#1              496  1145     1641
#All        1271368  1156  1272524

#fscore .818


#use adaboost
dt = DecisionTreeClassifier() 
clf = AdaBoostClassifier(n_estimators=100, base_estimator=dt,learning_rate=1)
#Above I have used decision tree as a base estimator, any ML learner as base estimator  
#100 decision tree are used as week learner
clf.fit(X_train,y_train)

#remove unwanted variables
del X,dataset,dataset_pred

# Predicting the Test set results 
y_pred = clf.predict(X_test) 

mat=pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
#positive is fraud, negative is not fraud
tp=mat.iloc[1,1]
tn=mat.iloc[0,0]
fp=mat.iloc[0,1]
fn=mat.iloc[1,0]
precision=tp/(tp+fp)
recall=tp/(tp+fn)
fscore=2*precision*recall/(precision+recall)
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show() 
logloss=log_loss(y_test, y_pred)
#Predicted        0     1      All
#True                             
#0          1270714   169  1270883
#1              197  1444     1641
#All        1270911  1613  1272524

#fscore .88

#so this is performing well

