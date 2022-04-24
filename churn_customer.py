import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,StandardScaler
scale = StandardScaler()
from sklearn import svm
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import sklearn.neural_network
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

from collections import OrderedDict
from sklearn.datasets import make_classification

###########################################################################
#read data
df = pd.read_excel ("C:/Users/Nazari/Desktop/project/churn customer/00.02.20/code and ppt/churn_test.xlsx", engine='openpyxl')

df = df[['برگشتی ریالی خالص','باقی مانده نهایی فاکتور','میانگین زمان دریافت پول',
         'میانگین زمان دریافت سند','برگشتی تعدادی','تعداد کالا',
         'فاصله اولین خرید تا امروز','فروش ناخالص تعدادی','فروش ناخالص ریالی',
         'تعداد ویزیت','DurationFrequency','count month','count last month','Churn']]

df = df.rename(columns={'برگشتی ریالی خالص':'RRG',
                        'باقی مانده نهایی فاکتور':'FIR',
                        'میانگین زمان دریافت پول':'ATRC',
                        'میانگین زمان دریافت سند':'ATRD',
                        'برگشتی تعدادی':'RN',
                        'تعداد کالا': 'QTY',
                        'فاصله اولین خرید تا امروز': 'LB',
                        'فروش ناخالص تعدادی': 'GQTY',
                        'فروش ناخالص ریالی': 'Value',
                        'تعداد ویزیت': 'CV',
                        'DurationFrequency':'DF',
                        'count month':'CM',
                        'count last month':'CLM'})

###########################################################################
#column name
'''
for col in df.columns:
    print(col)

###########################################################################
#feature selection with correlation

label_encoder = LabelEncoder()
df.iloc[:,0] = label_encoder.fit_transform(df.iloc[:,0]).astype('float64')
corr = df.corr()

columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = df.columns[columns]
df = df[selected_columns]
'''

###########################################################################
#column name
'''
print('new columns:######################################')
for col in df.columns:
    print(col)
'''
###################################################################################################################
#Logistic Regression Model
###########################################################################
#find x & y
x = df[['RRG','FIR','ATRC','ATRD','RN','QTY',
        'LB','GQTY','Value','CV','CM','CLM','DF']]
x = scale.fit_transform(x)
y = df['Churn']
###########################################################################
#set train and test data
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.20)
###################################################################################################################
#Random Forest Method
#fit Random Forest model
randf = RandomForestClassifier(max_features = 4,n_estimators=500,
                               min_samples_split=100,random_state =50,
                               min_samples_leaf=100,n_jobs=1)
randf.fit(X_train,y_train)
y_pred = randf.predict(X_test)
print("Accuracy Random Forest:",metrics.accuracy_score(y_test, y_pred))

'''
cm = confusion_matrix(y_test, randf.predict(X_test))

score = randf.score(X_train, y_train)
print ("Training set score:",score)
print(classification_report(y_test,y_pred))

RF = SelectFromModel(RandomForestClassifier(n_estimators=500,
                      min_samples_split=100, min_samples_leaf=100,n_jobs=1))
RF.fit(X_train, y_train)
print("Feature selection:",RF.get_support())
print("Feature selection:",randf.feature_importances_)
'''
'''
y_probability = randf.predict_proba(X_test)
y_probability = pd.DataFrame(y_probability, columns=['0','1'])
X_test_inversed = scale.inverse_transform(X_test)
X_test_inversed = pd.DataFrame(X_test_inversed,
                  columns=['RRG','FIR','ATRC','ATRD','RN','QTY','LB',
                 'GQTY','Value','CV','CM','CLM'])
y_test_DataFrame = pd.DataFrame(y_test, columns=['Churn'])
result1 = pd.concat([X_test_inversed,y_test_DataFrame, y_probability], axis=1)
#result1.to_excel("output1.xlsx")
'''
'''
#confusion matrix
#cm = confusion_matrix(y_train, LR.predict(X_train))
print("Accuracy logestic regression:",metrics.accuracy_score(y_test, y_pred))
print ("Training set score:",score)
#print ('train data:',cm)
#print('test data:',confusion_matrix(y_test,y_pred))
#print(classification_report(y_test,y_pred))

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()
'''

###################################################################################################################
#Neural Network
#fit Neural Network model
NN = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, )
            , activation='relu', solver='adam',alpha=0.0001,
            batch_size='auto', learning_rate='invscaling',
            learning_rate_init=0.001, power_t=0.5, max_iter=1000,shuffle=True,
            random_state=None, tol=0.0001, verbose=False, warm_start=False,
            momentum=0.9, nesterovs_momentum=True, early_stopping=False,
            validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, 
                                                 n_iter_no_change=10)

NN.fit(X_train, y_train)
y_pred = NN.predict(X_test)
print("Accuracy Neural Network:",metrics.accuracy_score(y_test, y_pred))

score = NN.score(X_train, y_train)
print ("Training set score:",score)
###################################################################################################################
#Decision Tree
#fit Decision Tree model
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(max_depth = 10,random_state = 101,
                             max_features = None,min_samples_leaf = 15)
DT.fit(X_train,y_train)
y_pred = DT.predict(X_test)
print("Accuracy Decision Tree:",metrics.accuracy_score(y_test, y_pred))
score = DT.score(X_train, y_train)
print ("Training set score:",score)

###################################################################################################################
#preidct new data
#Enter your features
'''
new_candidates = {'RRG':[0],'FIR':[160430705],'ATRC':[1],'ATRD':[1],
	          'RN' : [0],'QTY':[17] , 'LB':[2],'GQTY':[522],
		  'Value':[226326102],'CV':[0],'CM':[0],'CLM':[26]}
df2 = pd.DataFrame(new_candidates,columns=
 ['RRG','FIR','ATRC','ATRD','RN','QTY','LB','GQTY','Value','CV','CM','CLM'])

scaled = scale.transform(df2)
y_pred = randf.predict([scaled[0]])
print ('predict churn in Random Forest:',y_pred)

#y_pred = LR.predict([scaled[0]])
#print ('predict churn in logestic regression:',y_pred)

y_pred = NN.predict([scaled[0]])
print ('predict churn in Neural Network:',y_pred)
y_pred = DT.predict([scaled[0]])
print ('predict churn in Support vector Machine:',y_pred)
'''
'''
NT = pd.read_excel ('D:/NewTest.xlsx', engine='openpyxl')
NC = NT[['برگشتی ریالی خالص','باقی مانده نهایی فاکتور',
         'میانگین زمان دریافت پول','میانگین زمان دریافت سند','برگشتی تعدادی',
         'تعداد کالا','فاصله اولین خرید تا امروز','فروش ناخالص تعدادی',
         'فروش ناخالص ریالی','تعداد ویزیت','count month','count last month']]

CI = NT[['کد مشتری']]

scaled1 = scale.transform(NC)
y_pred_RF = randf.predict(scaled1)
y_pred_RF = pd.DataFrame(y_pred_RF, columns=['Churn_RF'])

result = pd.concat([CI,NC,y_probability,y_pred_RF], axis=1)
result.to_excel("output.xlsx",index = False)

'''
