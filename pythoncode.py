import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from IPython.display import display 
sns.set()
#from mlxtend.plotting import plot_decision_regions 
#import missingno as msno
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
import warnings 
warnings.filterwarnings('ignore')
#matplotlib inline
diabetes_df = pd.read_csv('diabetes.csv') 
display(diabetes_df.head())#displays top 5 values 
display(diabetes_df.columns)#displays the columns available
display(diabetes_df.info())#information about the type of data 
display(diabetes_df.isnull().head(10))#checks for null values 
display(diabetes_df.isnull().sum())#checks how many null values are there
23
diabetes_df_copy = diabetes_df.copy(deep = True)
diabetes_df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = 
diabetes_df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
print(diabetes_df_copy.isnull().sum())
p = diabetes_df.hist(figsize = (20,20))#creates a histogram before removing null values 
plt.show()
#splitting the data
X = diabetes_df.drop('Outcome', axis=1) 
y = diabetes_df['Outcome']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33,
random_state=7)
#randomforest
from sklearn.ensemble import RandomForestClassifier 
rfc = RandomForestClassifier(n_estimators=200) 
rfc.fit(X_train, y_train)
rfc_train = rfc.predict(X_train) 
from sklearn import metrics
print("Accuracy_Score =", format(metrics.accuracy_score(y_train, rfc_train))) 
from sklearn import metrics
predictions = rfc.predict(X_test) 
print("Random forest values")
print("Accuracy_Score =", format(metrics.accuracy_score(y_test, predictions))) 
from sklearn.metrics import classification_report, confusion_matrix 
print(confusion_matrix(y_test, predictions))
24
print(classification_report(y_test,predictions))
#decision tree
from sklearn.tree import DecisionTreeClassifier 
dtree = DecisionTreeClassifier() 
dtree.fit(X_train, y_train)
from sklearn import metrics 
predictions = dtree.predict(X_test) 
print("Decision tree values")
print("Accuracy Score =", format(metrics.accuracy_score(y_test,predictions))) 
from sklearn.metrics import classification_report, confusion_matrix 
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test,predictions))
#xgboost classifier
from xgboost import XGBClassifier
xgb_model = XGBClassifier(gamma=0) 
xgb_model.fit(X_train, y_train)
from sklearn import metrics
xgb_pred = xgb_model.predict(X_test) 
print("xgboost values")
print("Accuracy Score =", format(metrics.accuracy_score(y_test, xgb_pred)))
#supportvectormachine
from sklearn.svm import SVC
svc_model = SVC() 
svc_model.fit(X_train, y_train) 
svc_pred = svc_model.predict(X_test)
25
from sklearn import metrics
print("SVM Values")
print("Accuracy Score =", format(metrics.accuracy_score(y_test, svc_pred))) 
print("Accuracy Score =", format(metrics.accuracy_score(y_test, svc_pred))) 
from sklearn.metrics import classification_report, confusion_matrix 
print(confusion_matrix(y_test, svc_pred)) 
print(classification_report(y_test,svc_pred))
#getting feature importance
rfc.feature_importances_
(pd.Series(rfc.feature_importances_, index=X.columns).plot(kind='barh')) 
plt.show()
import pickle
# Firstly we will be using the dump() function to save the model using pickle 
saved_model = pickle.dumps(rfc)
# Then we will be loading that saved model
rfc_from_pickle = pickle.loads(saved_model)
# lastly, after loading that model we will use this to make predictions
rfc_from_pickle.predict(X_test) 
display(diabetes_df.head())
display(diabetes_df.tail())
display(rfc.predict([[0,137,40,35,168,43.1,2.228,33]])) #5th patient
display(rfc.predict([[10,101,76,48,180,32.9,0.171,63]])) #763 th patien
