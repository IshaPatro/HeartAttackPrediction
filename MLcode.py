#importing the library
import numpy as np
import pandas as pd

#reading the dataset and creating the dataframe
dataset = pd.read_csv("data.csv")

#converting all string values to nan
dataset = dataset.convert_objects(convert_numeric=True)

#dividing coloumns between dependent and independent variables
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 13].values

#fitting NaN value with the average values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy ="mean", axis = 0)
imputer = imputer.fit(x[:,0:13])   
x[:, 0:13] = imputer.transform(x[:, 0:13])
   
#scalng the data on the same scale
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x = sc_X.fit_transform(x)

#dividing data between test set and training set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x , y , test_size=0.1,random_state=8)

#Fitting logistics regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train,y_train)
# Predicting the Test set results
y_pred_lr = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm_lr = confusion_matrix(y_test,y_pred_lr)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(x_train, y_train)
# Predicting the Test set results
y_pred_knn = classifier.predict(x_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)

#Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf',random_state = 0)
classifier.fit(x_train, y_train)
# Predicting the Test set results
y_pred_svm = classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm_svm = confusion_matrix(y_test,y_pred_svm)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)
# Predicting the Test set results
y_pred_nb = classifier.predict(x_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_nb = confusion_matrix(y_test, y_pred_nb)

#Merging the predictions
y_predm=pd.DataFrame({'Logistic Regression':y_pred_lr,'SVM':y_pred_svm,'KNN':y_pred_knn,'Naive Bayes':y_pred_nb})

#Generating a dataframe of zeroes
zero_data=np.zeros(shape=(1,59))
y_pred=pd.DataFrame(zero_data)
y_pred=y_predm.mode(axis=1)
y_predict=y_pred.iloc[:,0].values

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predict)