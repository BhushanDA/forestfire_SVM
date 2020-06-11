#Importing Libraries
import pandas as pd 
import numpy as np 
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

#Reading Data
letters = pd.read_csv(r"D:\Python\New folder\forestfires.csv")
letters.head()
letters.describe()
letters.columns
colnames = list(letters.columns)

#Dropping Useless Columns
letters=letters.drop(letters.ix[:, 'dayfri':'monthsep'].columns, axis = 1) 
letters=letters.drop(letters.ix[:, 'month':'day'].columns, axis = 1)

#Preprocesing on Target column
number = preprocessing.LabelEncoder()
letters['size_category'] = number.fit_transform(letters['size_category'])

#Normalizing data
X_normalized = preprocessing.normalize(letters.iloc[:,0:9], norm='l2')
L=pd.DataFrame(X_normalized)

#Adding column to normalize data
f_column = letters["size_category"]
L = pd.concat([L,f_column], axis = 1)

#Train Test Split
train,test = train_test_split(L,test_size = 0.3)
test.head()

train_X = train.iloc[:,0:9]
train_y = train.iloc[:,9]
test_X  = test.iloc[:,0:9]
test_y  = test.iloc[:,9]
tarin_y=pd.DataFrame(train_y)


# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid'

# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear==test_y) # Accuracy =83.33

# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)

np.mean(pred_test_poly==test_y) # Accuracy = 78.84

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y) # Accuracy = 80.12

#kernel=sigmoid
model_rbf = SVC(kernel = "sigmoid")
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y)#Accuracy=78.84
