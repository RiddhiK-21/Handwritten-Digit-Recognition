import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
from joblib import dump, load


#...................Training Data..........................................
MNIST_train_df=pd.read_csv('mnist_train.csv')
print('Training Data-Shape')
print(MNIST_train_df.shape)
print()


X_tr = MNIST_train_df.iloc[:,1:]    #data in csv
y_tr = MNIST_train_df.iloc[:,0]     #labels



#...................Splitting Training Data into Train and Test.............
X_train, X_test, y_train, y_test = train_test_split(X_tr,y_tr,stratify=y_tr)




#....................SVM model with rbf kernel............................
model_rbf = svm.SVC(kernel='rbf',degree=3,gamma='scale')



#.....................Training the Model*.................................
print('Model Training')
print()
model_rbf.fit(X_train, y_train)



#....................Testing the Splitted Test Data.......................
print('predicting')
print()
y_pred=model_rbf.predict(X_test)



#....................Accuracy of the Model on training dataset.............
print('Accuracy Score of Training Data')
print(accuracy_score(y_test, y_pred)*100)
print()


#..........confusion matrix of training dataset...........................
disp = metrics.confusion_matrix(y_test, y_pred)
print(f"Confusion matrix for Training Data:\n{disp}")
print()
cm = ConfusionMatrixDisplay(confusion_matrix=disp,display_labels=set(y_tr))
cm.plot(cmap='Blues',include_values=True)
plt.show()



#..........Saving the Trained Model......................................
dump(model_rbf, 'svm.joblib')
clf = load('svm.joblib') 



#................Test dataset.................................................
MNIST_test_df=pd.read_csv('mnist_test.csv')
print('Testing Data-Shape')
print(MNIST_test_df.shape)
print()

X_test_tr = MNIST_test_df.iloc[:,1:]    #data in csv 
y_test_tr = MNIST_test_df.iloc[:,0]     #labels



#...............Testing Model on Test Data....................................
y_test_pred=model_rbf.predict(X_test_tr)



#.............Accuracy of Model on Testing Dataset.............................
print('Accuracy Score of Testing Data')
print(accuracy_score(y_test, y_pred)*100)
print()



#.............confusion matrix for test dataset.................................
disp_test = metrics.confusion_matrix(y_test_tr, y_test_pred)
print(f"Confusion matrix for Test Data:\n{disp_test}")
print()
cm = ConfusionMatrixDisplay(confusion_matrix=disp_test,display_labels=set(y_test_tr))
cm.plot(cmap='Blues',include_values=True)
plt.show()
