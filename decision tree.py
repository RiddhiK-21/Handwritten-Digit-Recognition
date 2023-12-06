import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree,metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from joblib import dump, load


#...................Training Data..........................................
data_train=pd.read_csv('mnist_train.csv')
print('Training Data-Shape')
print(data_train.shape)
print()

x_tr=data_train.iloc[:,1:] #data
y_tr=data_train.iloc[:,0] #labels


#...................Splitting Training Data into Train and Test.............
X_train, X_test, y_train, y_test = train_test_split(x_tr,y_tr)



#....................Decision Tree Classifier................................
clf=DecisionTreeClassifier()



#.....................Training the Model*.................................
print('Model Training')
print()
clf.fit(X_train,y_train)


#....................Testing the Splitted Test Data.......................
print('predicting')
print()
pred=clf.predict(X_test)

#....................Accuracy of the Model on training dataset.............
print('Accuracy Score of Training Data')
print(accuracy_score(y_test, pred)*100)
print()

#..........confusion matrix of training dataset...........................
cm = metrics.confusion_matrix(y_test, pred)
print(f"Confusion matrix for Training Data:\n{cm}")
print()
disp=metrics.ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues',include_values=True)
plt.show()



#..........Saving the Trained Model......................................
dump(clf, 'decision_tree.joblib')
clf = load('decision_tree.joblib')


#................Test dataset.................................................
data_test=pd.read_csv('mnist_test.csv')
print('Testing Data-Shape')
print(data_test.shape)
print()

x_test=data_test.iloc[:,1:] #data
y_test=data_test.iloc[:,0] #labels


#...............Testing Model on Test Data....................................
pred=clf.predict(x_test)



#.............Accuracy of Model on Testing Dataset.............................
print('Accuracy Score of Testing Data')
print(accuracy_score(y_test, pred)*100)
print()



#.............confusion matrix for test dataset.................................
cm_test = metrics.confusion_matrix(y_test, pred)
print(f"Confusion matrix for Test Data:\n{cm_test}")
print()
disp=metrics.ConfusionMatrixDisplay(cm_test)
disp.plot(cmap='Blues',include_values=True)
plt.show()

