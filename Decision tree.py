import pandas as pd
data=pd.read_csv("SAHeart_clean.csv")

x=data.drop('chd', axis=1)
y= data['chd']


import sklearn.tree as dtree
import sklearn.metrics as ac
import sklearn.model_selection as kfx

##Splitting the data into training and testing
x_train, x_test, y_train,y_test =kfx.train_test_split(
    x,y,
    test_size=0.5,
    random_state=42,
    stratify=y
)

##Creating the decision tree
dt = dtree.DecisionTreeClassifier(

    criterion='gini',
    max_depth=5,
    random_state=42
)

## 2 fold cross validation
cv_scores = kfx.cross_val_score(

    dt,
    x_train,
    y_train,
    cv=2,
    scoring='accuracy'
 )

    

print("Cross Validation Scores:", cv_scores)
print("Average CV Accuracy:", cv_scores.mean())

##training

dt.fit(x_train, y_train)

##testing
y_predicted=dt.predict(x_test)

a1=ac.accuracy_score(y_test,y_predicted)
c1=ac.confusion_matrix(y_test,y_predicted)

print("Accuracy:",a1)
print("Confusion Matrix:",c1)

##Visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(18,10))

dtree.plot_tree(
    dt,
    feature_names=x.columns,
    class_names=['No CHD', 'CHD'],
    
)

plt.show()