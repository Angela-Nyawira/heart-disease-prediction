import pandas as pd

data = pd.read_csv("SAHeart_clean.csv")

x=data.drop('chd', axis=1)
y=data['chd']

##Splitting the data for training and testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(
    x, y, 
    test_size=0.5,
    random_state=42,
    stratify=y

)

##Scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)

x_test_scaled = scaler.transform(x_test)

##Choosing the best value of k
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
k_values = range(1, 21)
cv_means = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(
        knn,
        x_train_scaled,
        y_train,
        cv=2,
        scoring='accuracy'
    )

    cv_means.append(scores.mean())

best_k = k_values[cv_means.index(max(cv_means))]
print("Best value of k:",best_k)
best_cv = max(cv_means)
print("Best Cross Validation Accuracy:", best_cv)

knn = KNeighborsClassifier(n_neighbors=best_k)

# training

knn.fit(x_train_scaled, y_train)


y_predicted = knn.predict(x_test_scaled)

##Accuracy metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
)


print("Testing Accuracy:", accuracy_score(y_test, y_predicted))


print("Confusion Matrix")
print(confusion_matrix(y_test, y_predicted))


