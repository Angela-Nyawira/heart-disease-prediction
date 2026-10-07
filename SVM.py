import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("SAHeart_clean.csv")


x=df.drop(columns=['chd'])
y=df['chd']

x_train, x_test, y_train, y_test = train_test_split (x,y,test_size=0.5, random_state=42)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

svm_model=SVC(kernel='linear',random_state=42)
svm_model.fit(x_train_scaled,y_train)


svm_predictions=svm_model.predict(x_test_scaled)
svm_accuracy=accuracy_score(y_test, svm_predictions) * 100


print(f"overall accuracy: {svm_accuracy:.2f}%\n")
print(classification_report(y_test, svm_predictions))



