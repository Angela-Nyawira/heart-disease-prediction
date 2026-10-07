import pandas as pd
df=pd.read_csv("SAHeart_clean.csv")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score

x=df.drop(columns=['chd'])
y=df['chd']

x_train, x_test, y_train, y_test = train_test_split (x,y,test_size=0.5, random_state=42)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

ann_model=MLPClassifier(hidden_layer_sizes=(10,10), max_iter=1000, random_state=42)
ann_model.fit(x_train_scaled, y_train)

predictions=ann_model.predict(x_test_scaled)


print(f"overall accuracy: {accuracy_score(y_test, predictions)}")
print(classification_report(y_test, predictions))

import matplotlib.pyplot as plt

plt.figure (figsize=(10,6))
plt.plot(ann_model.loss_curve_,color='blue', linewidth=2,label=' Training Loss')
plt.title('Neural Network(Loss Curve)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=12)
plt.show()
