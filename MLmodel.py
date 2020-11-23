from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
wine= load_wine()
X=wine.data
y=wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.5)
clf=RandomForestClassifier(n_estimators=10)
clf.fit(X_train, y_train)
predicted=clf.predict(X_test)
print(accuracy_score(predicted,y_test))
