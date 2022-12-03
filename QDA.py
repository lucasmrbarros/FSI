import numpy as np
import pandas as pd

#importando dados
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00579/MI.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
dataset = pd.read_csv(url, names=names)

#processamento de dados

#labels e features
X = dataset.iloc[:, 0:4].values #rever
y = dataset.iloc[:, 4].values #rever

#divisao entre dados de treinamento e testes
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#Escalonamento
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#executando LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

#X_train = QDA.fit(X_train, y_train) ## QuadraticDiscriminantAnalysis.fit() missing 1 required positional argument: 'y'
#X_test = QDA.transform(X_test)

clf = QDA()
clf.fit(X_train, y_train)
#Treinamento e predicoes

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(max_depth=2, random_state=0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#Perfomace
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
print('Accuracy' + str(accuracy_score(y_test, y_pred)))
