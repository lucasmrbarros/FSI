import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('MI.csv', header=None)
dataset.replace('?', 10, inplace=True)

X_ad = dataset[list(dataset.columns[1:91]) + list(dataset.columns[95:99]) + list(dataset.columns[105:113])]
X_3d = dataset.iloc[:, 1:113]

def classify(CHOSEN_CLASSIFIER):
    for i in range (112,124):
        Y = dataset[i]

        X_ad_train, X_ad_test, Y_ad_train, Y_ad_test = train_test_split(X_ad, Y, test_size=0.3, random_state=0)
        X_3d_train, X_3d_test, Y_3d_train, Y_3d_test = train_test_split(X_3d, Y, test_size=0.3, random_state=0)

        clf_ad = CHOSEN_CLASSIFIER

        clf_ad .fit(X_ad_train, Y_ad_train)
        Y_ad_pred = clf_ad.predict(X_ad_test)

        clf_3d = CHOSEN_CLASSIFIER

        clf_ad.fit(X_3d_train, Y_3d_train)
        Y_3d_pred = clf_3d.predict(X_3d_test)

        print("Coluna {}".format(i))
        print("Admissao")
        print("======================================================")
        print(confusion_matrix(Y_ad_test, Y_ad_pred))
        print(accuracy_score(Y_ad_test, Y_ad_pred))
        print(classification_report(Y_ad_test, Y_ad_pred))
        print("______________________________________________________")

        print("Coluna {}".format(i))
        print("3° Dia")
        print("======================================================")
        print(confusion_matrix(Y_3d_test, Y_3d_pred))
        print(accuracy_score(Y_3d_test, Y_3d_pred))
        print(classification_report(Y_3d_test, Y_3d_pred))
        print("______________________________________________________")

print("#######################LDA###################################")
classify(LDA())

print("#######################QDA###################################")
classify(QDA())

print("#######################DTC###################################")
classify(DTC())

print("#######################KNN###################################")
classify(KNN(n_neighbors=10))


