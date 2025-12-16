from sklearn.metrics import classification_report, confusion_matrix
from SVM import svm_model
from LogisticRregression import logistic_regression_model
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import os
#charger le jeu de données prepared_data.csv
data = pd.read_csv('data/prepared_data.csv')

#séparer les caractéristiques et la cible
X = data.drop('Attrition', axis=1)
y = data['Attrition']

#split les données d'entrainement et de test
X_test, X_train, y_test, y_train = train_test_split(X, y, test_size=0.2, random_state=42)


#modèle SVM

#vérifier si le mdèle SVM est déjà entraîné et sauvegardé
if os.path.exists('trained_models/svm_model.pkl'):
    #charger le modèle existant
    model = pd.read_pickle('trained_models/svm_model.pkl')
    print("Modèle SVM chargé depuis le fichier.")
    report = classification_report(y_test, model.predict(X_test))
    cm = None  # La matrice de confusion n'est pas calculée ici
else:
    #entraîner un nouveau modèle SVM
    model, report, cm = svm_model(X, y, test_size=0.2, random_state=42, kernel='linear')
    print("Nouveau modèle SVM entraîné.")

#afficher les résultats
print("\n", "="*50)
print("Résultats du modèle SVM sur les données de test :")
print("Rapport de classification :\n", report)
print("Matrice de confusion :\n", cm)

#Modèle de régression logistique
#vérifier si le modèle de régression logistique est déjà entraîné et sauvegardé
if os.path.exists('trained_models/logistic_regression_model.pkl'):
    #charger le modèle existant
    model = pd.read_pickle('trained_models/logistic_regression_model.pkl')
    print("Modèle de régression logistique chargé depuis le fichier.")
    report = classification_report(y_test, model.predict(X_test))
    cm = None  # La matrice de confusion n'est pas calculée ici
else:
    #entraîner un nouveau modèle de régression logistique
    model, report, cm = logistic_regression_model(X, y, test_size=0.2, random_state=42, C=1.0, max_iter=100)
    print("Nouveau modèle de régression logistique entraîné.")

#afficher les résultats
print("\n", "="*50)
print("Résultats du modèle de régression logistique sur les données de test :")
print("Rapport de classification :\n", report)
print("Matrice de confusion :\n", cm)

#afficher un graphique sur l'importance des caractéristiques pour le modèle de régression logistique
import matplotlib.pyplot as plt
import numpy as np
if 'LogisticRegression' in str(type(model)):
    importance = model.coef_[0]
    features = X.columns
    indices = np.argsort(importance)

    plt.figure(figsize=(10, 6))
    plt.title('Importance des caractéristiques pour la régression logistique')
    plt.barh(range(len(indices)), importance[indices], align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Importance')
    plt.show()