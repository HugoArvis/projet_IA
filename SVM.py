#Import du modèle SVM de sklearn et des bibliothèques nécessaires
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

#fonction pour le modèle SVM
def svm_model(X, y, test_size=0.2, random_state=42, kernel='linear', C=1.0):
    """
    Entraîne un modèle SVM sur les données fournies et évalue ses performances.

    Paramètres:
    X : array-like, caractéristiques d'entrée
    y : array-like, étiquettes de sortie
    test_size : float, proportion des données à utiliser pour le test
    random_state : int, graine pour la reproductibilité
    kernel : str, type de noyau à utiliser dans le SVM
    C : float, paramètre de régularisation

    Retourne:
    model : objet SVC entraîné
    report : rapport de classification
    cm : matrice de confusion
    """

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialiser le modèle SVM avec les paramètres spécifiés
    model = SVC(kernel=kernel, C=C)

    # Entraîner le modèle
    model.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Générer le rapport de classification et la matrice de confusion
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # sauvegarder le modèle entraîné
    pd.to_pickle(model, 'trained_models/svm_model.pkl')

    return model, report, cm