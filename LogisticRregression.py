#Import du modèle de régression logistique de sklearn et des bibliothèques nécessaires
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

#fonction pour le modèle de régression logistique
def logistic_regression_model(X, y, test_size=0.2, random_state=42, C=1.0, max_iter=100, class_weight = 'balanced'):
    """
    Entraîne un modèle de régression logistique sur les données fournies et évalue ses performances.

    Paramètres:
    X : array-like, caractéristiques d'entrée
    y : array-like, étiquettes de sortie
    test_size : float, proportion des données à utiliser pour le test
    random_state : int, graine pour la reproductibilité
    C : float, paramètre de régularisation inverse
    max_iter : int, nombre maximum d'itérations pour l'optimisation

    Retourne:
    model : objet LogisticRegression entraîné
    report : rapport de classification
    cm : matrice de confusion
    """

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Initialiser le modèle de régression logistique avec les paramètres spécifiés
    model = LogisticRegression(C=C, max_iter=max_iter, class_weight=class_weight)

    # Entraîner le modèle
    model.fit(X_train, y_train)

    # Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Générer le rapport de classification et la matrice de confusion
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # sauvegarder le modèle entraîné
    pd.to_pickle(model, 'trained_models/logistic_regression_model.pkl')

    return model, report, cm