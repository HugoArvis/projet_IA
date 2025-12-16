#Import du modèle SVM de sklearn et des bibliothèques nécessaires
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.combine import SMOTEENN
import pandas as pd
import os
import joblib


#fonction pour le modèle SVM
def svm_model(X, y, test_size=0.2, random_state=42, kernel='linear', C=1.0, use_saved_model=False):
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

    # Créer le dossier si nécessaire
    os.makedirs('trained_models', exist_ok=True)

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model_path = os.path.join('trained_models', 'svm_model.pkl')

    #vérifier si le modèle SVM est déjà entraîné
    if use_saved_model and os.path.exists(model_path):
        #charger le modèle existant
        print("Chargement du modèle SVM existant...")
        model = joblib.load(model_path)
    else:
        #ajuster le déséquilibre des classes avec SMOTE
        print("Application de SMOTE...")
        print(f"Avant SMOTE - Classe 0: {sum(y_train == 0)}, Classe 1: {sum(y_train == 1)}")

        smoteenn = SMOTEENN(random_state=random_state)
        X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train, y_train)

        print(f"Après SMOTEENN - Classe 0: {sum(y_train_resampled == 0)}, Classe 1: {sum(y_train_resampled == 1)}")

        #créer un nouveau modèle et l'entraîner
        print("Entraînement du modèle SVM...")
        model = SVC(kernel=kernel, C=C, random_state=random_state)
        model.fit(X_train_resampled, y_train_resampled)

        #saouvegarder le modèle entraîné
        joblib.dump(model, model_path)
        print(f"Modèle SVM sauvegardé dans {model_path}")

    # Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Générer le rapport de classification et la matrice de confusion
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, report, cm