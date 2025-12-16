# Import du modèle de régression logistique de sklearn et des bibliothèques nécessaires
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.combine import SMOTEENN
import os
import joblib  # Meilleur que pickle pour les modèles sklearn


# Fonction pour le modèle de régression logistique
def logistic_regression_model(X, y, class_weight='balanced', test_size=0.2,
                              random_state=42, C=1.0, max_iter=1000,
                              use_saved_model=False):
    """
    Entraîne un modèle de régression logistique sur les données fournies et évalue ses performances.

    Paramètres:
    X : array-like, caractéristiques d'entrée
    y : array-like, étiquettes de sortie
    class_weight : str or dict, pondération des classes ('balanced' recommandé)
    test_size : float, proportion des données à utiliser pour le test
    random_state : int, graine pour la reproductibilité
    C : float, paramètre de régularisation inverse
    max_iter : int, nombre maximum d'itérations pour l'optimisation
    use_saved_model : bool, utiliser un modèle sauvegardé si disponible

    Retourne:
    model : objet LogisticRegression entraîné
    report : rapport de classification
    cm : matrice de confusion
    """

    # Créer le dossier si nécessaire
    os.makedirs('trained_models', exist_ok=True)

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Vérifier s'il existe un modèle de régression logistique déjà entraîné
    model_path = 'trained_models/logistic_regression_model.pkl'

    if use_saved_model and os.path.exists(model_path):
        # Charger le modèle existant
        print("Chargement du modèle existant...")
        model = joblib.load(model_path)
    else:
        # Ajuster le déséquilibre des classes avec SMOTE
        print("Application de SMOTE...")
        print(f"Avant SMOTE - Classe 0: {sum(y_train == 0)}, Classe 1: {sum(y_train == 1)}")

        smoteenn = SMOTEENN(random_state=random_state)
        X_train_resampled, y_train_resampled = smoteenn.fit_resample(X_train, y_train)

        print(f"Après SMOTEENN - Classe 0: {sum(y_train_resampled == 0)}, Classe 1: {sum(y_train_resampled == 1)}")

        # Créer un nouveau modèle et l'entraîner
        print("Entraînement du modèle...")
        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            class_weight=class_weight,
            solver='lbfgs'  # Solver recommandé
        )
        model.fit(X_train_resampled, y_train_resampled)

        # Sauvegarder le modèle entraîné
        joblib.dump(model, model_path)
        print(f"Modèle sauvegardé dans {model_path}")

    # Faire des prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Générer le rapport de classification et la matrice de confusion
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, report, cm