import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings

warnings.filterwarnings('ignore')

# Imports des modèles optimisés
from optimized_models import (
    logistic_regression_optimized,
    svm_optimized,
    compare_models,
    plot_feature_importance
)


def load_and_prepare_data(filepath='data/prepared_data.csv'):
    """
    Charge et prépare les données avec normalisation
    """
    print("=" * 70)
    print("CHARGEMENT ET PRÉPARATION DES DONNÉES")
    print("=" * 70)

    # Charger les données
    data = pd.read_csv(filepath)
    print(f"✓ Données chargées: {data.shape[0]} lignes, {data.shape[1]} colonnes")

    # Séparer features et target
    X = data.drop('Attrition', axis=1)
    y = data['Attrition']

    # Afficher la distribution des classes
    print(f"\nDistribution des classes:")
    print(f"  - Classe 0 (Pas d'attrition): {sum(y == 0)} ({sum(y == 0) / len(y) * 100:.1f}%)")
    print(f"  - Classe 1 (Attrition): {sum(y == 1)} ({sum(y == 1) / len(y) * 100:.1f}%)")

    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n✓ Train set: {len(X_train)} exemples")
    print(f"✓ Test set: {len(X_test)} exemples")

    # CRUCIAL: Normaliser les données pour SVM
    print("\n✓ Normalisation des données (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convertir en DataFrame pour garder les noms de colonnes
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler


def main():
    """
    Fonction principale pour exécuter tous les modèles
    """
    # Charger et préparer les données
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_prepare_data()

    print("\n" + "=" * 70)
    print("ENTRAÎNEMENT DES MODÈLES")
    print("=" * 70)

    # ========================================================================
    # MODÈLE 1: RÉGRESSION LOGISTIQUE
    # ========================================================================
    print("\n" + "-" * 70)
    print("1. RÉGRESSION LOGISTIQUE")
    print("-" * 70)

    lr_model, lr_report, lr_cm, lr_proba = logistic_regression_optimized(
        X_train, X_test, y_train, y_test,
        use_smote=True,
        use_saved_model=False
    )

    # ========================================================================
    # MODÈLE 2: SVM (VERSION RAPIDE)
    # ========================================================================
    print("\n" + "-" * 70)
    print("2. SVM (OPTIMISÉ POUR LA VITESSE)")
    print("-" * 70)

    svm_model, svm_report, svm_cm, svm_proba = svm_optimized(
        X_train, X_test, y_train, y_test,
        use_smote=True,
        use_saved_model=False
    )

    # ========================================================================
    # COMPARAISON DES MODÈLES
    # ========================================================================
    print("\n" + "=" * 70)
    print("COMPARAISON DES MODÈLES")
    print("=" * 70)

    models_dict = {
        'Logistic Regression': (lr_model, lr_proba),
        'SVM': (svm_model, svm_proba)
    }

    compare_models(models_dict, y_test)

    # ========================================================================
    # ANALYSE DES FEATURES IMPORTANTES
    # ========================================================================
    print("\n" + "=" * 70)
    print("IMPORTANCE DES CARACTÉRISTIQUES")
    print("=" * 70)

    # Pour la régression logistique
    plot_feature_importance(lr_model, feature_names, model_type='logistic', top_n=15)

    # Afficher les top features
    importance = lr_model.coef_[0]
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': importance,
        'Abs_Coefficient': np.abs(importance)
    }).sort_values('Abs_Coefficient', ascending=False)

    print("\nTop 10 features les plus importantes (Régression Logistique):")
    print(feature_importance_df.head(10)[['Feature', 'Coefficient']])

    print("\n" + "=" * 70)
    print("ANALYSE TERMINÉE")
    print("=" * 70)
    print("\n💡 RECOMMANDATIONS:")
    print("  - Les modèles sont maintenant optimisés et équilibrés")
    print("  - Le SVM est beaucoup plus rapide grâce à kernel='linear' et normalisation")
    print("  - Les features importantes sont correctement identifiées")
    print("  - Utilisez les probabilités pour ajuster le seuil si nécessaire")


if __name__ == "__main__":
    main()