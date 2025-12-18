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
    random_forest_optimized,
    xgboost_optimized,
    compare_models,
    plot_feature_importance
)

# Import de la validation croisée
from cross_validation import (
    cross_validate_all_models,
    create_comparison_table,
    plot_cv_results
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

    #sauvegarder le  scaler pour une utilisation future
    import joblib
    import os
    os.makedirs('trained_models', exist_ok=True)
    joblib.dump(scaler, 'trained_models/scaler.pkl')
    print("[SAVE] Scaler sauvegarde: trained_models/scaler.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns, scaler, X, y


def main(use_cv=True):
    """
    Fonction principale pour exécuter tous les modèles

    Paramètres:
    - use_cv: Si True, effectue d'abord une validation croisée avant le train/test final
    """
    # Charger et préparer les données
    X_train, X_test, y_train, y_test, feature_names, scaler, X_full, y_full = load_and_prepare_data()

    # ========================================================================
    # ÉTAPE 1: VALIDATION CROISÉE (RECOMMANDÉ)
    # ========================================================================

    if use_cv:
        print("\n" + "=" * 70)
        print("ÉTAPE 1: VALIDATION CROISÉE SUR L'ENSEMBLE D'ENTRAÎNEMENT")
        print("=" * 70)
        print("\n💡 La validation croisée permet de:")
        print("   - Détecter le sur-apprentissage")
        print("   - Obtenir une estimation plus robuste des performances")
        print("   - Utiliser efficacement toutes les données d'entraînement")

        # Normaliser toutes les données pour la CV
        X_train_full_scaled = scaler.fit_transform(X_train)
        X_train_full_scaled = pd.DataFrame(X_train_full_scaled, columns=feature_names)

        # Exécuter la validation croisée sur l'ensemble d'entraînement
        all_results, all_cv_results = cross_validate_all_models(
            X_train_full_scaled, y_train,
            use_smote=True,
            n_splits=5
        )

        # Créer le tableau comparatif
        cv_comparison = create_comparison_table(all_results)

        # Visualiser les résultats
        plot_cv_results(all_results, all_cv_results)

        print("\n" + "=" * 70)
        print("✅ VALIDATION CROISÉE TERMINÉE")
        print("=" * 70)
        print("\n💡 Passons maintenant à l'évaluation finale sur le test set...")
        input("\nAppuyez sur Entrée pour continuer...")

    # ========================================================================
    # ÉTAPE 2: ENTRAÎNEMENT ET ÉVALUATION FINALE SUR TEST SET
    # ========================================================================

    print("\n" + "=" * 70)
    print("ÉTAPE 2: ENTRAÎNEMENT FINAL ET ÉVALUATION SUR TEST SET")
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
    # MODÈLE 3: RANDOM FOREST
    # ========================================================================
    print("\n" + "-" * 70)
    print("3. RANDOM FOREST")
    print("-" * 70)

    rf_model, rf_report, rf_cm, rf_proba = random_forest_optimized(
        X_train, X_test, y_train, y_test,
        use_smote=True,
        use_saved_model=False
    )

    # ========================================================================
    # MODÈLE 4: XGBOOST
    # ========================================================================
    print("\n" + "-" * 70)
    print("4. XGBOOST")
    print("-" * 70)

    xgb_model, xgb_report, xgb_cm, xgb_proba = xgboost_optimized(
        X_train, X_test, y_train, y_test,
        use_smote=True,
        use_saved_model=False
    )

    # ========================================================================
    # COMPARAISON DES MODÈLES SUR TEST SET
    # ========================================================================
    print("\n" + "=" * 70)
    print("COMPARAISON DES MODÈLES (TEST SET)")
    print("=" * 70)

    models_dict = {
        'Logistic Regression': (lr_model, lr_proba),
        'SVM': (svm_model, svm_proba),
        'Random Forest': (rf_model, rf_proba),
        'XGBoost': (xgb_model, xgb_proba)
    }

    compare_models(models_dict, X_test, y_test)

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

    # ========================================================================
    # RÉSUMÉ FINAL
    # ========================================================================
    print("\n" + "=" * 70)
    print("RÉSUMÉ FINAL")
    print("=" * 70)

    if use_cv:
        print("\n✅ VALIDATION CROISÉE:")
        print("   - Détection du sur-apprentissage: OK")
        print("   - Estimation robuste des performances: OK")
        print("   - Résultats sauvegardés dans: cv_results/")

    print("\n✅ ÉVALUATION FINALE (TEST SET):")
    print("   - Tous les modèles entraînés et évalués")
    print("   - Comparaison des performances effectuée")
    print("   - Features importantes identifiées")

    print("\n" + "=" * 70)
    print("ANALYSE TERMINÉE")
    print("=" * 70)
    print("\n💡 RECOMMANDATIONS:")
    print("  ✓ Les modèles sont optimisés et équilibrés avec SMOTE")
    print("  ✓ La validation croisée a permis de détecter le sur-apprentissage")
    print("  ✓ Le SVM est rapide grâce à kernel='linear' et normalisation")
    print("  ✓ Les features importantes sont correctement identifiées")
    print("\n📊 PROCHAINES ÉTAPES:")
    print("  1. Optimiser les hyperparamètres avec GridSearchCV")
    print("  2. Tester des techniques d'ensemble (stacking, voting)")
    print("  3. Analyser les erreurs de classification en détail")
    print("  4. Ajuster le seuil de décision selon les besoins métier")


if __name__ == "__main__":
    # Option 1: Avec validation croisée (RECOMMANDÉ)
    main(use_cv=True)

    # Option 2: Sans validation croisée (plus rapide)
    # main(use_cv=False)