#!/usr/bin/env python3
# ============================================================================
# RUN_TUNING.PY - Script pour exécuter uniquement le fine-tuning
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

from hyperparameter_tuning import run_full_tuning


def main():
    """
    Script principal pour le fine-tuning des modèles
    """

    print("=" * 80)
    print("SCRIPT DE FINE-TUNING DES MODÈLES")
    print("=" * 80)

    # Charger les données
    print("\n📂 Chargement des données...")
    data = pd.read_csv('data/prepared_data.csv')

    X = data.drop('Attrition', axis=1)
    y = data['Attrition']

    print(f"✓ Données chargées: {X.shape[0]} lignes, {X.shape[1]} colonnes")

    # Split train/test
    print("\n📊 Séparation train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"✓ Train set: {len(X_train)} exemples")
    print(f"✓ Test set: {len(X_test)} exemples")

    # Normalisation
    print("\n🔄 Normalisation des données...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    print("✓ Normalisation terminée")

    # Sauvegarder le scaler
    os.makedirs('tuned_models', exist_ok=True)
    joblib.dump(scaler, 'tuned_models/scaler.pkl')
    print("💾 Scaler sauvegardé: tuned_models/scaler.pkl")

    # Menu de configuration
    print("\n" + "=" * 80)
    print("CONFIGURATION DU FINE-TUNING")
    print("=" * 80)

    print("\n🔧 Choisissez le type de recherche:")
    print("   1. Grid Search (exhaustif, LENT mais précis)")
    print("   2. Random Search (échantillonnage aléatoire, RAPIDE)")

    search_choice = input("\nVotre choix (1 ou 2): ")
    search_type = 'grid' if search_choice == '1' else 'random'

    print("\n📊 Choisissez la taille de la grille:")
    print("   1. Grille réduite (rapide, ~5-10 min)")
    print("   2. Grille complète (lent, ~20-60 min)")

    grid_choice = input("\nVotre choix (1 ou 2): ")
    small_grid = (grid_choice == '1')

    print("\n⚖️  Utiliser SMOTE pour équilibrer les classes?")
    print("   1. Oui (RECOMMANDÉ pour classes déséquilibrées)")
    print("   2. Non")

    smote_choice = input("\nVotre choix (1 ou 2): ")
    use_smote = (smote_choice == '1')

    # Résumé de la configuration
    print("\n" + "=" * 80)
    print("RÉSUMÉ DE LA CONFIGURATION")
    print("=" * 80)
    print(f"✓ Type de recherche: {search_type.upper()}")
    print(f"✓ Taille de grille: {'Réduite' if small_grid else 'Complète'}")
    print(f"✓ SMOTE: {'Activé' if use_smote else 'Désactivé'}")
    print(f"✓ Validation croisée: 5 folds")

    # Estimation du temps
    if search_type == 'grid':
        time_estimate = "5-10 minutes" if small_grid else "20-60 minutes"
    else:
        time_estimate = "3-7 minutes" if small_grid else "10-30 minutes"

    print(f"\n⏱️  Temps estimé: {time_estimate}")

    input("\nAppuyez sur Entrée pour lancer le fine-tuning...")

    # Lancer le fine-tuning
    print("\n" + "=" * 80)
    print("DÉMARRAGE DU FINE-TUNING")
    print("=" * 80)

    tuned_models, best_params, results = run_full_tuning(
        X_train_scaled, X_test_scaled, y_train, y_test,
        use_smote=use_smote,
        search_type=search_type,
        small_grid=small_grid
    )

    # Afficher les instructions finales
    print("\n" + "=" * 80)
    print("📁 FICHIERS GÉNÉRÉS")
    print("=" * 80)
    print("\nLes fichiers suivants ont été créés dans le dossier 'tuned_models/':")
    print("\n🔹 Modèles optimisés:")
    print("   • logistic_regression_tuned.pkl")
    print("   • svm_(linear)_tuned.pkl")
    print("   • random_forest_tuned.pkl")
    print("   • xgboost_tuned.pkl")

    print("\n🔹 Résultats et analyses:")
    print("   • best_hyperparameters.csv - Meilleurs hyperparamètres")
    print("   • final_test_results.csv - Résultats sur test set")
    print("   • *_cv_results.csv - Résultats détaillés de la CV")
    print("   • *_tuning_analysis.png - Graphiques d'analyse")

    print("\n🔹 Utilitaires:")
    print("   • scaler.pkl - Objet de normalisation")

    print("\n" + "=" * 80)
    print("💡 COMMENT UTILISER LES MODÈLES OPTIMISÉS")
    print("=" * 80)

    print("""
import joblib
import pandas as pd

# 1. Charger le scaler
scaler = joblib.load('tuned_models/scaler.pkl')

# 2. Charger le meilleur modèle (exemple: Random Forest)
model = joblib.load('tuned_models/random_forest_tuned.pkl')

# 3. Préparer vos nouvelles données
# new_data = pd.read_csv('new_employees.csv')
# new_data_scaled = scaler.transform(new_data)

# 4. Faire des prédictions
# predictions = model.predict(new_data_scaled)
# probabilities = model.predict_proba(new_data_scaled)[:, 1]

# 5. Identifier les employés à risque
# at_risk = new_data[predictions == 1]
# print(f"Employés à risque: {len(at_risk)}")
    """)

    print("\n" + "=" * 80)
    print("✅ FINE-TUNING TERMINÉ AVEC SUCCÈS!")
    print("=" * 80)

    # Afficher le meilleur modèle
    best_model_row = results.iloc[0]
    print(f"\n🏆 Meilleur modèle sur le test set:")
    print(f"   Modèle: {best_model_row['Modèle']}")
    print(f"   F1-Score: {best_model_row['F1-Score']:.4f}")
    print(f"   Precision: {best_model_row['Precision']:.4f}")
    print(f"   Recall: {best_model_row['Recall']:.4f}")
    print(f"   ROC-AUC: {best_model_row['ROC-AUC']:.4f}")


if __name__ == "__main__":
    main()