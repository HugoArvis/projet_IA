# ============================================================================
# HYPERPARAMETER_TUNING.PY - Fine-tuning des hyperparamètres
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    make_scorer, f1_score, precision_score, recall_score,
    roc_auc_score, classification_report, confusion_matrix
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import time
import joblib
import os
from xgboost import XGBClassifier

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)


# ============================================================================
# GRILLES D'HYPERPARAMÈTRES
# ============================================================================

def get_param_grids():
    """
    Définit les grilles d'hyperparamètres pour chaque modèle
    """
    param_grids = {
        'Logistic Regression': {
            'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
            'classifier__penalty': ['l2'],
            'classifier__solver': ['lbfgs', 'liblinear'],
            'classifier__max_iter': [2000, 3000],
            'classifier__class_weight': ['balanced', None]
        },

        'SVM (Linear)': {
            'classifier__C': [0.01, 0.1, 1, 10],
            'classifier__kernel': ['linear'],
            'classifier__class_weight': ['balanced', None],
            'classifier__cache_size': [1000]
        },

        'Random Forest': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [5, 10, 15, 20, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__class_weight': ['balanced', 'balanced_subsample', None],
            'classifier__max_features': ['sqrt', 'log2']
        },

        'XGBoost': {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 5, 7, 10],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__subsample': [0.7, 0.8, 0.9, 1.0],
            'classifier__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'classifier__scale_pos_weight': [1, 2, 3, 5]
        }
    }

    return param_grids


def get_smaller_param_grids():
    """
    Grilles réduites pour des recherches plus rapides
    """
    param_grids = {
        'Logistic Regression': {
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l2'],
            'classifier__max_iter': [2000],
            'classifier__class_weight': ['balanced']
        },

        'SVM (Linear)': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear'],
            'classifier__class_weight': ['balanced']
        },

        'Random Forest': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 15, None],
            'classifier__min_samples_split': [2, 5],
            'classifier__class_weight': ['balanced']
        },

        'XGBoost': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [5, 7],
            'classifier__learning_rate': [0.05, 0.1],
            'classifier__scale_pos_weight': [2, 3]
        }
    }

    return param_grids


# ============================================================================
# FINE-TUNING D'UN MODÈLE SPÉCIFIQUE
# ============================================================================

def tune_model(model, param_grid, X_train, y_train, model_name,
               use_smote=True, search_type='grid', n_iter=50, cv=5):
    """
    Effectue le fine-tuning d'un modèle avec GridSearchCV ou RandomizedSearchCV

    Paramètres:
    -----------
    model : modèle sklearn à optimiser
    param_grid : dictionnaire des hyperparamètres à tester
    X_train, y_train : données d'entraînement
    model_name : nom du modèle
    use_smote : appliquer SMOTE dans le pipeline
    search_type : 'grid' ou 'random'
    n_iter : nombre d'itérations pour RandomizedSearchCV
    cv : nombre de folds pour la validation croisée
    """

    print(f"\n{'=' * 80}")
    print(f"FINE-TUNING: {model_name}")
    print(f"{'=' * 80}")

    # Créer le pipeline avec SMOTE si demandé
    if use_smote:
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])
        print("✓ Pipeline avec SMOTE créé")
    else:
        pipeline = ImbPipeline([
            ('classifier', model)
        ])

    # Configuration de la validation croisée
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Scorer principal : F1-Score
    scorer = make_scorer(f1_score)

    # Afficher les informations
    print(f"✓ Type de recherche: {search_type.upper()}")
    print(f"✓ Nombre de folds: {cv}")
    print(f"✓ Métrique d'optimisation: F1-Score")

    if search_type == 'grid':
        n_combinations = 1
        for key, values in param_grid.items():
            n_combinations *= len(values)
        print(f"✓ Nombre de combinaisons à tester: {n_combinations}")
        print(f"✓ Total d'entraînements: {n_combinations * cv}")
    else:
        print(f"✓ Nombre d'itérations: {n_iter}")
        print(f"✓ Total d'entraînements: {n_iter * cv}")

    # Créer l'objet de recherche
    start_time = time.time()

    if search_type == 'grid':
        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=skf,
            scoring=scorer,
            n_jobs=-1,
            verbose=1,
            return_train_score=True
        )
    else:  # random
        search = RandomizedSearchCV(
            pipeline,
            param_grid,
            n_iter=n_iter,
            cv=skf,
            scoring=scorer,
            n_jobs=-1,
            verbose=1,
            return_train_score=True,
            random_state=42
        )

    print(f"\n🔄 Lancement de la recherche d'hyperparamètres...")
    print(f"   (Cela peut prendre plusieurs minutes...)\n")

    # Lancer la recherche
    search.fit(X_train, y_train)

    search_time = time.time() - start_time

    # Résultats
    print(f"\n✓ Recherche terminée en {search_time:.2f}s ({search_time / 60:.2f} min)")
    print(f"\n{'─' * 80}")
    print("RÉSULTATS DU FINE-TUNING")
    print(f"{'─' * 80}")

    print(f"\n🏆 Meilleur score (F1) : {search.best_score_:.4f}")
    print(f"\n📊 Meilleurs hyperparamètres:")
    for param, value in search.best_params_.items():
        param_name = param.replace('classifier__', '')
        print(f"   • {param_name}: {value}")

    # Analyse des scores
    cv_results = pd.DataFrame(search.cv_results_)

    # Top 5 des combinaisons
    print(f"\n📈 Top 5 des meilleures combinaisons:")
    top_5 = cv_results.nlargest(5, 'mean_test_score')[
        ['mean_test_score', 'std_test_score', 'params']
    ]
    for idx, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"\n   {idx}. Score: {row['mean_test_score']:.4f} (±{row['std_test_score']:.4f})")
        params_clean = {k.replace('classifier__', ''): v
                        for k, v in row['params'].items()}
        for param, value in params_clean.items():
            print(f"      {param}: {value}")

    # Sauvegarder le meilleur modèle
    os.makedirs('tuned_models', exist_ok=True)
    model_path = f'tuned_models/{model_name.lower().replace(" ", "_")}_tuned.pkl'
    joblib.dump(search.best_estimator_, model_path)
    print(f"\n💾 Meilleur modèle sauvegardé: {model_path}")

    # Sauvegarder les résultats de la recherche
    results_path = f'tuned_models/{model_name.lower().replace(" ", "_")}_cv_results.csv'
    cv_results.to_csv(results_path, index=False)
    print(f"💾 Résultats CV sauvegardés: {results_path}")

    return search.best_estimator_, search.best_params_, search.best_score_, cv_results


# ============================================================================
# FINE-TUNING DE TOUS LES MODÈLES
# ============================================================================

def tune_all_models(X_train, y_train, use_smote=True, search_type='grid',
                    small_grid=False, cv=5):
    """
    Effectue le fine-tuning de tous les modèles

    Paramètres:
    -----------
    X_train, y_train : données d'entraînement
    use_smote : appliquer SMOTE
    search_type : 'grid' ou 'random'
    small_grid : utiliser une grille réduite (plus rapide)
    cv : nombre de folds
    """

    print("=" * 80)
    print("FINE-TUNING DE TOUS LES MODÈLES")
    print("=" * 80)

    print(f"\n✓ Dataset: {X_train.shape[0]} exemples, {X_train.shape[1]} features")
    print(f"✓ Type de recherche: {search_type.upper()}")
    print(f"✓ Grille: {'Réduite (rapide)' if small_grid else 'Complète (lent)'}")
    print(f"✓ SMOTE: {'Activé' if use_smote else 'Désactivé'}")
    print(f"✓ Cross-validation: {cv} folds")

    # Définir les modèles
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42,
            solver='lbfgs',
            n_jobs=-1
        ),
        'SVM (Linear)': SVC(
            probability=True,
            random_state=42,
            cache_size=1000
        ),
        'Random Forest': RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
    }

    # Obtenir les grilles de paramètres
    param_grids = get_smaller_param_grids() if small_grid else get_param_grids()

    # Stocker les résultats
    tuned_models = {}
    best_params_all = {}
    best_scores_all = {}

    # Tuner chaque modèle
    for model_name, model in models.items():
        best_model, best_params, best_score, cv_results = tune_model(
            model,
            param_grids[model_name],
            X_train, y_train,
            model_name,
            use_smote=use_smote,
            search_type=search_type,
            cv=cv
        )

        tuned_models[model_name] = best_model
        best_params_all[model_name] = best_params
        best_scores_all[model_name] = best_score

    return tuned_models, best_params_all, best_scores_all


# ============================================================================
# ÉVALUATION DES MODÈLES TUNÉS
# ============================================================================

def evaluate_tuned_models(tuned_models, X_test, y_test):
    """
    Évalue les performances des modèles tunés sur le test set
    """

    print("\n" + "=" * 80)
    print("ÉVALUATION DES MODÈLES TUNÉS (TEST SET)")
    print("=" * 80)

    results = []

    for model_name, model in tuned_models.items():
        print(f"\n{'─' * 80}")
        print(f"Modèle: {model_name}")
        print(f"{'─' * 80}")

        # Prédictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Métriques
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        # Classification report
        print(classification_report(y_test, y_pred,
                                    target_names=['No Attrition', 'Attrition']))

        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nMatrice de confusion:")
        print(f"  TN={cm[0, 0]:<5} FP={cm[0, 1]:<5}")
        print(f"  FN={cm[1, 0]:<5} TP={cm[1, 1]:<5}")

        print(f"\n🎯 Métriques principales:")
        print(f"   F1-Score  : {f1:.4f}")
        print(f"   Precision : {precision:.4f}")
        print(f"   Recall    : {recall:.4f}")
        print(f"   ROC-AUC   : {roc_auc:.4f}")

        results.append({
            'Modèle': model_name,
            'F1-Score': f1,
            'Precision': precision,
            'Recall': recall,
            'ROC-AUC': roc_auc
        })

    # Tableau comparatif
    results_df = pd.DataFrame(results).sort_values('F1-Score', ascending=False)

    print("\n" + "=" * 80)
    print("TABLEAU COMPARATIF FINAL")
    print("=" * 80)
    print("\n" + results_df.to_string(index=False))

    # Meilleur modèle
    best_model = results_df.iloc[0]
    print(f"\n{'=' * 80}")
    print(f"🏆 MEILLEUR MODÈLE: {best_model['Modèle']}")
    print(f"{'=' * 80}")
    print(f"   F1-Score  : {best_model['F1-Score']:.4f}")
    print(f"   Precision : {best_model['Precision']:.4f}")
    print(f"   Recall    : {best_model['Recall']:.4f}")
    print(f"   ROC-AUC   : {best_model['ROC-AUC']:.4f}")

    return results_df


# ============================================================================
# VISUALISATION DES RÉSULTATS
# ============================================================================

def plot_tuning_results(cv_results_dict, model_name):
    """
    Visualise les résultats du tuning pour un modèle
    """
    cv_results = cv_results_dict

    # Prendre les top 20 combinaisons
    top_results = cv_results.nlargest(20, 'mean_test_score')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Scores des top 20 combinaisons
    x = range(len(top_results))
    axes[0].barh(x, top_results['mean_test_score'],
                 xerr=top_results['std_test_score'],
                 alpha=0.7, color='steelblue', capsize=5)
    axes[0].set_yticks(x)
    axes[0].set_yticklabels([f"Config {i + 1}" for i in x])
    axes[0].set_xlabel('F1-Score')
    axes[0].set_title(f'Top 20 Configurations - {model_name}',
                      fontweight='bold', fontsize=14)
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].invert_yaxis()

    # 2. Train vs Test scores
    axes[1].scatter(top_results['mean_train_score'],
                    top_results['mean_test_score'],
                    alpha=0.6, s=100, color='coral')
    axes[1].plot([0.5, 1.0], [0.5, 1.0], 'r--', lw=2,
                 label='Train = Test (idéal)')
    axes[1].set_xlabel('Train F1-Score')
    axes[1].set_ylabel('Test F1-Score')
    axes[1].set_title('Train vs Test Scores (Overfitting Detection)',
                      fontweight='bold', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs('tuned_models', exist_ok=True)
    filename = f'tuned_models/{model_name.lower().replace(" ", "_")}_tuning_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Graphique sauvegardé: {filename}")
    plt.close()


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def run_full_tuning(X_train, X_test, y_train, y_test,
                    use_smote=True, search_type='grid', small_grid=True):
    """
    Exécute le processus complet de fine-tuning
    """

    print("=" * 80)
    print("PROCESSUS COMPLET DE FINE-TUNING")
    print("=" * 80)

    # Étape 1: Fine-tuning
    print("\n📍 ÉTAPE 1: FINE-TUNING DES HYPERPARAMÈTRES")
    tuned_models, best_params, best_scores = tune_all_models(
        X_train, y_train,
        use_smote=use_smote,
        search_type=search_type,
        small_grid=small_grid,
        cv=5
    )

    # Étape 2: Évaluation sur test set
    print("\n📍 ÉTAPE 2: ÉVALUATION SUR TEST SET")
    results_df = evaluate_tuned_models(tuned_models, X_test, y_test)

    # Étape 3: Sauvegarder le résumé
    print("\n📍 ÉTAPE 3: SAUVEGARDE DES RÉSULTATS")
    os.makedirs('tuned_models', exist_ok=True)

    # Sauvegarder les meilleurs hyperparamètres
    best_params_df = pd.DataFrame([
        {'Model': model, 'Parameter': param.replace('classifier__', ''), 'Value': value}
        for model, params in best_params.items()
        for param, value in params.items()
    ])
    best_params_df.to_csv('tuned_models/best_hyperparameters.csv', index=False)
    print("✓ Meilleurs hyperparamètres sauvegardés: tuned_models/best_hyperparameters.csv")

    # Sauvegarder les résultats finaux
    results_df.to_csv('tuned_models/final_test_results.csv', index=False)
    print("✓ Résultats finaux sauvegardés: tuned_models/final_test_results.csv")

    print("\n" + "=" * 80)
    print("✅ FINE-TUNING TERMINÉ AVEC SUCCÈS")
    print("=" * 80)

    print("\n💡 RECOMMANDATIONS:")
    print("   1. Vérifiez les graphiques de tuning dans tuned_models/")
    print("   2. Les modèles optimisés sont dans tuned_models/")
    print("   3. Utilisez ces modèles pour vos prédictions en production")
    print("   4. Réévaluez périodiquement avec de nouvelles données")

    return tuned_models, best_params, results_df


# ============================================================================
# EXÉCUTION
# ============================================================================

if __name__ == "__main__":
    # Charger les données
    print("Chargement des données...")
    data = pd.read_csv('data/prepared_data.csv')

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X = data.drop('Attrition', axis=1)
    y = data['Attrition']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    # Lancer le fine-tuning
    tuned_models, best_params, results = run_full_tuning(
        X_train_scaled, X_test_scaled, y_train, y_test,
        use_smote=True,
        search_type='grid',  # ou 'random' pour plus rapide
        small_grid=True  # True pour tests rapides, False pour recherche exhaustive
    )