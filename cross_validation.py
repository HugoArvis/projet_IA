# ============================================================================
# CROSS_VALIDATION.PY - Validation croisée pour les modèles d'attrition
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import time
import joblib
import os
from xgboost import XGBClassifier

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)


# ============================================================================
# FONCTION DE VALIDATION CROISÉE POUR UN MODÈLE
# ============================================================================

def cross_validate_model(model, X, y, model_name, n_splits=5, use_smote=True):
    """
    Effectue une validation croisée stratifiée avec SMOTE

    Paramètres:
    - model: modèle sklearn à évaluer
    - X, y: données d'entraînement
    - model_name: nom du modèle pour l'affichage
    - n_splits: nombre de folds (défaut=5)
    - use_smote: appliquer SMOTE dans chaque fold
    """

    print(f"\n{'=' * 70}")
    print(f"VALIDATION CROISÉE: {model_name}")
    print(f"{'=' * 70}")

    # Configuration du K-Fold stratifié (important pour classes déséquilibrées)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Créer un pipeline avec SMOTE si demandé
    if use_smote:
        print(f"✓ Pipeline avec SMOTE activé")
        # Pipeline imbalanced-learn qui applique SMOTE dans chaque fold
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])
    else:
        pipeline = model

    # Définir les métriques
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),
        'roc_auc': 'roc_auc'
    }

    print(f"✓ Stratified K-Fold avec {n_splits} folds")
    print(f"✓ Métriques: Accuracy, Precision, Recall, F1-Score, ROC-AUC")
    print(f"\n🔄 Exécution de la validation croisée...")

    start_time = time.time()

    # Exécuter la validation croisée
    cv_results = cross_validate(
        pipeline, X, y,
        cv=skf,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
        verbose=0
    )

    cv_time = time.time() - start_time

    # Calculer les statistiques
    results = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']

        results[metric] = {
            'test_mean': test_scores.mean(),
            'test_std': test_scores.std(),
            'train_mean': train_scores.mean(),
            'train_std': train_scores.std(),
            'diff': train_scores.mean() - test_scores.mean()
        }

    # Afficher les résultats
    print(f"\n✓ Validation croisée terminée en {cv_time:.2f}s")
    print(f"\n{'─' * 70}")
    print(f"📊 RÉSULTATS DE LA VALIDATION CROISÉE")
    print(f"{'─' * 70}")

    for metric_name, metric_results in results.items():
        print(f"\n{metric_name.upper()}:")
        print(f"  Test  : {metric_results['test_mean']:.4f} (± {metric_results['test_std']:.4f})")
        print(f"  Train : {metric_results['train_mean']:.4f} (± {metric_results['train_std']:.4f})")
        print(f"  Diff  : {metric_results['diff']:.4f}", end="")

        # Indicateur de sur-apprentissage
        if metric_results['diff'] > 0.10:
            print(" ⚠️ SUR-APPRENTISSAGE DÉTECTÉ")
        elif metric_results['diff'] > 0.05:
            print(" ⚡ Léger sur-apprentissage")
        else:
            print(" ✓ Bon équilibre")

    # Scores individuels de chaque fold
    print(f"\n{'─' * 70}")
    print(f"SCORES PAR FOLD (F1-Score):")
    print(f"{'─' * 70}")
    for i, score in enumerate(cv_results['test_f1'], 1):
        print(f"  Fold {i}: {score:.4f}")

    return results, cv_results


# ============================================================================
# VALIDATION CROISÉE POUR TOUS LES MODÈLES
# ============================================================================

def cross_validate_all_models(X, y, use_smote=True, n_splits=5):
    """
    Effectue la validation croisée sur tous les modèles
    """

    print("=" * 70)
    print("VALIDATION CROISÉE - COMPARAISON DES MODÈLES")
    print("=" * 70)
    print(f"\n✓ Dataset: {X.shape[0]} exemples, {X.shape[1]} features")
    print(f"✓ Nombre de folds: {n_splits}")
    print(f"✓ SMOTE: {'Activé' if use_smote else 'Désactivé'}")
    print(f"✓ Distribution: Classe 0={sum(y == 0)} ({sum(y == 0) / len(y) * 100:.1f}%), "
          f"Classe 1={sum(y == 1)} ({sum(y == 1) / len(y) * 100:.1f}%)")

    # Définir les modèles à tester
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=42,
            class_weight='balanced',
            solver='lbfgs',
            n_jobs=-1
        ),
        'SVM (Linear)': SVC(
            kernel='linear',
            C=1.0,
            random_state=42,
            class_weight='balanced',
            probability=True,
            cache_size=1000
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=1,
            use_label_encoder=False,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
    }

    # Stocker tous les résultats
    all_results = {}
    all_cv_results = {}

    # Évaluer chaque modèle
    for model_name, model in models.items():
        results, cv_results = cross_validate_model(
            model, X, y, model_name,
            n_splits=n_splits,
            use_smote=use_smote
        )
        all_results[model_name] = results
        all_cv_results[model_name] = cv_results

    return all_results, all_cv_results


# ============================================================================
# TABLEAU COMPARATIF
# ============================================================================

def create_comparison_table(all_results):
    """
    Crée un tableau comparatif des performances
    """

    print("\n" + "=" * 70)
    print("TABLEAU COMPARATIF FINAL")
    print("=" * 70)

    # Créer le DataFrame
    comparison_data = []
    for model_name, results in all_results.items():
        comparison_data.append({
            'Modèle': model_name,
            'Accuracy': f"{results['accuracy']['test_mean']:.4f} (±{results['accuracy']['test_std']:.4f})",
            'Precision': f"{results['precision']['test_mean']:.4f} (±{results['precision']['test_std']:.4f})",
            'Recall': f"{results['recall']['test_mean']:.4f} (±{results['recall']['test_std']:.4f})",
            'F1-Score': f"{results['f1']['test_mean']:.4f} (±{results['f1']['test_std']:.4f})",
            'ROC-AUC': f"{results['roc_auc']['test_mean']:.4f} (±{results['roc_auc']['test_std']:.4f})",
            'F1_Mean': results['f1']['test_mean'],  # Pour le tri
            'Overfitting': 'Oui' if results['f1']['diff'] > 0.10 else 'Non'
        })

    df = pd.DataFrame(comparison_data)
    df = df.sort_values('F1_Mean', ascending=False)

    # Afficher le tableau
    display_df = df.drop('F1_Mean', axis=1)
    print("\n" + display_df.to_string(index=False))

    # Identifier le meilleur modèle
    best_model = df.iloc[0]
    print(f"\n{'=' * 70}")
    print(f"🏆 MEILLEUR MODÈLE: {best_model['Modèle']}")
    print(f"{'=' * 70}")
    print(f"  F1-Score: {best_model['F1-Score']}")
    print(f"  ROC-AUC: {best_model['ROC-AUC']}")
    print(f"  Sur-apprentissage: {best_model['Overfitting']}")

    return df


# ============================================================================
# VISUALISATIONS
# ============================================================================

def plot_cv_results(all_results, all_cv_results):
    """
    Crée des visualisations des résultats de validation croisée
    """

    print("\n" + "=" * 70)
    print("GÉNÉRATION DES VISUALISATIONS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    models = list(all_results.keys())

    # 1. Comparaison F1-Scores
    f1_means = [all_results[m]['f1']['test_mean'] for m in models]
    f1_stds = [all_results[m]['f1']['test_std'] for m in models]

    axes[0, 0].barh(models, f1_means, xerr=f1_stds, capsize=5, alpha=0.7, color='steelblue')
    axes[0, 0].set_xlabel('F1-Score', fontsize=12)
    axes[0, 0].set_title('Comparaison des F1-Scores (avec écart-type)', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')

    # 2. Comparaison ROC-AUC
    roc_means = [all_results[m]['roc_auc']['test_mean'] for m in models]
    roc_stds = [all_results[m]['roc_auc']['test_std'] for m in models]

    axes[0, 1].barh(models, roc_means, xerr=roc_stds, capsize=5, alpha=0.7, color='coral')
    axes[0, 1].set_xlabel('ROC-AUC Score', fontsize=12)
    axes[0, 1].set_title('Comparaison des ROC-AUC Scores', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')

    # 3. Train vs Test (détection sur-apprentissage)
    train_f1 = [all_results[m]['f1']['train_mean'] for m in models]
    test_f1 = [all_results[m]['f1']['test_mean'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    axes[1, 0].barh(x - width / 2, train_f1, width, label='Train', alpha=0.7, color='lightgreen')
    axes[1, 0].barh(x + width / 2, test_f1, width, label='Test', alpha=0.7, color='lightcoral')
    axes[1, 0].set_yticks(x)
    axes[1, 0].set_yticklabels(models)
    axes[1, 0].set_xlabel('F1-Score', fontsize=12)
    axes[1, 0].set_title('Train vs Test F1-Score (Détection Sur-apprentissage)',
                         fontsize=14, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # 4. Boxplot des scores par fold
    f1_scores_by_model = {}
    for model_name in models:
        f1_scores_by_model[model_name] = all_cv_results[model_name]['test_f1']

    bp_data = [f1_scores_by_model[name] for name in models]
    bp = axes[1, 1].boxplot(bp_data, labels=models, patch_artist=True, vert=False)

    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    axes[1, 1].set_xlabel('F1-Score', fontsize=12)
    axes[1, 1].set_title('Distribution des F1-Scores par Fold', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    # Sauvegarder
    os.makedirs('cv_results', exist_ok=True)
    plt.savefig('cv_results/cross_validation_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Graphiques sauvegardés: cv_results/cross_validation_comparison.png")
    plt.close()


# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def run_cross_validation_analysis(filepath='data/prepared_data.csv', n_splits=5):
    """
    Fonction principale pour exécuter l'analyse complète avec validation croisée
    """

    print("=" * 70)
    print("ANALYSE AVEC VALIDATION CROISÉE")
    print("=" * 70)

    # Charger les données
    print("\n📂 Chargement des données...")
    data = pd.read_csv(filepath)

    X = data.drop('Attrition', axis=1)
    y = data['Attrition']

    print(f"✓ Données chargées: {X.shape[0]} lignes, {X.shape[1]} colonnes")

    # Validation croisée sur tous les modèles
    all_results, all_cv_results = cross_validate_all_models(
        X, y,
        use_smote=True,
        n_splits=n_splits
    )

    # Créer le tableau comparatif
    comparison_df = create_comparison_table(all_results)

    # Visualisations
    plot_cv_results(all_results, all_cv_results)

    # Sauvegarder les résultats
    os.makedirs('cv_results', exist_ok=True)
    comparison_df.to_csv('cv_results/cv_comparison_table.csv', index=False)
    print("\n✓ Tableau comparatif sauvegardé: cv_results/cv_comparison_table.csv")

    # Recommandations
    print("\n" + "=" * 70)
    print("💡 RECOMMANDATIONS")
    print("=" * 70)

    best_model_name = comparison_df.iloc[0]['Modèle']
    best_f1 = all_results[best_model_name]['f1']['test_mean']
    best_overfitting = all_results[best_model_name]['f1']['diff']

    print(f"""
🎯 Le meilleur modèle est: {best_model_name}
   - F1-Score moyen: {best_f1:.4f}
   - Écart Train/Test: {best_overfitting:.4f}

✅ Avantages de la validation croisée:
   - Estimation plus robuste des performances
   - Détection du sur-apprentissage
   - Utilisation de toutes les données
   - Réduction de la variance des estimations

📊 Prochaines étapes:
   1. Optimiser les hyperparamètres du meilleur modèle (GridSearchCV)
   2. Analyser les erreurs sur les différents folds
   3. Tester des techniques d'ensemble (stacking, voting)
   4. Valider sur un test set final complètement séparé
   5. Analyser l'importance des features
""")

    return all_results, all_cv_results, comparison_df