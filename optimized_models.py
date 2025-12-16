# ============================================================================
# OPTIMIZED_MODELS.PY - Modèles ML optimisés pour l'attrition
# ============================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score
)
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import joblib
import time


# ============================================================================
# RÉGRESSION LOGISTIQUE OPTIMISÉE
# ============================================================================

def logistic_regression_optimized(X_train, X_test, y_train, y_test,
                                  use_smote=True, use_saved_model=False,
                                  C=1.0, max_iter=2000):
    """
    Régression logistique optimisée avec SMOTEENN

    Paramètres:
    - use_smote: utiliser SMOTEENN pour équilibrer les données
    - use_saved_model: charger un modèle existant si disponible
    """
    os.makedirs('trained_models', exist_ok=True)
    model_path = 'trained_models/logistic_regression_optimized.pkl'

    # Charger modèle existant si demandé
    if use_saved_model and os.path.exists(model_path):
        print("📁 Chargement du modèle existant...")
        model = joblib.load(model_path)
    else:
        # Appliquer SMOTEENN si demandé
        if use_smote:
            print("⚖️  Application de SMOTEENN pour équilibrer les données...")
            print(f"   Avant: Classe 0={sum(y_train == 0)}, Classe 1={sum(y_train == 1)}")

            smoteenn = SMOTEENN(random_state=42)
            X_train_balanced, y_train_balanced = smoteenn.fit_resample(X_train, y_train)

            print(f"   Après: Classe 0={sum(y_train_balanced == 0)}, Classe 1={sum(y_train_balanced == 1)}")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train

        # Entraîner le modèle
        print("🔧 Entraînement de la régression logistique...")
        start_time = time.time()

        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=42,
            class_weight='balanced',  # Important même avec SMOTE
            solver='lbfgs',
            n_jobs=-1  # Parallélisation
        )
        model.fit(X_train_balanced, y_train_balanced)

        training_time = time.time() - start_time
        print(f"✓ Entraînement terminé en {training_time:.2f}s")

        # Sauvegarder
        joblib.dump(model, model_path)
        print(f"💾 Modèle sauvegardé: {model_path}")

    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Évaluation
    print("\n📊 RÉSULTATS - RÉGRESSION LOGISTIQUE")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=['No Attrition', 'Attrition']))

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nMatrice de confusion:")
    print(f"  TN={cm[0, 0]:<5} FP={cm[0, 1]:<5}")
    print(f"  FN={cm[1, 0]:<5} TP={cm[1, 1]:<5}")

    # Métriques supplémentaires
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\n🎯 ROC-AUC Score: {roc_auc:.4f}")

    return model, classification_report(y_test, y_pred), cm, y_proba


# ============================================================================
# SVM OPTIMISÉ (RAPIDE)
# ============================================================================

def svm_optimized(X_train, X_test, y_train, y_test,
                  use_smote=True, use_saved_model=False,
                  kernel='linear', C=1.0):
    """
    SVM optimisé pour la vitesse

    IMPORTANT:
    - kernel='linear' est BEAUCOUP plus rapide que 'rbf' pour des données à haute dimension
    - Les données DOIVENT être normalisées (StandardScaler) avant
    - Utilisez class_weight='balanced' pour gérer le déséquilibre
    """
    os.makedirs('trained_models', exist_ok=True)
    model_path = 'trained_models/svm_optimized.pkl'

    # Charger modèle existant si demandé
    if use_saved_model and os.path.exists(model_path):
        print("📁 Chargement du modèle SVM existant...")
        model = joblib.load(model_path)
    else:
        # Appliquer SMOTE (pas SMOTEENN car SVM est déjà lent)
        if use_smote:
            print("⚖️  Application de SMOTE pour équilibrer les données...")
            print(f"   Avant: Classe 0={sum(y_train == 0)}, Classe 1={sum(y_train == 1)}")

            # SMOTE seul est plus rapide que SMOTEENN
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

            print(f"   Après: Classe 0={sum(y_train_balanced == 0)}, Classe 1={sum(y_train_balanced == 1)}")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train

        # Entraîner le modèle
        print("🔧 Entraînement du SVM (cela peut prendre du temps)...")
        start_time = time.time()

        model = SVC(
            kernel=kernel,  # 'linear' est BEAUCOUP plus rapide
            C=C,
            random_state=42,
            class_weight='balanced',
            probability=True,  # Nécessaire pour predict_proba
            cache_size=1000  # Augmente la vitesse
        )
        model.fit(X_train_balanced, y_train_balanced)

        training_time = time.time() - start_time
        print(f"✓ Entraînement terminé en {training_time:.2f}s")

        # Sauvegarder
        joblib.dump(model, model_path)
        print(f"💾 Modèle sauvegardé: {model_path}")

    # Prédictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Évaluation
    print("\n📊 RÉSULTATS - SVM")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=['No Attrition', 'Attrition']))

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nMatrice de confusion:")
    print(f"  TN={cm[0, 0]:<5} FP={cm[0, 1]:<5}")
    print(f"  FN={cm[1, 0]:<5} TP={cm[1, 1]:<5}")

    # Métriques supplémentaires
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\n🎯 ROC-AUC Score: {roc_auc:.4f}")

    return model, classification_report(y_test, y_pred), cm, y_proba


# ============================================================================
# FONCTION DE COMPARAISON
# ============================================================================

def compare_models(models_dict, y_test):
    """
    Compare les performances de plusieurs modèles

    models_dict: {'nom_modele': (model, y_proba), ...}
    """
    print("\n" + "=" * 70)
    print("TABLEAU COMPARATIF DES PERFORMANCES")
    print("=" * 70)

    results = []

    for name, (model, y_proba) in models_dict.items():
        y_pred = model.predict(y_proba.reshape(-1, 1) if len(y_proba.shape) == 1 else y_proba) if hasattr(model,
                                                                                                          'predict') else (
                    y_proba > 0.5).astype(int)

        # Recalculer avec les vrais y_pred du modèle
        if hasattr(model, 'predict'):
            # Pour récupérer X_test, on doit le passer différemment
            # Utilisons directement les prédictions déjà calculées
            pass

        f1 = f1_score(y_test, (y_proba > 0.5).astype(int))
        precision = precision_score(y_test, (y_proba > 0.5).astype(int))
        recall = recall_score(y_test, (y_proba > 0.5).astype(int))
        roc_auc = roc_auc_score(y_test, y_proba)

        results.append({
            'Modèle': name,
            'F1-Score': f1,
            'Precision': precision,
            'Recall': recall,
            'ROC-AUC': roc_auc
        })

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    # Identifier le meilleur modèle
    best_model = df_results.loc[df_results['F1-Score'].idxmax(), 'Modèle']
    print(f"\n🏆 Meilleur modèle (F1-Score): {best_model}")


# ============================================================================
# VISUALISATION DE L'IMPORTANCE DES FEATURES
# ============================================================================

def plot_feature_importance(model, feature_names, model_type='logistic', top_n=15):
    """
    Visualise l'importance des features

    model_type: 'logistic' ou 'svm' (linéaire uniquement)
    """
    if model_type == 'logistic' or (model_type == 'svm' and hasattr(model, 'coef_')):
        importance = model.coef_[0]

        # Créer DataFrame
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance,
            'Abs_Importance': np.abs(importance)
        }).sort_values('Abs_Importance', ascending=False).head(top_n)

        # Plot
        plt.figure(figsize=(12, 8))
        colors = ['red' if x < 0 else 'green' for x in feature_importance_df['Importance']]

        plt.barh(range(len(feature_importance_df)),
                 feature_importance_df['Importance'],
                 color=colors, alpha=0.7)

        plt.yticks(range(len(feature_importance_df)),
                   feature_importance_df['Feature'])
        plt.xlabel('Coefficient (Impact sur l\'attrition)', fontsize=12)
        plt.title(f'Top {top_n} Features - {model_type.upper()}', fontsize=14, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)

        # Légende
        plt.text(0.02, 0.98, '🟢 Vert = Réduit l\'attrition\n🔴 Rouge = Augmente l\'attrition',
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))

        plt.tight_layout()
        plt.savefig(f'trained_models/{model_type}_feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Graphique sauvegardé: trained_models/{model_type}_feature_importance.png")
        plt.show()
    else:
        print(f"⚠️  L'importance des features n'est pas disponible pour ce type de modèle")


# ============================================================================
# ANALYSE DES FEATURES PROBLÉMATIQUES
# ============================================================================

def analyze_suspicious_features(model, feature_names, X_test, y_test):
    """
    Identifie les features qui ont un comportement suspect
    """
    importance = model.coef_[0]

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': importance,
        'Abs_Coefficient': np.abs(importance)
    }).sort_values('Abs_Coefficient', ascending=False)

    print("\n🔍 ANALYSE DES FEATURES")
    print("=" * 70)
    print("\nTop 5 features avec le plus d'impact:")
    print(feature_importance_df.head(5)[['Feature', 'Coefficient']])

    print("\n⚠️  Features suspectes (coefficient inattendu):")
    print("   Vérifiez si ces features ont du sens dans votre contexte:")
    print(feature_importance_df.head(10)[['Feature', 'Coefficient']])