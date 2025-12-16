import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def analyze_data(filepath='data/prepared_data.csv'):
    """
    Effectue une analyse exploratoire complète des données
    """
    print("=" * 70)
    print("ANALYSE EXPLORATOIRE DES DONNÉES")
    print("=" * 70)

    # Charger les données
    data = pd.read_csv(filepath)

    # Informations générales
    print(f"\n📊 Forme du dataset: {data.shape}")
    print(f"   - Nombre de lignes: {data.shape[0]}")
    print(f"   - Nombre de colonnes: {data.shape[1]}")

    # Distribution de la target
    print("\n🎯 Distribution de la variable cible (Attrition):")
    attrition_counts = data['Attrition'].value_counts()
    for label, count in attrition_counts.items():
        percentage = count / len(data) * 100
        label_name = "Pas d'attrition" if label == 0 else "Attrition"
        print(f"   - {label_name}: {count} ({percentage:.1f}%)")

    # Ratio de déséquilibre
    ratio = attrition_counts[0] / attrition_counts[1]
    print(f"   - Ratio de déséquilibre: {ratio:.2f}:1")

    if ratio > 3:
        print("   ⚠️  ATTENTION: Déséquilibre important détecté!")
        print("   → Recommandation: Utiliser SMOTE/SMOTEENN + class_weight='balanced'")

    # Statistiques descriptives
    print("\n📈 Statistiques descriptives (features numériques):")
    X = data.drop('Attrition', axis=1)
    print(X.describe().T[['mean', 'std', 'min', 'max']].head(10))

    # Valeurs manquantes
    print("\n🔍 Valeurs manquantes:")
    missing = data.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("   ✓ Aucune valeur manquante")

    # Corrélations avec la target
    print("\n🔗 Corrélations avec Attrition (Top 10):")
    correlations = X.corrwith(data['Attrition']).abs().sort_values(ascending=False)
    print(correlations.head(10))

    # Features potentiellement problématiques
    print("\n⚠️  Features avec variance très faible:")
    low_variance = X.var().sort_values().head(5)
    print(low_variance)

    # Identifier les features binaires
    print("\n🔢 Features binaires détectées:")
    binary_features = [col for col in X.columns if X[col].nunique() == 2]
    print(f"   {len(binary_features)} features binaires trouvées")
    print(f"   Exemples: {binary_features[:5]}")

    return data, correlations


def plot_top_correlations(data, n=10):
    """
    Visualise les features les plus corrélées avec Attrition
    """
    X = data.drop('Attrition', axis=1)
    y = data['Attrition']

    # Calculer corrélations
    correlations = X.corrwith(y).sort_values(ascending=False)
    top_positive = correlations.head(n // 2)
    top_negative = correlations.tail(n // 2)
    top_features = pd.concat([top_positive, top_negative])

    # Plot
    plt.figure(figsize=(10, 8))
    colors = ['green' if x > 0 else 'red' for x in top_features.values]
    plt.barh(range(len(top_features)), top_features.values, color=colors, alpha=0.7)
    plt.yticks(range(len(top_features)), top_features.index)
    plt.xlabel('Corrélation avec Attrition', fontsize=12)
    plt.title(f'Top {n} Features corrélées avec l\'Attrition', fontsize=14, fontweight='bold')
    plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
    plt.tight_layout()
    plt.savefig('trained_models/correlation_analysis.png', dpi=300, bbox_inches='tight')
    print("\n📊 Graphique sauvegardé: trained_models/correlation_analysis.png")
    plt.show()


def check_data_quality(data):
    """
    Vérifie la qualité des données et identifie les problèmes potentiels
    """
    print("\n" + "=" * 70)
    print("VÉRIFICATION DE LA QUALITÉ DES DONNÉES")
    print("=" * 70)

    X = data.drop('Attrition', axis=1)

    issues = []

    # 1. Colonnes constantes
    constant_cols = [col for col in X.columns if X[col].nunique() == 1]
    if constant_cols:
        issues.append(f"⚠️  {len(constant_cols)} colonnes constantes trouvées: {constant_cols}")

    # 2. Duplicatas
    duplicates = data.duplicated().sum()
    if duplicates > 0:
        issues.append(f"⚠️  {duplicates} lignes dupliquées trouvées")

    # 3. Valeurs infinies
    inf_cols = [col for col in X.columns if np.isinf(X[col]).any()]
    if inf_cols:
        issues.append(f"⚠️  Valeurs infinies dans: {inf_cols}")

    # 4. Échelles très différentes (important pour SVM)
    scales = X.std()
    if scales.max() / scales.min() > 100:
        issues.append(f"⚠️  Échelles très différentes détectées (ratio: {scales.max() / scales.min():.0f}:1)")
        issues.append("   → Recommandation: UTILISER StandardScaler (OBLIGATOIRE pour SVM)")

    # Afficher les résultats
    if issues:
        print("\n❌ PROBLÈMES DÉTECTÉS:\n")
        for issue in issues:
            print(issue)
    else:
        print("\n✅ Aucun problème majeur détecté")

    return issues


def generate_data_report(filepath='data/prepared_data.csv'):
    """
    Génère un rapport complet sur les données
    """
    data, correlations = analyze_data(filepath)
    check_data_quality(data)
    plot_top_correlations(data, n=10)

    print("\n" + "=" * 70)
    print("RECOMMANDATIONS POUR VOS MODÈLES")
    print("=" * 70)
    print("""
    1. 📊 PRÉTRAITEMENT:
       ✓ Utiliser StandardScaler (OBLIGATOIRE pour SVM)
       ✓ Les données sont déjà encodées (one-hot encoding)

    2. ⚖️  GESTION DU DÉSÉQUILIBRE:
       ✓ Utiliser SMOTEENN ou SMOTE
       ✓ Ajouter class_weight='balanced' dans les modèles

    3. 🚀 OPTIMISATION SVM:
       ✓ Utiliser kernel='linear' (beaucoup plus rapide)
       ✓ Normaliser les données AVANT
       ✓ Réduire C si trop lent (ex: C=0.1)

    4. 📈 RÉGRESSION LOGISTIQUE:
       ✓ Augmenter max_iter à 2000 minimum
       ✓ Utiliser solver='lbfgs' et n_jobs=-1
       ✓ Interpréter les coefficients avec précaution
    """)


if __name__ == "__main__":
    generate_data_report()
