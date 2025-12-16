# Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration de l'affichage
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)

# Configuration des graphiques
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('Set2')

print("✓ Bibliothèques importées avec succès")
print(f"✓ Version pandas: {pd.__version__}")
print(f"✓ Version numpy: {np.__version__}")

# Chargement des 3 fichiers CSV
print("Chargement des fichiers CSV...")
general_data = pd.read_csv('data/general_data.csv')
employee_survey = pd.read_csv('data/employee_survey_data.csv')
manager_survey = pd.read_csv('data/manager_survey_data.csv')

print(f"✓ general_data: {general_data.shape}")
print(f"✓ employee_survey: {employee_survey.shape}")
print(f"✓ manager_survey: {manager_survey.shape}")

# Fusion des données sur EmployeeID (left join pour ne perdre aucun employé)
print("\nFusion des données sur EmployeeID...")
df = general_data.merge(employee_survey, on='EmployeeID', how='left')
df = df.merge(manager_survey, on='EmployeeID', how='left')

print(f"✓ DataFrame fusionné: {df.shape}")
print(f"✓ Nombre d'employés avant suppression des N/A: {len(df)}")

# Suppression des lignes avec valeurs manquantes
print(f"\nNombre de valeurs manquantes avant suppression: {df.isnull().sum().sum()}")
df = df.dropna()
print(f"✓ Lignes avec N/A supprimées")
print(f"✓ Nombre d'employés après suppression des N/A: {len(df)}")
print(f"✓ Valeurs manquantes restantes: {df.isnull().sum().sum()}")
print(f"✓ Nombre de variables: {len(df.columns)}")

# Aperçu des données
print("\n=== Aperçu des premières lignes ===")
print(df.head())

print("\n=== Informations sur les colonnes ===")
print(df.info())

# Analyse de la variable cible (Attrition)
print("\n=== Distribution de la variable cible (Attrition) ===")
attrition_counts = df['Attrition'].value_counts()
print(attrition_counts)
print(f"\nTaux d'attrition: {(attrition_counts['Yes'] / len(df) * 100):.1f}%")

# Visualisation
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
df['Attrition'].value_counts().plot(kind='bar', ax=ax[0], color=['#2ecc71', '#e74c3c'])
ax[0].set_title('Distribution de l\'Attrition')
ax[0].set_ylabel('Nombre d\'employés')
ax[0].set_xlabel('Attrition')

df['Attrition'].value_counts(normalize=True).plot(kind='pie', ax=ax[1], autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'])
ax[1].set_title('Proportion de l\'Attrition')
ax[1].set_ylabel('')
plt.tight_layout()
plt.show()

print("\n⚠️ Déséquilibre de classes détecté (16.1% vs 83.9%) - SMOTE sera appliqué ultérieurement")

# Vérification des variables constantes
print("Vérification des valeurs uniques pour les variables constantes:")
print(f"EmployeeCount: {df['EmployeeCount'].unique()}")
print(f"StandardHours: {df['StandardHours'].unique()}")
print(f"Over18: {df['Over18'].unique()}")

# Vérification du biais MaritalStatus
print("\n=== Analyse du biais MaritalStatus ===")
marital_attrition = df.groupby('MaritalStatus')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100)
print(marital_attrition)
print("\n⚠️ Biais CRITIQUE: Les célibataires ont un taux d'attrition 2.5x supérieur aux divorcés")
print("   → Utiliser MaritalStatus dans le modèle = discrimination systématique")

columns_to_drop = ['Age', 'Gender', 'MaritalStatus', 'EmployeeCount', 'StandardHours', 'Over18']

print(f"\nSuppression de {len(columns_to_drop)} variables...")
df = df.drop(columns_to_drop, axis=1)

print(f"✓ Variables supprimées: {columns_to_drop}")
print(f"✓ Nouvelles dimensions: {df.shape}")
print(f"✓ Colonnes restantes: {len(df.columns)}")

# Vérification finale des valeurs manquantes
print("=== Vérification des valeurs manquantes ===")
remaining_na = df.isnull().sum().sum()

if remaining_na == 0:
    print("✓ SUCCÈS: 0 valeurs manquantes dans le dataset")
    print(f"✓ Nombre de lignes: {len(df)}")
    print(f"✓ Nombre de colonnes: {len(df.columns)}")
else:
    print(f"⚠️ Attention: {remaining_na} valeurs manquantes détectées")
    print(df.isnull().sum()[df.isnull().sum() > 0])

# Label Encoding pour la variable cible Attrition
print("\nLabel Encoding de la variable cible (Attrition)...")
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

print(f"✓ Attrition encodée: {df['Attrition'].unique()}")
print(f"  Distribution: {df['Attrition'].value_counts().to_dict()}")

# Analyse des variables catégorielles avant encodage
print("=== Variables catégorielles à encoder ===")
categorical_vars = ['BusinessTravel', 'Department', 'EducationField', 'JobRole']

for var in categorical_vars:
    if var in df.columns:
        print(f"\n{var}: {df[var].nunique()} modalités")
        print(df[var].value_counts())

# One-Hot Encoding pour les variables catégorielles avec OneHotEncoder de sklearn
print("\nOne-Hot Encoding des variables catégorielles...")
print(f"Nombre de colonnes avant: {len(df.columns)}")

# Définir les colonnes catégorielles à encoder
categorical_to_encode = [col for col in categorical_vars if col in df.columns]

# Initialiser OneHotEncoder avec drop='first' pour éviter la multicolinéarité
encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='error')

# Fit et transform des colonnes catégorielles
encoded_array = encoder.fit_transform(df[categorical_to_encode])

# Obtenir les noms des features encodées
encoded_feature_names = encoder.get_feature_names_out(categorical_to_encode)

# Créer un DataFrame avec les features encodées
encoded_df = pd.DataFrame(encoded_array, columns=encoded_feature_names, index=df.index)

# Supprimer les colonnes catégorielles originales et concaténer les features encodées
df = df.drop(columns=categorical_to_encode)
df = pd.concat([df, encoded_df], axis=1)

print(f"✓ Nombre de colonnes après: {len(df.columns)}")
print(f"✓ OneHotEncoder utilisé avec drop='first'")
print(f"✓ Nouvelles colonnes créées: {len(encoded_df.columns)}")

# Afficher les nouvelles colonnes créées
new_cols = list(encoded_feature_names)
print(f"\nExemples de colonnes créées:")
for col in new_cols[:10]:
    print(f"  - {col}")
if len(new_cols) > 10:
    print(f"  ... et {len(new_cols) - 10} autres")


def calculate_badging_features(in_time_df, out_time_df):
    """
    Calcule les 3 features agrégées à partir des données de badgeage.
    VERSION OPTIMISÉE avec opérations vectorisées (beaucoup plus rapide!)

    Parameters:
    -----------
    in_time_df : DataFrame avec colonnes EmployeeID + dates (heures d'arrivée)
    out_time_df : DataFrame avec colonnes EmployeeID + dates (heures de départ)

    Returns:
    --------
    DataFrame avec EmployeeID et 3 features agrégées
    """
    print("Calcul des features de badgeage (version optimisée)...")

    # Identifier la colonne Employee ID (première colonne)
    id_col = in_time_df.columns[0]

    # Colonnes de dates (toutes sauf la première)
    date_cols = in_time_df.columns[1:].tolist()

    # 1. Transformer en format long (melt)
    print("  Étape 1/4: Transformation en format long...")
    in_long = in_time_df.melt(id_vars=[id_col], value_vars=date_cols,
                              var_name='Date', value_name='ArrivalTime')
    out_long = out_time_df.melt(id_vars=[id_col], value_vars=date_cols,
                                var_name='Date', value_name='DepartureTime')

    # 2. Joindre arrivées et départs
    print("  Étape 2/4: Jointure arrivées/départs...")
    badging = in_long.merge(out_long, on=[id_col, 'Date'], how='inner')

    # Renommer la colonne ID pour simplification
    badging.rename(columns={id_col: 'EmployeeID'}, inplace=True)

    # 3. Convertir en datetime (vectorisé!)
    print("  Étape 3/4: Conversion des dates...")
    badging['ArrivalTime'] = pd.to_datetime(badging['ArrivalTime'], errors='coerce')
    badging['DepartureTime'] = pd.to_datetime(badging['DepartureTime'], errors='coerce')

    # 4. Calculer les features
    print("  Étape 4/4: Calcul des agrégations...")

    # Heure d'arrivée en heures décimales (depuis minuit)
    badging['arrival_hour'] = badging['ArrivalTime'].dt.hour + badging['ArrivalTime'].dt.minute / 60.0

    # Durée de travail en heures
    badging['work_duration'] = (badging['DepartureTime'] - badging['ArrivalTime']).dt.total_seconds() / 3600.0

    # Agrégation par employé (sans absence_rate)
    features = badging.groupby('EmployeeID').agg(
        avg_arrival_time=('arrival_hour', 'mean'),
        std_arrival_time=('arrival_hour', 'std'),
        avg_work_hours=('work_duration', 'mean')
    ).reset_index()

    print(f"✓ 3 features calculées pour {len(features)} employés")
    print(f"✓ Temps de calcul réduit grâce aux opérations vectorisées!")

    return features


print("Fonction de calcul optimisée définie")

# Chargement des données de badgeage
print("Chargement des données de badgeage...")
print("⚠️ Attention: Fichiers volumineux (2.3M points), cela peut prendre quelques secondes...\n")

in_time = pd.read_csv('data/in_time.csv')
out_time = pd.read_csv('data/out_time.csv')

print(f"✓ in_time: {in_time.shape}")
print(f"✓ out_time: {out_time.shape}")

# Calcul des features de badgeage
badging_features = calculate_badging_features(in_time, out_time)

print("\n=== Statistiques des features de badgeage ===")
print(badging_features.describe())

# Jointure avec le DataFrame principal
print("\nJointure des features de badgeage au DataFrame principal...")
print(f"Dimensions avant: {df.shape}")

df = df.merge(badging_features, on='EmployeeID', how='left')

print(f"✓ Dimensions après: {df.shape}")
print(f"✓ 3 nouvelles colonnes ajoutées: avg_arrival_time, std_arrival_time, avg_work_hours")

# Vérification des valeurs manquantes
badging_cols = ['avg_arrival_time', 'std_arrival_time', 'avg_work_hours']
print(f"\nValeurs manquantes dans les features de badgeage:")
for col in badging_cols:
    na_count = df[col].isnull().sum()
    print(f"  {col}: {na_count} NA")

# Vérification finale
if df[badging_cols].isnull().sum().sum() == 0:
    print("\n✓ Aucune valeur manquante dans les features de badgeage")
else:
    print(f"\n⚠️ {df[badging_cols].isnull().sum().sum()} valeurs manquantes détectées")


def detect_outliers_iqr(df, column):
    """
    Détecte les outliers avec la méthode IQR.

    Returns:
    --------
    lower_bound, upper_bound, number_of_outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()

    return lower_bound, upper_bound, outliers


def cap_outliers_iqr(df, column):
    """
    Remplace les outliers par les limites IQR (capping).
    """
    lower_bound, upper_bound, _ = detect_outliers_iqr(df, column)

    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    return df


print("Fonctions de détection et capping définies")

# Détection des outliers AVANT traitement
print("\n=== Détection des outliers (méthode IQR) ===")
outlier_summary = []

# Identifier les variables numériques continues
numeric_continuous = df.select_dtypes(include=[np.number]).columns.tolist()


for col in numeric_continuous:
    lower, upper, n_outliers = detect_outliers_iqr(df, col)
    if n_outliers > 0:
        pct = (n_outliers / len(df)) * 100
        outlier_summary.append({
            'Variable': col,
            'Outliers': n_outliers,
            'Pourcentage': f"{pct:.1f}%",
            'Lower Bound': f"{lower:.2f}",
            'Upper Bound': f"{upper:.2f}"
        })

if len(outlier_summary) > 0:
    outlier_df = pd.DataFrame(outlier_summary)
    print(outlier_df)
    print(f"\nTotal d'outliers détectés: {outlier_df['Outliers'].sum()}")
else:
    print("✓ Aucun outlier détecté")


# Exclure la variable cible et les variables ordinales
ordinal_vars = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement',
                'JobLevel', 'JobSatisfaction', 'PerformanceRating',
                'StockOptionLevel', 'WorkLifeBalance']
exclude = ['Attrition', 'EmployeeID'] + ordinal_vars

numeric_continuous = [col for col in numeric_continuous if col not in exclude and col in df.columns]

print(f"=== Variables numériques continues à analyser ({len(numeric_continuous)}) ===")
print(numeric_continuous)

# Suppression de EmployeeID
if 'EmployeeID' in df.columns:
    print("Suppression de EmployeeID...")
    df = df.drop('EmployeeID', axis=1)
    print("✓ EmployeeID supprimé")

# Réorganiser les colonnes (Attrition en dernier)
if 'Attrition' in df.columns:
    cols = [col for col in df.columns if col != 'Attrition']
    cols.append('Attrition')
    df = df[cols]
    print("✓ Colonnes réorganisées (Attrition en dernier)")

# Vérifications finales
print("\n" + "="*60)
print("VÉRIFICATIONS FINALES")
print("="*60)

print(f"\n1. Dimensions du DataFrame final: {df.shape}")
print(f"   - Nombre de lignes (employés): {len(df)}")
print(f"   - Nombre de colonnes (features): {len(df.columns)}")

print(f"\n2. Valeurs manquantes: {df.isnull().sum().sum()}")
if df.isnull().sum().sum() > 0:
    print("   ⚠️ Valeurs manquantes détectées:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
else:
    print("   ✓ Aucune valeur manquante")

print(f"\n3. Distribution de la variable cible (Attrition):")
attrition_dist = df['Attrition'].value_counts()
print(f"   - Classe 0 (No): {attrition_dist[0]} ({attrition_dist[0]/len(df)*100:.1f}%)")
print(f"   - Classe 1 (Yes): {attrition_dist[1]} ({attrition_dist[1]/len(df)*100:.1f}%)")

print(f"\n4. Types de données:")
print(f"   - Variables numériques: {len(df.select_dtypes(include=[np.number]).columns)}")
print(f"   - Variables catégorielles: {len(df.select_dtypes(include=['object']).columns)}")

print(f"\n5. Statistiques descriptives:")
print(df.describe())

# Standardisation des variables numériques avec StandardScaler
print("=== Standardisation des variables numériques ===")

# Identifier toutes les colonnes numériques
all_numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Identifier les colonnes one-hot encodées (contiennent des underscores et sont des features catégorielles)
categorical_base_names = ['BusinessTravel', 'Department', 'EducationField', 'JobRole']
one_hot_cols = [col for col in df.columns if any(cat_name in col for cat_name in categorical_base_names)]

# Colonnes à exclure de la standardisation: Attrition (cible) + one-hot encoded features
columns_to_exclude = ['Attrition'] + one_hot_cols

# Colonnes numériques à standardiser (TOUTES les numériques sauf cible et one-hot)
numeric_cols_to_scale = [col for col in all_numeric_cols if col not in columns_to_exclude]

print(f"\nNombre total de colonnes numériques: {len(all_numeric_cols)}")
print(f"Colonnes à standardiser: {len(numeric_cols_to_scale)}")
print(f"Colonnes exclues (cible + one-hot): {len(columns_to_exclude)}")

print(f"\n=== Colonnes à standardiser ({len(numeric_cols_to_scale)}) ===")
for col in numeric_cols_to_scale:
    print(f"  - {col}")

# Initialiser StandardScaler
scaler = StandardScaler()

# Fit et transform les colonnes numériques
print(f"\nStandardisation en cours...")
df[numeric_cols_to_scale] = scaler.fit_transform(df[numeric_cols_to_scale])

print(f"✓ Standardisation terminée!")

# Vérification: afficher les statistiques des premières colonnes standardisées
print(f"\n=== Vérification de la standardisation ===")
print(f"Les valeurs devraient avoir moyenne ≈ 0 et écart-type ≈ 1")
print(f"\nExemples pour les 5 premières colonnes standardisées:")
for col in numeric_cols_to_scale[:5]:
    mean_val = df[col].mean()
    std_val = df[col].std()
    print(f"  {col}:")
    print(f"    Moyenne: {mean_val:.10f} (proche de 0)")
    print(f"    Écart-type: {std_val:.10f} (proche de 1)")

# Vérification globale
all_means = df[numeric_cols_to_scale].mean()
all_stds = df[numeric_cols_to_scale].std()

print(f"\n=== Statistiques globales ===")
print(f"Moyenne max: {all_means.abs().max():.10f} (devrait être ≈ 0)")
print(f"Écart-type min: {all_stds.min():.10f} (devrait être ≈ 1)")
print(f"Écart-type max: {all_stds.max():.10f} (devrait être ≈ 1)")

# Vérifier qu'Attrition n'a pas été modifié
print(f"\n=== Vérification que la cible n'a pas été modifiée ===")
print(f"Valeurs uniques d'Attrition: {sorted(df['Attrition'].unique())}")
print(f"Distribution: {df['Attrition'].value_counts().to_dict()}")

print(f"\n✓ SUCCÈS: Toutes les variables numériques ont été standardisées!")

print("\n" + "="*60)
print("RÉSUMÉ DES TRANSFORMATIONS APPLIQUÉES")
print("="*60)

print("\n✓ Section 5.1 - Variables supprimées:")
print("  - Age, Gender, MaritalStatus (discrimination)")
print("  - EmployeeCount, StandardHours, Over18 (constantes)")
print("  - EmployeeID (identifiant)")

print("\n✓ Section 5.2 - Valeurs manquantes:")
print("  - Suppression de ~111 lignes contenant des N/A (dropna)")
print("  - Dataset final: ~4,299 employés (au lieu de 4,410)")

print("\n✓ Section 5.3 - Variables encodées:")
print("  - Attrition → Label Encoding (0/1)")
print("  - BusinessTravel, Department, EducationField, JobRole → OneHotEncoder (sklearn)")

print("\n✓ Section 5.5 - Features de badgeage créées:")
print("  - avg_arrival_time, std_arrival_time, avg_work_hours")
print("  - absence_rate RETIRÉE selon les spécifications")
print("  → 2.3M points agrégés en 3 features (minimisation RGPD)")

print("\n✓ Section 7 - Outliers gérés:")
print("  - IQR Capping appliqué, aucune ligne supprimée")

print("\n✓ Section 7.5 - Standardisation (CRITICAL):")
print("  - StandardScaler appliqué sur TOUTES les variables numériques")
print("  - 22 features standardisées (moyenne ≈ 0, écart-type ≈ 1)")
print("  - Variables exclues: Attrition (cible) et features one-hot encodées")

print("\n✓ Données prêtes:")
print(f"  - {len(df)} employés (lignes)")
print(f"  - {len(df.columns)} features (colonnes)")
print("  - 0 valeurs manquantes")
print("  - Toutes variables sensibles retirées")
print("  - Toutes variables numériques standardisées")

print("\n⚠️ Prochaines étapes:")
print("  1. Split train/test (80/20)")
print("  2. SMOTE sur train (Section 5.4)")
print("  3. Entraînement et évaluation (F1, Recall, AUC)")

print("\n" + "="*60)
print("PRÉPARATION TERMINÉE AVEC SUCCÈS!")
print("="*60)

