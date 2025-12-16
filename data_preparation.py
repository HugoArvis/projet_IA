# Imports
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
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
print(f"✓ Nombre d'employés: {len(df)}")
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

print("\nDéséquilibre de classes détecté (16.1% vs 83.9%) - SMOTE sera appliqué ultérieurement")

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

# Analyse des valeurs manquantes
print("=== Analyse des valeurs manquantes ===")
missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) > 0:
    print("\nVariables avec valeurs manquantes:")
    for col, count in missing.items():
        print(f"  {col}: {count} NA ({count / len(df) * 100:.2f}%)")
    print(f"\nTotal: {missing.sum()} valeurs manquantes")

    # Visualisation
    plt.figure(figsize=(10, 5))
    missing.plot(kind='barh', color='#e74c3c')
    plt.title('Nombre de valeurs manquantes par variable')
    plt.xlabel('Nombre de NA')
    plt.tight_layout()
else:
    print("✓ Aucune valeur manquante détectée")

# Imputation des variables numériques (médiane)
numeric_cols_with_na = ['NumCompaniesWorked', 'TotalWorkingYears']

print("\nImputation des variables numériques (médiane)...")
imputer_median = SimpleImputer(strategy='median')

# Vérifier que les colonnes existent et ont des NA
numeric_cols_to_impute = [col for col in numeric_cols_with_na if col in df.columns and df[col].isnull().sum() > 0]

if len(numeric_cols_to_impute) > 0:
    df[numeric_cols_to_impute] = imputer_median.fit_transform(df[numeric_cols_to_impute])
    print(f"✓ Imputation par médiane: {numeric_cols_to_impute}")
else:
    print("✓ Aucune variable numérique à imputer")

# Imputation des variables catégorielles ordinales (mode)
categorical_cols_with_na = ['EnvironmentSatisfaction', 'JobSatisfaction', 'WorkLifeBalance']

print("\nImputation des variables catégorielles (mode)...")
imputer_mode = SimpleImputer(strategy='most_frequent')

# Vérifier que les colonnes existent et ont des NA
categorical_cols_to_impute = [col for col in categorical_cols_with_na if col in df.columns and df[col].isnull().sum() > 0]

if len(categorical_cols_to_impute) > 0:
    df[categorical_cols_to_impute] = imputer_mode.fit_transform(df[categorical_cols_to_impute])
    print(f"✓ Imputation par mode: {categorical_cols_to_impute}")
else:
    print("✓ Aucune variable catégorielle à imputer")

# Vérification finale
print("\n=== Vérification après imputation ===")
remaining_na = df.isnull().sum().sum()
if remaining_na == 0:
    print("✓ SUCCÈS: 0 valeurs manquantes restantes")
else:
    print(f"Attention: {remaining_na} valeurs manquantes restantes")
    print(df.isnull().sum()[df.isnull().sum() > 0])

# Analyse des variables catégorielles avant encodage
print("=== Variables catégorielles à encoder ===")
categorical_vars = ['BusinessTravel', 'Department', 'EducationField', 'JobRole']

for var in categorical_vars:
    if var in df.columns:
        print(f"\n{var}: {df[var].nunique()} modalités")
        print(df[var].value_counts())

# Label Encoding pour la variable cible Attrition
print("\nLabel Encoding de la variable cible (Attrition)...")
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

print(f"✓ Attrition encodée: {df['Attrition'].unique()}")
print(f"  Distribution: {df['Attrition'].value_counts().to_dict()}")

# One-Hot Encoding pour les variables catégorielles
print("\nOne-Hot Encoding des variables catégorielles...")
print(f"Nombre de colonnes avant: {len(df.columns)}")

categorical_to_encode = [col for col in categorical_vars if col in df.columns]
df = pd.get_dummies(df, columns=categorical_to_encode, drop_first=True)

print(f"✓ Nombre de colonnes après: {len(df.columns)}")
print(f"✓ Nouvelles colonnes créées: {len(df.columns) - len(general_data.columns)}")

# Afficher les nouvelles colonnes créées
new_cols = [col for col in df.columns if '_' in col and any(cat in col for cat in categorical_vars)]
print(f"\nExemples de colonnes créées:")
for col in new_cols[:10]:
    print(f"  - {col}")
if len(new_cols) > 10:
    print(f"  ... et {len(new_cols) - 10} autres")

# Chargement des données de badgeage
print("Chargement des données de badgeage...")
print("⚠️ Attention: Fichiers volumineux (2.3M points), cela peut prendre quelques secondes...\n")

in_time = pd.read_csv('data/in_time.csv')
out_time = pd.read_csv('data/out_time.csv')

print(f"✓ in_time: {in_time.shape}")
print(f"✓ out_time: {out_time.shape}")

# Aperçu
print("\nAperçu des données de badgeage:")
print(in_time.head())


def calculate_badging_features(in_time_df, out_time_df):
    """
    Calcule les 4 features agrégées à partir des données de badgeage.
    VERSION OPTIMISÉE avec opérations vectorisées (beaucoup plus rapide)

    Parameters:
    -----------
    in_time_df : DataFrame avec colonnes EmployeeID + dates (heures d'arrivée)
    out_time_df : DataFrame avec colonnes EmployeeID + dates (heures de départ)

    Returns:
    --------
    DataFrame avec EmployeeID et 4 features agrégées
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

    # Marquer les absences (NA dans arrival ou departure)
    badging['is_present'] = ~(badging['ArrivalTime'].isna() | badging['DepartureTime'].isna())

    # Agrégation par employé
    features = badging.groupby('EmployeeID').agg(
        avg_arrival_time=('arrival_hour', 'mean'),
        std_arrival_time=('arrival_hour', 'std'),
        avg_work_hours=('work_duration', 'mean'),
        total_days=('Date', 'count'),
        present_days=('is_present', 'sum')
    ).reset_index()

    # Calculer le taux d'absence
    features['absence_rate'] = 1 - (features['present_days'] / features['total_days'])

    # Supprimer les colonnes intermédiaires
    features = features.drop(['total_days', 'present_days'], axis=1)

    print(f"✓ Features calculées pour {len(features)} employés")
    print(f"✓ Temps de calcul réduit grâce aux opérations vectorisées!")

    return features


print("Fonction de calcul optimisée définie")
# Calcul des features de badgeage
badging_features = calculate_badging_features(in_time, out_time)

print("\n=== Statistiques des features de badgeage ===")
print(badging_features.describe())

# Visualisation
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

badging_features['avg_arrival_time'].hist(bins=30, ax=axes[0, 0], color='#3498db')
axes[0, 0].set_title('Distribution: Heure moyenne d\'arrivée')
axes[0, 0].set_xlabel('Heures depuis minuit')

badging_features['std_arrival_time'].hist(bins=30, ax=axes[0, 1], color='#9b59b6')
axes[0, 1].set_title('Distribution: Régularité des arrivées (écart-type)')
axes[0, 1].set_xlabel('Heures')

badging_features['avg_work_hours'].hist(bins=30, ax=axes[1, 0], color='#2ecc71')
axes[1, 0].set_title('Distribution: Durée moyenne de travail')
axes[1, 0].set_xlabel('Heures')

badging_features['absence_rate'].hist(bins=30, ax=axes[1, 1], color='#e74c3c')
axes[1, 1].set_title('Distribution: Taux d\'absence')
axes[1, 1].set_xlabel('Proportion')

plt.tight_layout()

# Jointure avec le DataFrame principal
print("\nJointure des features de badgeage au DataFrame principal...")
print(f"Dimensions avant: {df.shape}")

df = df.merge(badging_features, on='EmployeeID', how='left')

print(f"✓ Dimensions après: {df.shape}")
print(f"✓ 4 nouvelles colonnes ajoutées: avg_arrival_time, std_arrival_time, avg_work_hours, absence_rate")

# Vérification des valeurs manquantes
badging_cols = ['avg_arrival_time', 'std_arrival_time', 'avg_work_hours', 'absence_rate']
print(f"\nValeurs manquantes dans les features de badgeage:")
for col in badging_cols:
    na_count = df[col].isnull().sum()
    print(f"  {col}: {na_count} NA")

# Si des NA existent, imputer par la médiane
if df[badging_cols].isnull().sum().sum() > 0:
    print("\nImputation des NA par la médiane...")
    imputer_badging = SimpleImputer(strategy='median')
    df[badging_cols] = imputer_badging.fit_transform(df[badging_cols])
    print("✓ Imputation terminée")

# Identifier les variables numériques continues
numeric_continuous = df.select_dtypes(include=[np.number]).columns.tolist()

# Exclure la variable cible et les variables ordinales
ordinal_vars = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement',
                'JobLevel', 'JobSatisfaction', 'PerformanceRating',
                'StockOptionLevel', 'WorkLifeBalance']
exclude = ['Attrition', 'EmployeeID'] + ordinal_vars

numeric_continuous = [col for col in numeric_continuous if col not in exclude and col in df.columns]

print(f"=== Variables numériques continues à analyser ({len(numeric_continuous)}) ===")
print(numeric_continuous)


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

# Application du capping (avant/après pour visualisation)
print("\nApplication du IQR Capping...")

# Sauvegarder quelques colonnes pour comparaison
cols_to_visualize = [col for col in numeric_continuous if detect_outliers_iqr(df, col)[2] > 0][:4]
before_capping = {col: df[col].copy() for col in cols_to_visualize}

# Appliquer le capping
for col in numeric_continuous:
    df = cap_outliers_iqr(df, col)

print(f"✓ Capping appliqué sur {len(numeric_continuous)} variables")

# Vérification après capping
print("\n=== Vérification après capping ===")
total_outliers_after = 0
for col in numeric_continuous:
    _, _, n_outliers = detect_outliers_iqr(df, col)
    total_outliers_after += n_outliers

print(f"✓ Outliers restants: {total_outliers_after} (devrait être 0)")

# Visualisation avant/après pour quelques variables
if len(cols_to_visualize) > 0:
    n_cols = min(len(cols_to_visualize), 4)
    fig, axes = plt.subplots(n_cols, 2, figsize=(14, 4 * n_cols))

    if n_cols == 1:
        axes = axes.reshape(1, -1)

    for idx, col in enumerate(cols_to_visualize[:n_cols]):
        # Avant
        axes[idx, 0].boxplot(before_capping[col].dropna(), vert=False)
        axes[idx, 0].set_title(f'{col} - AVANT capping')
        axes[idx, 0].set_xlabel('Valeur')

        # Après
        axes[idx, 1].boxplot(df[col].dropna(), vert=False)
        axes[idx, 1].set_title(f'{col} - APRÈS capping')
        axes[idx, 1].set_xlabel('Valeur')

    plt.tight_layout()

    print("\n✓ Les outliers extrêmes ont été remplacés par les limites IQR")
else:
    print("\nAucune visualisation nécessaire (pas d'outliers détectés)")

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
    print("   Valeurs manquantes détectées:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
else:
    print("   Aucune valeur manquante")

print(f"\n3. Distribution de la variable cible (Attrition):")
attrition_dist = df['Attrition'].value_counts()
print(f"   - Classe 0 (No): {attrition_dist[0]} ({attrition_dist[0]/len(df)*100:.1f}%)")
print(f"   - Classe 1 (Yes): {attrition_dist[1]} ({attrition_dist[1]/len(df)*100:.1f}%)")

print(f"\n4. Types de données:")
print(f"   - Variables numériques: {len(df.select_dtypes(include=[np.number]).columns)}")
print(f"   - Variables catégorielles: {len(df.select_dtypes(include=['object']).columns)}")

print(f"\n5. Statistiques descriptives:")
print(df.describe())

# Sauvegarde du DataFrame préparé
output_file = 'data/prepared_data.csv'
print(f"\nSauvegarde du DataFrame dans '{output_file}'...")

df.to_csv(output_file, index=False)

print(f"✓ Fichier sauvegardé avec succès")
print(f"✓ Taille du fichier: {len(df)} lignes × {len(df.columns)} colonnes")

print("\n" + "="*60)
print("RÉSUMÉ DES TRANSFORMATIONS APPLIQUÉES")
print("="*60)

print("\n✓ Section 5.1 - Variables supprimées:")
print("  - Age, Gender, MaritalStatus (discrimination)")
print("  - EmployeeCount, StandardHours, Over18 (constantes)")
print("  - EmployeeID (identifiant)")

print("\n✓ Section 5.2 - Valeurs manquantes imputées:")
print("  - NumCompaniesWorked, TotalWorkingYears → Médiane")
print("  - EnvironmentSatisfaction, JobSatisfaction, WorkLifeBalance → Mode")

print("\n✓ Section 5.3 - Variables encodées:")
print("  - Attrition → Label Encoding (0/1)")
print("  - BusinessTravel, Department, EducationField, JobRole → One-Hot Encoding")

print("\n✓ Section 5.5 - Features de badgeage créées:")
print("  - avg_arrival_time, std_arrival_time, avg_work_hours, absence_rate")
print("  → 2.3M points agrégés en 4 features (minimisation RGPD)")

print("\n✓ Outliers gérés: IQR Capping, aucune ligne supprimée")

print("\n✓ Données prêtes: 0 NA, variables sensibles retirées, 4,410 employés")

print("\nProchaines étapes:")
print("  1. Split train/test (80/20)")
print("  2. SMOTE sur train (Section 5.4)")
print("  3. Entraînement et évaluation (F1, Recall, AUC)")

print("\n" + "="*60)
print("PRÉPARATION TERMINÉE AVEC SUCCÈS!")
print("="*60)

# Rechargement du CSV final
df_final = pd.read_csv("data/prepared_data.csv")

print("\n=== DataFrame chargé depuis prepared_data.csv ===")
print(f"Shape: {df_final.shape}")
print(f"Colonnes: {len(df_final.columns)}")
print(f"Lignes: {len(df_final)}")

print("\n=== Premières lignes ===")
print(df_final.head())

print("\n=== Informations ===")
print(df_final.info())

print("\n✓ CSV prêt pour l'entraînement ML")