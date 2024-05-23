import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Cerinta 1

# Citirea datelor din train.csv
df = pd.read_csv('./data/train.csv')

# Determinarea numarului de coloane si tipurile de date
print(f"Numar de coloane: {df.shape[1]}")
print("Tipuri de date:")
print(df.dtypes)

# Numarul de valori lipsa pentru fiecare coloana
print("Valori lipsa per coloana:")
print(df.isnull().sum())

# Numarul de linii
print(f"Numar de linii: {df.shape[0]}")

# Verificarea liniilor duplicate
print(f"Numar de linii duplicate: {df.duplicated().sum()}")


# Cerinta 2

# Procentul persoanelor care au supravietuit si celor care nu au supravietuit
survived_percentage = df['Survived'].value_counts(normalize=True) * 100
print(survived_percentage)

# Procentul pasagerilor pentru fiecare tip de clasa
pclass_percentage = df['Pclass'].value_counts(normalize=True) * 100
print(pclass_percentage)

# Procentul barbatilor si femeilor
sex_percentage = df['Sex'].value_counts(normalize=True) * 100
print(sex_percentage)

# Grafic pentru supravietuire
plt.figure(figsize=(8, 6))
sns.barplot(x=survived_percentage.index, y=survived_percentage.values)
plt.title('Procentul de Supravietuire')
plt.xlabel('Supravietuire')
plt.ylabel('Procent')

plt.savefig('./figures/survival_percentage.png')
plt.close()

# Grafic pentru clasa
plt.figure(figsize=(8, 6))
sns.barplot(x=pclass_percentage.index, y=pclass_percentage.values)
plt.title('Procentul de Pasageri pe Clasa')
plt.xlabel('Clasa')
plt.ylabel('Procent')

plt.savefig('./figures/PClass_percentage.png')
plt.close()

# Grafic pentru sex
plt.figure(figsize=(8, 6))
sns.barplot(x=sex_percentage.index, y=sex_percentage.values)
plt.title('Procentul de Barbati si Femei')
plt.xlabel('Sex')
plt.ylabel('Procent')

plt.savefig('./figures/Sex_percentage.png')
plt.close()


# Cerinta 3

# Generarea histogramelor pentru fiecare coloana numerica
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

df[numeric_columns].hist(bins=15, figsize=(15, 10), layout=(3, 3))
plt.tight_layout()
plt.savefig('./figures/Histograme.png')
plt.close()

# Cerinta 4

# Coloanele cu valori lipsa
missing_values = df.isnull().sum()
columns_with_na = missing_values[missing_values > 0]
print(columns_with_na)

# Procentul valorilor lipsa pentru fiecare dintre cele doua clase
for column in columns_with_na.index:
    print(f"\nProcentul valorilor lipsa pentru {column} per clasa de supravietuire:")
    print(df[df['Survived'] == 0][column].isnull().mean() * 100)
    print(df[df['Survived'] == 1][column].isnull().mean() * 100)

# Cerinta 5 si 6

# Definirea categoriilor de varsta

max_age = df['Age'].max()
 
df['Age_Category'] = pd.cut(df['Age'], bins=[0, 20, 40, 60, max_age], labels=['[0, 20]', '[21, 40]', '[41, 60]', f'[61, {int(max_age)}]'], right=False)

# Numarul de pasageri pentru fiecare categorie de varsta
age_category_counts = df['Age_Category'].value_counts()
print(age_category_counts)

# Grafic pentru numarul de pasageri pe categorii de varsta
sns.countplot(x='Age_Category', data=df)
plt.title('Numarul de Pasageri pe Categorii de Varsta')
plt.xlabel('Categorie de Varsta')
plt.ylabel('Numar de Pasageri')
plt.savefig('./figures/Analiza_categorii_de_varsta.png')
plt.close()


# Numarul de barbati supravietuitori pentru fiecare categorie de varsta
male_survival_by_age = df[df['Sex'] == 'male'].groupby('Age_Category', observed=True)['Survived'].mean()
print(male_survival_by_age)

# Grafic pentru rata de supravietuire a barbatilor pe categorii de varsta
sns.barplot(x=male_survival_by_age.index, y=male_survival_by_age.values)
plt.title('Rata de Supravietuire a Barbatiilor pe Categorii de Varsta')
plt.xlabel('Categorie de Varsta')
plt.ylabel('Rata de Supravietuire')
plt.savefig('./figures/Analiza_barbati_pe_categorii_de_varsta.png')
plt.close()

# Cerinta 7

# Definirea copiilor ca fiind persoanele cu varsta < 18 ani
df['IsChild'] = df['Age'] < 18

# Procentul copiilor la bord
child_percentage = df['IsChild'].mean() * 100
print(f"Procentul copiilor la bord: {child_percentage:.2f}%")

# Rata de supravietuire pentru copii si adulti
child_survival_rate = df[df['IsChild'] == True]['Survived'].mean()
adult_survival_rate = df[df['IsChild'] == False]['Survived'].mean()

# Grafic pentru rata de supravietuire a copiilor si adultilor
survival_rates = [child_survival_rate, adult_survival_rate]
labels = ['Copii', 'Adulti']
sns.barplot(x=labels, y=survival_rates)
plt.title('Rata de Supravietuire pentru Copii si Adulti')
plt.xlabel('Categorie')
plt.ylabel('Rata de Supravietuire')
plt.savefig('./figures/Analiza_rata_de_supravietuire_copii_adulti.png')
plt.close()

# Cerinta 8

# Functie pentru completarea valorilor lipsa numerice
def fill_missing_numerical(df, column, group_column):
    df[column] = df.groupby(group_column)[column].transform(lambda x: x.fillna(x.mean()))
    return df

# Functie pentru completarea valorilor lipsa categoriale
def fill_missing_categorical(df, column, group_column):
    df[column] = df.groupby(group_column)[column].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else np.nan))
    return df

# Completarea valorilor lipsa pentru varsta (Age) pe baza mediei starea de supravietuire (Survived)
df = fill_missing_numerical(df, 'Age', 'Survived')

# Completarea valorilor lipsa pentru embarcare (Embarked) pe baza celei mai frecvente valori din fiecare clasa (Pclass)
df = fill_missing_categorical(df, 'Embarked', 'Pclass')

# Completarea valorilor lipsa pentru cabina (Cabin) pe baza celei mai frecvente valori din fiecare clasa (Pclass)
df = fill_missing_categorical(df, 'Cabin', 'Pclass')

# Salvarea DataFrame-ului completat
df.to_csv('./data/train_filled.csv', index=False)

# Cerinta 9

# Extragem titlurile de nobilime din coloana 'Name'
df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Verificam daca titlurile corespund sexului persoanei respective
title_gender_mapping = {
    'Mr': 'male',
    'Miss': 'female',
    'Mrs': 'female',
    'Master': 'male',
    'Dr': 'unknown',  # Vom considera Dr. ca fiind necunoscut pentru sex
    'Rev': 'unknown',
    'Col': 'unknown',
    'Major': 'unknown',
    'Mlle': 'female',
    'Countess': 'female',
    'Ms': 'female',
    'Lady': 'female',
    'Jonkheer': 'male',
    'Don': 'male',
    'Mme': 'female',
    'Capt': 'unknown',
    'Sir': 'male'
}

df['Gender'] = df['Title'].map(title_gender_mapping)

# Calculam numarul de persoane pentru fiecare titlu
title_counts = df['Title'].value_counts()

# Reprezentam grafic numarul de persoane pentru fiecare titlu
plt.figure(figsize=(10, 6))
sns.countplot(x='Title', data=df, order=title_counts.index)
plt.title('Numarul de persoane pentru fiecare titlu')
plt.xlabel('Titlu')
plt.ylabel('Numar de persoane')
plt.xticks(rotation=45)  # Rotim etichetele pe axa x pentru o mai buna vizibilitate
#plt.tight_layout()  # Pentru a evita suprapunerile
plt.savefig('./figures/Analiza_titluri_sex.png')
plt.close()

# Cerinta 10

# Crearea unei coloane pentru a indica daca un pasager este singur
df['IsAlone'] = (df['SibSp'] == 0) & (df['Parch'] == 0)

# Histograma pentru a vizualiza influenta faptului de a fi singur asupra supravietuirii
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='IsAlone', hue='Survived', multiple='stack', shrink=0.8)
plt.title('Influenta faptului de a fi singur asupra supravietuirii')
plt.xlabel('Este singur')
plt.ylabel('Numarul de pasageri')
plt.xticks([0, 1], ['Nu', 'Da'])
plt.savefig('./figures/Analiza_supravietuire_si_singuratate.png')
plt.close()

# Vizualizarea relatiei dintre tarif, clasa si supravietuire pentru primele 100 de inregistrari
first_100 = df.head(100)

plt.figure(figsize=(12, 8))
sns.stripplot(data=first_100, x='Pclass', y='Fare', hue='Survived', jitter=True, dodge=True)
plt.title('Relatia dintre tarif, clasa si supravietuire pentru primele 100 de inregistrari')
plt.xlabel('Clasa')
plt.ylabel('Tarif')
plt.savefig('./figures/Analiza_relatie_tarif_supravietuire.png')
plt.close()
