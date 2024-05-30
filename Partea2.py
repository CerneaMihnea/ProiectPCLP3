import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#Cerinta 1

#Citim datele fisierului

date = pd.read_csv('train.csv')

#Functie pentru eliminarea outlier

def elimina(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    inferior = Q1 - 1.5 * IQR
    superior = Q3 + 1.5 * IQR
    return data[(data[col] >= inferior) & (data[col] <= superior)]

# Aplicam functia pe setul de date
date_noi = elimina(date, 'Age')

# Verificarea dimensiunii
print(f"Dimensiunea initiala: {date.shape}")
print(f"Dimensiunea modificata: {date_noi.shape}")

#Salvam setului de date nou creat
date_noi.to_csv('vapor_IQR.csv')


#Cerinta 2

#Citim valoarea pentru z
z = int(input("Introduceti valoarea pentru z:"))

def elimina_z(data, col, z):
    varsta_medie = date['Age'].mean()  #calculeaza varsta medie a pasagerilor de pe titanic
    varsta_deviatie = date['Age'].std()  #calculeaza deviatie medie de varsta
    scor_pasageri = (date['Age'] - varsta_medie) / varsta_deviatie  #calculam scorul_pasagerilor
    data_new = data[scor_pasageri <= z]  #eliminam pasagerii care au varste dubioase
    return data_new
4

# Aplicam functia
fara_outlieri = elimina_z(date, 'Age', z)

# Afisam dimensiunea seturilor de date
print("Dimensiunea initiala:", date.shape)
print("Dimensiunea noua:", fara_outlieri.shape)

fara_outlieri.to_csv('vapor_z.csv')

#Cerinta 3
figura, axe = plt.subplots(1, 2, figsize=(12, 5))

# Subgraficul 1
sns.histplot(date['Age'], bins=10, kde=True, color='green', ax=axe[0])
axe[0].set_title('Distributia varstelor initial')
axe[0].set_xlabel('Varsta')
axe[0].set_ylabel('Frecventa')

# Subgraficul 2
sns.histplot(fara_outlieri['Age'], bins=10, kde=True, color='purple', ax=axe[1])
axe[1].set_title('Distributia varstelor dupa eliminarea outlier-ilor')
axe[1].set_xlabel('Varsta')
axe[1].set_ylabel('Frecventt')

# Afisam graficul
plt.tight_layout()
plt.show()

#Cerinta 4

final_data = elimina(fara_outlieri, 'Age')
#Protocolul de testare
pasager_antrenare, pasager_valid, stare_antrenare, stare_valid = train_test_split(final_data.drop(columns=['Survived']), final_data['Survived'], test_size=0.2)

#Preprocesarea datelor
def umple(data):
    medie_varsta = data['Age'].mean()
    medie_fare = data['Fare'].mean()

    data['Age'].fillna(medie_varsta, inplace=True)
    data['Fare'].fillna(medie_fare, inplace=True)

def encode(data):
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

def standardizeaza(data):
    medie_varsta = data['Age'].mean()
    varsta_std = data['Age'].std()
    data['Age'] = (data['Age'] - medie_varsta) / varsta_std

    medie_fare = data['Fare'].mean()
    fare_std = data['Fare'].std()
    data['Fare'] = (data['Fare'] - medie_fare) / fare_std

umple(pasager_antrenare)
umple(pasager_valid)

encode(pasager_antrenare)
encode(pasager_valid)

standardizeaza(pasager_antrenare)
standardizeaza(pasager_valid)

pasager_antrenare= pasager_antrenare.drop(columns=['Name', 'Cabin', 'Embarked', 'Ticket'])
pasager_valid = pasager_valid.drop(columns=['Name', 'Cabin', 'Embarked', 'Ticket'])

# 3. Antrenarea modelului
clf = RandomForestClassifier()
clf.fit(pasager_antrenare, stare_antrenare)

# 4. Evaluarea modelului
predictie = clf.predict(pasager_valid)
acuratete = accuracy_score(stare_valid, predictie)
print("Acuratetea implementarii este de: {:.2f}%".format(acuratete * 100))