
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', None)
sns.set(style='whitegrid')


df = pd.read_csv("vehicles.csv")


print("Dimensiunea datasetului:", df.shape)
print("\nPrimele 5 inregistrari:")
print(df.head())


print("\nInformatii despre tipurile de date:")
df.info()

print("\nStatistici descriptive:")
print(df.describe(include='all'))




cols_to_drop = ['id', 'url', 'region_url', 'VIN', 'image_url', 'description', 'county']
df = df.drop(columns=cols_to_drop, errors='ignore')


df = df.dropna(subset=['price', 'year', 'manufacturer', 'model'])


df = df[(df['price'] > 1000) & (df['price'] < 200000)]


plt.figure(figsize=(8,5))
sns.histplot(df['price'], bins=50, kde=True)
plt.title('Distributia preturilor masinilor')
plt.xlabel('Pret ($)')
plt.ylabel('Frecventa')
plt.show()




plt.figure(figsize=(10,6))
sns.countplot(y='manufacturer', data=df, order=df['manufacturer'].value_counts().index[:10])
plt.title('Top 10 producatori de masini')
plt.xlabel('Numar anunturi')
plt.ylabel('Producator')
plt.show()


plt.figure(figsize=(7,5))
sns.countplot(x='fuel', data=df)
plt.title('Distributia tipurilor de combustibil')
plt.show()




plt.figure(figsize=(7,5))
sns.boxplot(x='fuel', y='price', data=df)
plt.title('Pret în functie de combustibil')
plt.show()


plt.figure(figsize=(7,5))
sns.boxplot(x='transmission', y='price', data=df)
plt.title('Pret în functie de transmisie')
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='condition', y='price', data=df)
plt.title('Pret in functie de starea masinii')
plt.show()


corr = df[['price', 'year', 'odometer']].corr()
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Corelatii intre variabile numerice')
plt.show()




df['fuel'] = df['fuel'].fillna('unknown')
df['transmission'] = df['transmission'].fillna('unknown')
df['drive'] = df['drive'].fillna('unknown')
df['type'] = df['type'].fillna('unknown')
df['paint_color'] = df['paint_color'].fillna('unknown')


df_encoded = pd.get_dummies(df,
                            columns=['manufacturer', 'fuel', 'transmission', 'drive', 'type', 'paint_color', 'state'],
                            drop_first=True)

print("Dimensiunea dupa codificare:", df_encoded.shape)


