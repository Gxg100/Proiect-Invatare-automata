
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', None)
sns.set(style='whitegrid')


df = pd.read_csv("vehicles.csv")


print("Dimensiunea datasetului:", df.shape)
print("\nPrimele 5 înregistrări:")
print(df.head())


print("\nInformații despre tipurile de date:")
df.info()

print("\nStatistici descriptive:")
print(df.describe(include='all'))




cols_to_drop = ['id', 'url', 'region_url', 'VIN', 'image_url', 'description', 'county']
df = df.drop(columns=cols_to_drop, errors='ignore')


df = df.dropna(subset=['price', 'year', 'manufacturer', 'model'])


df = df[(df['price'] > 1000) & (df['price'] < 200000)]


plt.figure(figsize=(8,5))
sns.histplot(df['price'], bins=50, kde=True)
plt.title('Distribuția prețurilor mașinilor')
plt.xlabel('Preț ($)')
plt.ylabel('Frecvență')
plt.show()




plt.figure(figsize=(10,6))
sns.countplot(y='manufacturer', data=df, order=df['manufacturer'].value_counts().index[:10])
plt.title('Top 10 producători de mașini')
plt.xlabel('Număr anunțuri')
plt.ylabel('Producător')
plt.show()


plt.figure(figsize=(7,5))
sns.countplot(x='fuel', data=df)
plt.title('Distribuția tipurilor de combustibil')
plt.show()




plt.figure(figsize=(7,5))
sns.boxplot(x='fuel', y='price', data=df)
plt.title('Preț în funcție de combustibil')
plt.show()


plt.figure(figsize=(7,5))
sns.boxplot(x='transmission', y='price', data=df)
plt.title('Preț în funcție de transmisie')
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='condition', y='price', data=df)
plt.title('Preț în funcție de starea mașinii')
plt.show()


corr = df[['price', 'year', 'odometer']].corr()
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Corelații între variabile numerice')
plt.show()


plt.figure(figsize=(8,6))
sns.scatterplot(x='long', y='lat', hue='price', data=df, palette='viridis', alpha=0.5)
plt.title('Distribuția geografică a prețurilor')
plt.xlabel('Longitudine')
plt.ylabel('Latitudine')
plt.show()


df['fuel'] = df['fuel'].fillna('unknown')
df['transmission'] = df['transmission'].fillna('unknown')
df['drive'] = df['drive'].fillna('unknown')
df['type'] = df['type'].fillna('unknown')
df['paint_color'] = df['paint_color'].fillna('unknown')


df_encoded = pd.get_dummies(df,
                            columns=['manufacturer', 'fuel', 'transmission', 'drive', 'type', 'paint_color', 'state'],
                            drop_first=True)

print("Dimensiunea după codificare:", df_encoded.shape)


