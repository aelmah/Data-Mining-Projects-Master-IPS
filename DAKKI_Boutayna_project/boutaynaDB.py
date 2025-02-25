import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_excel("C:/Users/pc/Desktop/BD_Kmeans.xlsx")

X = df[['Revenu Annuel', 'Dépenses Annuelles']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, algorithm='elkan', random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)


plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['Cluster'], cmap='viridis', edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label="Centroides")
plt.xlabel("Revenu Annuel (Normalisé)")
plt.ylabel("Dépenses Annuelles (Normalisé)")
plt.title("Clustering K-Means avec k=3")
plt.legend()
plt.show()


df["Catégorie Revenu"] = pd.cut(df["Revenu Annuel"], bins=3, labels=["Faible", "Moyen", "Élevé"])
df["Catégorie Dépenses"] = pd.cut(df["Dépenses Annuelles"], bins=3, labels=["Faible", "Moyen", "Élevé"])
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
transactions = df[["Catégorie Revenu", "Catégorie Dépenses"]].astype(str).apply(lambda x: x + "_" + x.index, axis=1).values.tolist() 

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)


frequent_itemsets = apriori(df_encoded, min_support=0.2, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
import matplotlib.pyplot as plt
import seaborn as sns


pivot = rules.pivot(index="antecedents", columns="consequents", values="lift")
sns.heatmap(pivot, annot=True, cmap="YlGnBu")
plt.title("Corrélations Revenus/Dépenses (Lift)")
plt.show()