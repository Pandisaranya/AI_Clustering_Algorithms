import pandas as p
dataset=p.read_csv("Mall_Customers.csv")
X=dataset.iloc[:,3:5].values
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
hierarchy.dendrogram(hierarchy.linkage(X,method="ward"))
plt.title("Dendrogram")
plt.xlabel("Customer")
plt.ylabel("Euclidean distance")
plt.show
from sklearn.cluster import AgglomerativeClustering
clustermodel = AgglomerativeClustering(n_clusters=5)
label=clustermodel.fit_predict(X)
print(label)
supervised=dataset
supervised["cluster_group"]=label
supervised.to_csv("cluster_agglomerative.csv",index=False)
#print(centroids)
#print(supervised.columns[3])
#print(supervised.columns)
import seaborn as s
colors = ["red","blue","purple","green"]
#palette=colors
facet = s.lmplot(data=supervised,x=supervised.columns[3], y=supervised.columns[4],
                   hue=supervised.columns[5],fit_reg=False,legend=True,legend_out=True)

plt.show()
