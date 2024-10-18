import pandas as p
dataset=p.read_csv("Mall_Customers.csv")
x=dataset.iloc[:,3:5].values
from sklearn.cluster import AffinityPropagation
clustermodel=AffinityPropagation()
label=clustermodel.fit_predict(x)
print(label)
supervised=dataset
supervised["cluster_group"]=label
supervised.to_csv("cluster_affinity.csv",index=False)
#print(centroids)
#print(supervised.columns[3])
#print(supervised.columns)
import seaborn as s
import matplotlib.pyplot as plt
colors = ["red","blue","purple","green"]
#palette=colors
facet = s.lmplot(data=supervised,x=supervised.columns[3], y=supervised.columns[4],
                   hue=supervised.columns[5],fit_reg=False,legend=True,legend_out=True)

plt.show()

