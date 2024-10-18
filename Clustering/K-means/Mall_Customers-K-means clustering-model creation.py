import pandas as p
dataset=p.read_csv("Mall_Customers.csv")
x=dataset.iloc[:,3:5].values
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
list1=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(x)
    list1.append(kmeans.inertia_)
plt.plot(range(1,11),list1)
plt.title("elbow meathod")
plt.xlabel("wcss")
plt.show()
print(list1)
Y_kmeans=kmeans.fit_predict(x)
#print(Y_kmeans)
supervised=dataset
supervised["cluster_group"]=Y_kmeans
#print(supervised)
supervised.to_csv("cluster.csv",index=False)
centeroids=kmeans.cluster_centers_
#print(centeroids)
#print(supervised.columns[3])
#print(supervised.columns)
import seaborn as s
colors=['red','green','blue','purple']
#palette=colors
#ax=s.scatterplot(x[:,0],x[:,1],hue=y,palette=colors,alpha=0.5,s=7)
facet = s.lmplot(data=supervised,x=supervised.columns[3], y=supervised.columns[4],
                   hue=supervised.columns[5],fit_reg=False,legend=True)

plt.show()
import pickle as p
p.dump(kmeans,open("Mall_Customers-k-means clustering.sav","wb"))
print("file written")
