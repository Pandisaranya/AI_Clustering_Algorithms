import pickle as pkl
load_model=pkl.load(open("Mall_Customers-k-means clustering.sav","rb"))
income=int(input("enter your income"))
spending=int(input("enter your spending score"))
Cluster=load_model.predict([[income,spending]])
print("Cluster number is ",Cluster)
