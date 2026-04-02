import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


data_1=pd.read_csv(r"C:\Users\E\Desktop\Sleep_health_and_lifestyle_dataset.csv")
data_2=pd.get_dummies(data_1,columns=["Gender"],dtype=int,drop_first=True)
data_encoded=pd.get_dummies(data_2,columns=["Occupation","BMI Category"],dtype=int)
"""
data_encoded["Age"]=pd.qcut(data_encoded["Age"],q=4,labels=[0,1,2,3])
data_encoded["Sleep Duration"]=pd.qcut(data_encoded["Sleep Duration"],q=4,labels=[0,1,2,3])
data_encoded["Physical Activity Level"]=pd.qcut(data_encoded["Physical Activity Level"],q=5,labels=[0,1,2,3,4])
"""

data_encoded[["Systolic","Diastolic"]]=data_encoded["Blood Pressure"].str.split("/",expand=True)
data_encoded["Systolic"]=data_encoded["Systolic"].astype(int)
data_encoded["Diastolic"]=data_encoded["Diastolic"].astype(int)
data=data_encoded.drop(columns=["Blood Pressure"])


data=data.drop(columns=["Person ID","Sleep Disorder"])
matris=data.corr()

scale=data[["Age","Sleep Duration","Systolic","Diastolic","Daily Steps"]]
scaled=MinMaxScaler(feature_range=(0,1)).fit_transform(scale)

mean=data_1.groupby("Occupation")["Stress Level"].mean()

"""
errors=[]
for i in range(1,11):
    model=KMeans(n_clusters=i,random_state=21)
    model.fit(scaled)
    errors.append(model.inertia_)


plt.figure(figsize=(15,8))
sns.lineplot(errors)
"""



kmeans=KMeans(n_clusters=4,random_state=21)

kmeans.fit_predict(scaled)

cluster=kmeans.labels_

data_encoded["Cluster"]=cluster











plt.figure(figsize=(15,6))
sns.barplot(x=mean.index,y=mean.values)
plt.xticks(rotation=45)

plt.figure(figsize=(15,6))
sns.scatterplot(x=data_1["Sleep Duration"],y=data_1["Stress Level"],hue=data_1["Occupation"])

plt.figure(figsize=(15,6))
sns.scatterplot(x=data_1["Sleep Duration"],y=data_1["Stress Level"],hue=data_encoded["Cluster"],markers=["o", "s", "^", "D"],style=data_encoded["Cluster"],palette="colorblind",s=100)

plt.figure(figsize=(12,8))
sns.heatmap(matris,annot=True,cmap="coolwarm",fmt=".2f")
plt.show()