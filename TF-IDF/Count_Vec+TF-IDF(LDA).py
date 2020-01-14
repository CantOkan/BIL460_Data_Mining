# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 02:17:06 2019

@author: canok
"""

from sklearn.datasets import load_files
from nltk.corpus import stopwords
import nltk
import pickle
import re
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd

Stop_words=stopwords.words('english')


dataset=load_files('Web Page Classification/')


#datasetimizi oluşturuyoruz
#X:dataset
#Y:Classlar

X,y=dataset.data,dataset.target

from nltk.stem import WordNetLemmatizer
lem=WordNetLemmatizer()
            

corpus=[] #Documentları corpus kaydediyoruz
            #fakat doha öncesinde preprocessing(boşlukların atılması,noktalama işretleri,a an )

for i in range(0,len(X)):
    
    document= BeautifulSoup(X[i],'lxml')#html tagları arasında veri çekme
    sentences=""
    temp=document.getText()
    
    temp=temp.lower()
    temp=re.sub(r'\W',' ',temp)# noktalam işaretleri
    temp=re.sub(r'\d+'," ",temp)#delete digits
        
    temp=re.sub(r'\s+[a-z]\s+',' ',temp)# delete a,s ,c  single charcters

    temp=re.sub(r'\s+',' ',temp) #büyük spaceleri ' ' ile 
    
        ### Stemming - Lemmatizing
        ### word kökünü almak
        
    words=nltk.word_tokenize(temp)
    newwords=[lem.lemmatize(word) for word in words]
    sentences=' '.join(newwords)
    corpus.append(sentences)





## Creating our Bow model
from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(max_features=100, min_df=600 ,max_df=2.5 ,stop_words=Stop_words)

#ilk 20 frequent word
X=vectorizer.fit_transform(corpus).toarray()

#### ilgili Wordsler ve onların sıklıkları
df_WORDS=pd.DataFrame(X,columns=vectorizer.get_feature_names())


##### TF-IDF model #####

#TF-IDF uyguluyor 
from sklearn.feature_extraction.text import TfidfTransformer

transformer=TfidfTransformer() #TF-IDf objesi oluşturmak için

X=transformer.fit_transform(X).toarray()



#####################################################################
###♥ Liner Discrement Analysis ###################
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 

lda=LinearDiscriminantAnalysis(n_components=6)

#PCA göre
#2 parametre veriyoruz
#lda class arasındaki farkı maksimize etmeye çalışır 
#sınıflar arasındaki mesafeyi maksimize etmeye çalışır
lda_x=lda.fit_transform(X,y)

##############



### Machine l. modelini eğitmek için x_train,y_train olarak ayırıyoruz
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(lda_x,y,test_size=0.33,random_state=0)


import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import recall_score,precision_score,roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report

k_scores=[]
k_range=range(3,25)
#KNN için en uygun k değerini bulmaya çalışırız.
def find_k_hyperparameter(x_train,y_train):
    #Train datamızı 10 böldük ve her seferinden 9 tanesi train ,1 tanesi validation
    for k in k_range:
        #3-25 kadar sırasıyla k değerlerini deneriz
        knn=KNeighborsClassifier(n_neighbors=k)
        
        accurires=train_model(knn,x_train,y_train)
        """
        print("K={} için".format(k))
        print("Average accuracy= {}".format(np.mean(accurires)))
        #standart d.
    
        print("Standart dv:{}".format(np.std(accurires)))
        """
        acc=np.mean(accurires)
        
        k_scores.append(acc)
    
    return acc

#10-fold cross-validation ile modelimi eğitmek ve test etmek için  
    #bu fonksiyon kullanırız
    #fonksiyon parametre olarak trainset ile birlikte modelimizi alır
def train_model(model,x_train,y_train):
    #10-fold cross-validation 
    accurires=cross_val_score(estimator=model,X=x_train,y=y_train,cv=10)
    return accurires




print("KNN İÇİN ")

#DataSetimiz train ve test olamak üzere ikiye böleriz

#KNN için en iyi k değerini bulmaya çalışırız
knn=find_k_hyperparameter(x_train,y_train)

print("KNN ORTALAMA"+str(knn))

##◘ en iyi k değeri için bak
knn=KNeighborsClassifier(n_neighbors=5)
#k değerleri için Accuracylerimiz
plt.plot(k_range,k_scores)
plt.xlabel("Value of K for KNN")
plt.ylabel("Cross-val Acc.")
plt.show()
#Görüldüğü üzere bütün k değerleri için aynı Doğruluk oranı aldık.
#Bu durumda k üç alınmakta bir sakınca yoktur.

#Knn x_train ve y_train için oluşturulur
knn.fit(x_train,y_train)
#Knn alg. başarısı:
print("Knn model Accuracy:",knn.score(x_test,y_test))
y_pred=knn.predict(x_test)

from sklearn.metrics import recall_score,precision_score,roc_curve
from sklearn.metrics import confusion_matrix

#Modelin Confusion M. olusturulur
y_pred=knn.predict(x_test)


####  confusion Matrix and Other Evaluation Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score,precision_score

#Modelin Confusion M. olusturulur

#Recall ve Precision ölçütleri
print("Recall:",recall_score(y_test,y_pred, pos_label='positive',
                                           average='micro'))

print("Precision", precision_score(y_test, y_pred, pos_label='positive',
                                           average='micro'))

f,ax=plt.subplots(figsize=(5,5))



cm=confusion_matrix(y_test,y_pred)
#Seaborn kütüphanesini kullanarak Confusion M. Grafik olarak oluştururz
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("Y_pred")
plt.ylabel("y_true")

plt.show()


#Classification Report
print(classification_report(y_test,y_pred))



y_score = knn.fit(x_train, y_train)

from sklearn.metrics import roc_curve, auc

#Roc Curve
#falsePos.Rate
#TruePos.Rate değerleri bulunup Roc Curve oluşturulur
# Compute ROC curve and ROC area for each class




###Decision Tree



from sklearn.tree import DecisionTreeClassifier 

#Decision Tree C.
dtClassifier=DecisionTreeClassifier()

#train_model fonksiyonuna modelin 10-fold cross-validation geçmesi için gönderiz
DecisionTreeAccuracies=(train_model(dtClassifier,x_train,y_train))

print("DecisionTree Ortalama Doğruluk : "+str(DecisionTreeAccuracies.mean()))

print("DecisionTree Standart Sapması : "+str(DecisionTreeAccuracies.std()))

dtClassifier.fit(x_train,y_train)

print("Decision Tree Accuracy:",dtClassifier.score(x_test,y_test))
#Karar Ağacının Doğru tahmin oranı

y_pred=dtClassifier.predict(x_test)



print("Recall:",recall_score(y_test,y_pred, pos_label='positive',
                                           average='micro'))

print("Precision", precision_score(y_test, y_pred, pos_label='positive',
                                           average='micro'))

f,ax=plt.subplots(figsize=(5,5))



cm=confusion_matrix(y_test,y_pred)
#Seaborn kütüphanesini kullanarak Confusion M. Grafik olarak oluştururz
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("Y_pred")
plt.ylabel("y_true")

plt.show()



print(classification_report(y_test,y_pred))





####### Naive Bayes 

from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

#model'i train_model fonksiyonuna modelin 10-fold cross-validation geçmesi için gönderiz

NaiveBayesAccuracies=(train_model(nb,x_train,y_train))

print("NavieBayes Ortalama Doğruluk : "+str(NaiveBayesAccuracies.mean()))

print("NaiveBayes Standart Sapması : "+str(NaiveBayesAccuracies.std()))

nb.fit(x_train,y_train)


print("Accuracy of NaiveBayes: ",nb.score(x_test,y_test))
y_pred=nb.predict(x_test)
#Naive Bayes accuracy


print("Recall:",recall_score(y_test,y_pred, pos_label='positive',
                                           average='micro'))

print("Precision", precision_score(y_test, y_pred, pos_label='positive',
                                           average='micro'))

f,ax=plt.subplots(figsize=(5,5))



cm=confusion_matrix(y_test,y_pred)
#Seaborn kütüphanesini kullanarak Confusion M. Grafik olarak oluştururz
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("Y_pred")
plt.ylabel("y_true")

plt.show()



print(classification_report(y_test,y_pred))




#####
#######################SVMM###################################
## SVM kullanırken GridSearch'den yararlanacağız
## ve model için en iyi parametrelerini bulacağız (hyperparameter)

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


svm=SVC(kernel="linear")
svc_params = {'C':[1, 10],'degree':[1,10]}

#Cross-Validation=10 
#n_jobs=
svm_gs = GridSearchCV(svm, param_grid=svc_params,cv=10)
svm_gs.fit(x_train, y_train)

#Grid Search sonucunda
print("SVM best parameter:"+str(svm_gs.best_params_))


print("SVM best Score:"+str(svm_gs.best_score_))

#best parameter=> C:1 derece:1 olarak elde ediyoruz

#aynı modeli best parameterleri göz önünde bulundurarak oluşturuyoruz.
svm=SVC(kernel="linear",C=1,degree=1)

svm.fit(x_train,y_train)

print("Accuracy of SVM: ",svm.score(x_test,y_test))
y_pred=svm.predict(x_test)



print("Recall:",recall_score(y_test,y_pred, pos_label='positive',
                                           average='micro'))

print("Precision", precision_score(y_test, y_pred, pos_label='positive',
                                           average='micro'))

f,ax=plt.subplots(figsize=(5,5))



cm=confusion_matrix(y_test,y_pred)
#Seaborn kütüphanesini kullanarak Confusion M. Grafik olarak oluştururz
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("Y_pred")
plt.ylabel("y_true")

plt.show()





print(classification_report(y_test,y_pred))


###### Random Forest
## ensamble Learning. 


from sklearn.ensemble import RandomForestClassifier

#n_esti:kaç adet dec. tree
rf=RandomForestClassifier(n_estimators=100)
rf.fit(x_train,y_train)


print("Accuracy of Random Forest: ",rf.score(x_test,y_test))

y_pred=rf.predict(x_test)
#RAndom Forrest accuracy




print("Recall:",recall_score(y_test,y_pred, pos_label='positive',
                                           average='micro'))

print("Precision", precision_score(y_test, y_pred, pos_label='positive',
                                           average='micro'))

f,ax=plt.subplots(figsize=(5,5))



cm=confusion_matrix(y_test,y_pred)
#Seaborn kütüphanesini kullanarak Confusion M. Grafik olarak oluştururz
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("Y_pred")
plt.ylabel("y_true")

plt.show()




print(classification_report(y_test,y_pred))


##Logistic Regresion for multiple Classification

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver = 'lbfgs')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Accuracy of Logistic R.: ",model.score(x_test,y_test))

print(classification_report(y_test,y_pred))



print("Recall:",recall_score(y_test,y_pred, pos_label='positive',
                                           average='micro'))

print("Precision", precision_score(y_test, y_pred, pos_label='positive',
                                           average='micro'))

f,ax=plt.subplots(figsize=(5,5))



cm=confusion_matrix(y_test,y_pred)
#Seaborn kütüphanesini kullanarak Confusion M. Grafik olarak oluştururz
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("Y_pred")
plt.ylabel("y_true")

plt.show()



############## KMEANS

from sklearn.cluster import KMeans

wcss=[]

##ELBOX
for k in range(1,20):
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) #return wcss for k


plt.plot(range(1,20),wcss)

plt.xlabel("number of k values")
plt.ylabel("wcss")
plt.show()


#print(m_fit.n_clusters)

#print(m_fit.labels_)


#Görselleştirme

## for kmeans 


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 

lda=LinearDiscriminantAnalysis(n_components=3)

#LDA göre
#2 parametre veriyoruz
#lda class arasındaki farkı maksimize etmeye çalışır 
#sınıflar arasındaki mesafeyi maksimize etmeye çalışır
lda_x=lda.fit_transform(X,y)

#Att. setini 3 boyutta ifade etmeye çalıştık.
##############
kmeans=KMeans(n_clusters=7)

m_fit=kmeans.fit(lda_x)

labels=m_fit.labels_

y_kmeans = m_fit.predict(lda_x)
print(y_kmeans)

from mpl_toolkits.mplot3d import Axes3D



plt.rcParams['figure.figsize']=(16,9)
fig=plt.figure()
ax=Axes3D(fig)

ax.scatter(lda_x[:,0],lda_x[:,1],lda_x[:,2],
           c=labels,marker='s')

centers=m_fit.cluster_centers_
print(centers)
ax.scatter(centers[:,0],centers[:,1],centers[:,2],marker="D",
           c="red",s=1000)


print(classification_report(y,y_kmeans))


###Hiearchi
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 

from scipy.cluster.hierarchy import linkage,dendrogram
#◘ xtrain<lda_x  :)


data=x_train.copy()
#minimize variance
merg=linkage(data,method="ward")

#leaf
dendrogram(merg,leaf_rotation=90,leaf_font_size=3)

plt.ylabel("euclidean D.")
plt.show()


#Hiearcical clustering
#tüme varım
from sklearn.cluster import AgglomerativeClustering


hiyerartical_cluster=AgglomerativeClustering(n_clusters=7,affinity="euclidean",
                                             linkage="ward")

cluster=hiyerartical_cluster.fit_predict(lda_x)

data = pd.DataFrame(lda_x)

data["class"]=cluster
"""
ax.scatter(lda_x[:,0],lda_x[:,1],lda_x[:,2],
           c=cluster,marker='s')
"""


