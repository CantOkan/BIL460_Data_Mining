# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 23:06:35 2019

@author: canok
"""

from sklearn.datasets import load_files
from nltk.corpus import stopwords
import nltk
import pickle
import re
import numpy
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


### Machine l. modelini eğitmek için x_train,y_train olarak ayırıyoruz
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)


from sklearn.tree import DecisionTreeClassifier 

#Decision Tree C.
dtClassifier=DecisionTreeClassifier()
dtClassifier.fit(x_train,y_train)

y_pred=dtClassifier.predict(x_test)
#train_model fonksiyonuna modelin 10-fold cross-validation geçmesi için gönderiz

print("DecisionTree Ortalama Doğruluk : "+str(dtClassifier.score(x_test,y_test)))



from sklearn.metrics import classification_report

print(classification_report(y_test,y_pred))
