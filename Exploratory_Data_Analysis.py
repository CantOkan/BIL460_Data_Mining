# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 20:15:12 2019

@author: canok
"""

from sklearn.datasets import load_files
from nltk.corpus import stopwords
import nltk
import re
import numpy as np
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import roc_auc_score

import seaborn as sns

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
    



custom_map = {0: "materials.sector", 1: "energy.sector",2:"financial.sector",3:"healthcare.sector",
              4:"technology.sector",5:"transportation.sector",6:"utilities.sector"}


from sklearn.feature_extraction.text import CountVectorizer
vectorizer=CountVectorizer(max_features=10,stop_words=Stop_words)

#ilk 20 frequent word
X=vectorizer.fit_transform(corpus).toarray()

#### ilgili Wordsler ve onların sıklıkları
df_WORDS=pd.DataFrame(X,columns=vectorizer.get_feature_names())

df_WORDS["class"]=y

df_WORDS['class'] = df_WORDS['class'].map(custom_map)


df_WORDS['class'].value_counts().plot.barh().set_title('Classların Dağılımı')




#df_WORDS.plot.bar()
