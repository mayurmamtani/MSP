#!/usr/bin/env python
# coding: utf-8

# # Loading Libraries

# In[1]:


#get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sn


# In[3]:


# import dataset

df = pd.read_csv('Dataset/amazonLabelled - amazonLabelled.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.isnull().values.any()


# In[7]:


df["Sentiment"].value_counts()


# In[8]:


from sklearn.preprocessing import LabelEncoder


# In[9]:


lb=LabelEncoder()


# In[10]:


lb.fit(df["Sentiment"])


# In[11]:


df["Sentiment"]=lb.transform(df["Sentiment"])


# In[12]:


df.head()


# # Train Test Split

# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


X_train,X_test,y_train,y_test=train_test_split(df.drop("Sentiment",axis=1),df["Sentiment"],test_size=0.2)


# In[15]:


X_train.shape


# In[16]:


train_df=pd.concat([X_train,y_train],axis=1).to_csv("Dataset/train_set.csv",index=False)


# In[17]:


test_df=pd.concat([X_test,y_test],axis=1).to_csv("Dataset/test_set.csv",index=False)


# # Data ingestion step - Training dataset

# In[18]:


train_df=pd.read_csv("Dataset/train_set.csv")


# In[19]:


X_train,y_train=train_df["Feedback"],train_df["Sentiment"]


# In[20]:


X_train


# In[21]:


y_train


# In[22]:


y_train.value_counts()


# In[23]:


plt.hist(y_train)


# # Preprocessing Data

# In[24]:


#get_ipython().system('pip install nltk')


# In[26]:


import nltk


# In[27]:


nltk.download("all")


# In[28]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
lemmatizer=WordNetLemmatizer()


# In[29]:


def preprocess_data(data):
    corpus=[]
    for i in data:
        mess=re.sub("[^a-zA-Z0-9]"," ",i)
        mess=mess.lower().split()
        mess=[lemmatizer.lemmatize(word) for word in mess if word not in stopwords.words("english")]
        mess=" ".join(mess)
        corpus.append(mess)
    return corpus    


# In[30]:


corpus=preprocess_data(X_train)


# In[31]:


len(corpus)


# In[32]:


from sklearn.feature_extraction.text import CountVectorizer


# In[33]:


cv=CountVectorizer(ngram_range=(1,2))


# In[34]:


cv.fit(corpus)


# In[35]:


count_train=cv.transform(corpus)


# In[36]:


count_train.shape


# # Model Training

# In[37]:


from sklearn.ensemble import RandomForestClassifier


# In[38]:


rf=RandomForestClassifier(n_estimators=1200)


# In[39]:


rf.fit(count_train,y_train)


# In[40]:


rf.score(count_train,y_train)


# In[41]:


from sklearn.model_selection import cross_val_score


# In[42]:


scores=cross_val_score(rf,count_train,y_train,cv=3)


# In[43]:


scores.mean()


# In[44]:


scores.std()


# # Hyperparameter Tuning

# In[45]:


from sklearn.model_selection import GridSearchCV


# In[46]:


param_grid={'n_estimators': [700,1000,1200], 'min_samples_split': [2,4,8,16]}


# In[47]:


grid=GridSearchCV(rf,param_grid,n_jobs=-1)


# In[48]:


grid.fit(count_train,y_train)


# In[49]:


n_est=grid.best_params_["n_estimators"]
min_sam_splt=grid.best_params_["min_samples_split"]


# In[50]:


rf=RandomForestClassifier(n_estimators=n_est,min_samples_split=min_sam_splt)


# In[51]:


rf.fit(count_train,y_train)


# # Model Packaging Step

# In[52]:


import joblib


# In[53]:


joblib.dump(cv,"models/count_vectorizer.pkl")


# In[54]:


joblib.dump(rf,"models/rf_sent_model.pkl")


# In[ ]:




