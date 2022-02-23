#!/usr/bin/env python
# coding: utf-8

# In[59]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier


# In[60]:


df = pd.read_csv('./data/GSPC.csv')
df.head


# In[61]:


df = df.drop('Date', axis=1)


# In[62]:


df[-2::]


# In[63]:


predict_base = df[-1::]
predict_base


# In[64]:


base = df.drop(df[-1::].index, axis=0)
base.tail()


# In[65]:


base['Target'] = base['Close'][1:len(base)].reset_index(drop=True)
base.tail()


# In[66]:


predict = base[-1::].drop('Target', axis=1) 
predict


# In[67]:


train = base.drop(base[-1::].index, axis=0)
train.tail()


# In[68]:


train.loc[train['Target'] > train['Close'], 'Target'] = 1
train.loc[train['Target'] != 1, 'Target'] = 0
train.tail()


# In[69]:


y = train['Target']
x = train.drop('Target', axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
model = ExtraTreesClassifier()
model.fit(X_train, y_train)
result = model.score(X_test, y_test)
print("Accuracy: ", result)
predict['Target'] = model.predict(predict)


# In[70]:


predict # yesterday


# In[71]:


predict_base # tomorrow


# In[72]:


base.tail()


# In[73]:


base = base.append(predict_base, sort=True)
base.tail()


# In[ ]:


predict.to_csv('./data/yesterday.csv', index=False)
base.to_csv('./data/today.csv', index=False)
predict_base.to_csv('./data/tomorrow.csv', index=False)

