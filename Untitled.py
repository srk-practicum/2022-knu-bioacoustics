#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[57]:


common_voices_other_df = pd.read_csv(r'C:\common_voices\other.tsv', sep='\t')


# In[3]:


#common_voices_other_df = (common_voices_other_df[['client_id', 'path', 'gender']]).dropna()


# In[74]:


def age_to_number(age):
    if age=='teens':
        return 1
    elif age=='twenties':
        return 2
    elif age=='thirties':
        return 3
    else:
        return 4
def sex(x):
    if x != 'female':
        return 0
    else:
        return 1


# In[78]:


common_voices_other_df['age']=common_voices_other_df['age'].apply(age_to_number)
common_voices_other_df['gender']=common_voices_other_df['gender'].apply(sex)


# In[79]:


common_voices_other_df['gender'].value_counts()


# In[80]:


common_voices_other_df.columns


# In[81]:


common_voices_other_df['age'] = common_voices_other_df['age'].dropna()


# In[82]:


X = common_voices_other_df[['up_votes','down_votes','age']]


# In[83]:


y = common_voices_other_df['gender']


# In[84]:


X, y


# In[85]:


common_voices_other_df['locale'].value_counts()


# In[86]:


common_voices_other_df['accent'].value_counts()


# In[87]:


from sklearn.ensemble import RandomForestClassifier

clf=RandomForestClassifier()


# In[88]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[89]:


clf.fit(X_train, y_train)


# In[45]:


X_train.value_counts()


# In[90]:


y_pred = clf.predict(X_test)


# In[93]:


y_pred


# In[94]:


(y_pred == y_test).sum()/len(y_test)


# In[ ]:




