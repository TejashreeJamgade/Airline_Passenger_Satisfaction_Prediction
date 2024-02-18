#!/usr/bin/env python
# coding: utf-8

# # Objective
# 
# Predicting airline passenger satisfaction using random forest, gradient boosting, and KNN.

# # Below are the steps executed in this notebook
# 
# 1. IMPORT LIBRARIES
# 2. LOAD DATASET
# 3. DATA UNDERSTANDING
# 
#    1. Check Data Description
#    2. Check data info
# 3. Check Missing Value
# 4. DATA PREPARATION
# 
#    1. Handling Missing Value
#    2. Duplicated Data
# 5. STATISTICAL SUMMARY
# 
#    1. Numerical columns
#    2. Categorical Columns
# 
# 6. Outlier Detection
# 
# 7. Encoding Categorical Columns
# 
#    1. One-hot encoding
#    2. Label encoding on Target Column
# 
# 8. Export encoded data for EDA

# # 1. IMPORT LIBRARIES

# In[2]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # 2. LOAD DATASET

# In[3]:


df = pd.read_csv('airline_passenger_satisfaction.csv')
print('Total Row : ', len(df))
df.head(5)


# # 3.DATA UNDERSTANDING 

# # 1) check describes 

# In[16]:


df.describe()


# # 2. Check data info

# In[18]:


df.info()


# # 3. Check Missing Value

# In[4]:


null_value = (129880 - 129487 ) /129880
percentage = null_value * 100

print("missing value = {:.1f}%".format(percentage))


# In[ ]:





# In[5]:


df.dropna(inplace=True)


# In[6]:


df.info()


# In[ ]:





# In[ ]:





# In[7]:


df.isnull().sum()


# In[8]:


df.duplicated().sum()


# In[9]:


categoricals= list(df.select_dtypes(include=['object']).columns)


# In[10]:


numericals= list(df.select_dtypes(include=['float','int']).columns)


# In[11]:


categorical_count=(df.select_dtypes(include=['object']).columns)
numerical_count=(df.select_dtypes(include=['float','int']).columns)
# print column names
print('Categorical columns:', categorical_count,"->", categoricals)
print('Numerical columns:', numerical_count, "->",numericals)


# In[12]:


df[numericals].describe().T


# In[13]:


numericals


# In[14]:


for col in numericals if numericals[col]mean()>numericals[col]median()


# In[ ]:


filtered_columns = [col for col in numericals if df[col].mean() > df[col].median()]
print(filtered_columns)


# In[ ]:


df[categoricals].describe()


# In[ ]:


# adjust the figure size for better readability
plt.figure(figsize=(40,10))

# plotting
features = numericals
for i in range(0, len(features)):
    plt.subplot(1, len(features), i+1)
    sns.boxplot(y=df[features[i]], color='red')
    plt.tight_layout()


# # Encoding categorical columns

# In[ ]:


for col in categoricals:
    print(f"Unique values of {col}:{df[col].unique()}")


# # 1) One-hot encoding

# In[ ]:


df_encoded= pd.get_dummies(df,columns=['Gender','Customer Type','Type of Travel','Class'])


# In[ ]:


df_encoded['Satisfaction'].unique()


# In[ ]:


# df_encoded['Satisfaction'] = (df_encoded['Satisfaction'] != 'Satisfied').astype(int)
df_encoded['Satisfaction'] = df_encoded['Satisfaction'].replace({"Neutral or Dissatisfied":1,"Satisfied":0})


# In[ ]:


# Reorder column
df_encoded = df_encoded[['ID', 'Age', 'Flight Distance', 'Departure Delay', 'Arrival Delay',
       'Departure and Arrival Time Convenience', 'Ease of Online Booking',
       'Check-in Service', 'Online Boarding', 'Gate Location',
       'On-board Service', 'Seat Comfort', 'Leg Room Service', 'Cleanliness',
       'Food and Drink', 'In-flight Service', 'In-flight Wifi Service',
       'In-flight Entertainment', 'Baggage Handling',
'Gender_Female', 'Gender_Male', 'Customer Type_First-time',
       'Customer Type_Returning', 'Type of Travel_Business',
       'Type of Travel_Personal', 'Class_Business', 'Class_Economy',
       'Class_Economy Plus','Satisfaction']]


# In[ ]:


df_encoded.head(3)


# In[ ]:


df_encoded.to_csv("airline_passenger_satisfaction_EDA.csv",index=False)


# In[ ]:





# In[ ]:




