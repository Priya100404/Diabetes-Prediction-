#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[15]:


# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('diabetes.csv') 


# In[3]:





# In[16]:


diabetes_dataset.head()


# In[17]:


# number of rows and Columns in this dataset
diabetes_dataset.shape


# In[18]:


# getting the statistical measures of the data
diabetes_dataset.describe()


# In[19]:


diabetes_dataset['Outcome'].value_counts()


# 0 --> Non-Diabetic
# 
# 1 --> Diabetic

# In[20]:


diabetes_dataset.groupby('Outcome').mean()


# In[21]:


# separating the data and labels
X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']


# In[22]:


print(Y)


# In[12]:


scaler = StandardScaler()


# In[23]:


scaler.fit(X)


# In[25]:


standardized_data = scaler.transform(X)


# In[26]:


print(standardized_data)


# In[27]:


X = standardized_data
Y = diabetes_dataset['Outcome']


# In[28]:


print(X)
print(Y)


# In[29]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)


# In[30]:


print(X.shape, X_train.shape, X_test.shape)


# In[31]:


classifier = svm.SVC(kernel='linear')


# In[32]:


#training the support vector Machine Classifier
classifier.fit(X_train, Y_train)


# In[33]:


# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[34]:


print('Accuracy score of the training data : ', training_data_accuracy)


# In[36]:


input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')


# In[ ]:




