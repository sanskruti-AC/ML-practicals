#!/usr/bin/env python
# coding: utf-8

# ## Practical No.6:To perform an analyze of Gradient decienent method

# In[1]:


# Name: sanskruti ajay chandore
# Roll No.: 09
# Branch: B.E. CSE (Data Science)
# Year: Final Year
# Semester: VII


# ## 1. Install KaggleHub Library

# In[2]:


get_ipython().system('pip install kagglehub')


# ## 2. Download Dataset from Kaggle

# In[5]:


import kagglehub

path = kagglehub.dataset_download("camnugent/california-housing-prices")

print(path)


# ## 3. Check Dataset Files

# In[6]:


import os

print(os.listdir(path))


# ## 4. Load the Dataset

# In[7]:


import pandas as pd

df = pd.read_csv(path + "/housing.csv")

df.head()


# ## 5. Explore the Dataset

# In[8]:


print("Shape of dataset:", df.shape)

print("\nColumn names:")
print(df.columns)


# ## 6. Check Missing Values

# In[9]:


print(df.isnull().sum())


# ## 7. Handle Missing Values

# In[10]:


df = df.dropna()

print("New shape:", df.shape)


# ## 8. Convert Categorical Data

# In[11]:


df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

df.head()


# ## 9. Separate Features and Target Variable

# In[12]:


X = df.drop('median_house_value', axis=1)
y = df['median_house_value']

print("Features shape:", X.shape)
print("Target shape:", y.shape)


# ## 10. Split the Dataset

# In[13]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ## 10. Split the Dataset

# In[14]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# 
# ## 11. Feature Scaling

# In[15]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Feature scaling completed successfully.")


# ## 12. Apply Gradient Descent

# In[16]:


from sklearn.linear_model import SGDRegressor

model = SGDRegressor(
    max_iter=1000,
    learning_rate='invscaling',
    random_state=42
)

model.fit(X_train, y_train)

print("Gradient Descent model trained successfully.")


# ## 13. Make Predictions

# In[17]:


y_pred = model.predict(X_test)

print("Predictions:")
print(y_pred[:10])


# ## 14. Calculate Mean Squared Error

# In[18]:


from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)


# ## 15. Calculate Root Mean Squared Error

# In[19]:


import numpy as np

rmse = np.sqrt(mse)

print("Root Mean Squared Error:", rmse)


# ## 16. Calculate R² Score

# In[20]:


from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print("R² Score:", r2)


# ## 17. Actual vs Predicted Values

# In[21]:


import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))

plt.scatter(y_test, y_pred)

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values - Gradient Descent')
plt.show()


# In[ ]:




