#!/usr/bin/env python
# coding: utf-8

# ## Practical No.2: To perform an analysis of K-Nearest Neighbors (KNN).

# In[1]:


# Name: sanskruti ajay chandore
# Roll No.: 09
# Branch: B.E. CSE (Data Science)
# Year: Final Year
# Semester: VII


# ### 1.Install KaggleHub Library

# In[2]:


get_ipython().system('pip install kagglehub')


# ### 2.Download Dataset from Kaggle

# In[3]:


import kagglehub
path = kagglehub.dataset_download("uciml/iris")
print(path)


# ### 3.Check Dataset Files

# In[4]:


import os
print(os.listdir(path))


# ### 4.Load the Dataset

# In[5]:


import pandas as pd
df = pd.read_csv(path + "/Iris.csv")
df.head()


# ### 5.Explore the Dataset

# In[6]:


print("Shape of dataset:", df.shape)
print("\nColumn names:")
print(df.columns)


# ### 6.Data Preprocessing

# In[7]:


df = df.drop('Id', axis=1)
print("New shape:", df.shape)


# ### 7.Check Target Classes

# In[8]:


print(df['Species'].value_counts())


# ### 8.Separate Features and Target Variable

# In[9]:


X = df.drop('Species', axis=1)
y = df['Species']

print("Features shape:", X.shape)
print("Target shape:", y.shape)


# ### 9.Split the Dataset into Training and Testing Sets

# In[10]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ### 10.Feature Scaling

# In[11]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Feature scaling completed successfully.")


# ### 11.Train the KNN Model

# In[12]:


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)

model.fit(X_train, y_train)

print("KNN model trained successfully.")


# ### 12.Make Predictions

# In[13]:


y_pred = model.predict(X_test)

print("Predictions:")
print(y_pred)


# ### 13.Calculate Model Accuracy

# In[14]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)


# ### 14.Confusion Matrix

# In[15]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# ### 15.Classification Report

# In[16]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# ### 16.Visualize the Confusion Matrix

# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=['Setosa', 'Versicolor', 'Virginica'],
    yticklabels=['Setosa', 'Versicolor', 'Virginica']
)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - KNN')
plt.show()


# In[ ]:





# In[ ]:




