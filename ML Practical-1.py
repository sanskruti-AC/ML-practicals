#!/usr/bin/env python
# coding: utf-8

# ## Practical No.1:To perform an analysis of Logistic Regression.

# In[11]:


# Name:sanskruti ajay chandore 
# Roll No.: 09
# Branch: B.E. CSE (Data Science)
# Year: Final Year
# Semester: VII


# In[12]:


pip install kagglehub


# In[13]:


import kagglehub
# Download the Breast Cancer Wisconsin dataset from Kaggle
path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")

print(path)


# In[14]:


# Display the files available in the downloaded dataset
import os

print(os.listdir(path))


# ## 1. Import libraries

# In[15]:


import pandas as pd


# ## 2. Load the dataset

# In[16]:


df = pd.read_csv(path + "/data.csv")
df.head()


# ## 3. Check dataset information

# In[17]:


print("Shape of dataset:", df.shape)
print("\nColumn names:")
print(df.columns)


# ## 4. Remove unnecessary columns

# In[18]:


df = df.drop(['id', 'Unnamed: 32'], axis=1)

print("New shape:", df.shape)


# ## 5. Check target value

# In[19]:


print(df['diagnosis'].value_counts())


# ## 6. Convert target labels

# In[20]:


df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

print(df['diagnosis'].value_counts())


# ## 7. Separate features and target

# In[21]:


X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

print("X shape:", X.shape)
print("y shape:", y.shape)


# ## 8. Split the dataset

# In[22]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ## 9. Feature scaling

# In[23]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Feature scaling completed successfully.")


# ## 10. Train Logistic Regression

# In[24]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

print("Logistic Regression model trained successfully.")


# ## 11. Make predictions

# In[25]:


y_pred = model.predict(X_test)

print("Predictions:")
print(y_pred)


# ## 12. Calculate accuracy

# In[26]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)


# ## 13. Confusion Matrix

# In[27]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# ## 14. Classification Report

# In[28]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# ## 15. Confusion Matrix Visualization

# In[29]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Benign', 'Malignant'],
    yticklabels=['Benign', 'Malignant']
)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()


# In[ ]:





# In[ ]:




