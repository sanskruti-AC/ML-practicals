#!/usr/bin/env python
# coding: utf-8

# ## Practical No.4: To perform an analysis of Decision Tree (DT).

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

# In[3]:


import kagglehub

path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")

print(path)


# ## 3. Check Dataset Files

# In[4]:


import os

print(os.listdir(path))


# ## 4. Load the Dataset

# In[5]:


import pandas as pd

df = pd.read_csv(path + "/data.csv")

df.head()


# ## 5. Explore the Dataset

# In[6]:


print("Shape of dataset:", df.shape)

print("\nColumn names:")
print(df.columns)


# ## 6. Data Preprocessing

# In[7]:


df = df.drop(['id', 'Unnamed: 32'], axis=1)

print("New shape:", df.shape)


# ## 7. Convert Target Variable

# In[8]:


df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

print(df['diagnosis'].value_counts())


# ## 8. Separate Features and Target Variable

# In[9]:


X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

print("Features shape:", X.shape)
print("Target shape:", y.shape)


# ## 9. Split the Dataset

# In[10]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)


# ## 10. Train the Decision Tree Model

# In[11]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

print("Decision Tree model trained successfully.")


# ## 11. Make Predictions

# In[12]:


y_pred = model.predict(X_test)

print("Predictions:")
print(y_pred)


# ## 12. Calculate Accuracy

# In[13]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Accuracy (%):", accuracy * 100)


# ## 13. Confusion Matrix

# In[14]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)


# ## 14. Classification Report

# In[15]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# ## 15. Confusion Matrix Visualization

# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=['Benign', 'Malignant'],
    yticklabels=['Benign', 'Malignant']
)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Decision Tree')
plt.show()


# ## 16. Decision Tree Visualization

# In[17]:


from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=['Benign', 'Malignant'],
    filled=True,
    max_depth=3
)

plt.title('Decision Tree Visualization')
plt.show()


# In[ ]:





# In[ ]:




