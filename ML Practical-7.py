#!/usr/bin/env python
# coding: utf-8

# ## Practical No.7:To perform an analysis of Ensemble Technique.

# In[1]:


# Name: sanskruti ajay cahndore
# Roll No.: 09
# Branch: B.E. CSE (Data Science)
# Year: Final Year
# Semester: VII


# ## 1. Install and Import Required Libraries

# In[2]:


get_ipython().system('pip install kagglehub')


# ## 2. Download the Dataset from Kaggle

# In[3]:


import kagglehub

path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")

print(path)


# ## 3. Display the Files in the Dataset

# In[4]:


import os

print(os.listdir(path))


# ## 4. Load the Dataset

# In[5]:


import pandas as pd

df = pd.read_csv(path + "/data.csv")

df.head()


# ## 5. Display the Shape and Column Names of the Dataset

# In[6]:


print("Number of Rows and Columns:", df.shape)
print(df.columns)


# ## 6. Check for Missing Values

# In[7]:


print(df.isnull().sum())


# ## 7. Remove Unnecessary Columns

# In[8]:


df = df.drop(['id', 'Unnamed: 32'], axis=1)

df.head()


# ## 8. Convert the Target Variable into Numerical Values

# In[9]:


df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

df['diagnosis'].value_counts()


# ## 9. Separate Features and Target Variable

# In[10]:


X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)


# ## 10. Split the Dataset into Training and Testing Data

# In[11]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# ## 11. Standardize the Features

# In[12]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ## 12. Create Individual Machine Learning Models

# In[13]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

model1 = LogisticRegression(max_iter=1000, random_state=42)
model2 = SVC(kernel='linear', probability=True, random_state=42)
model3 = RandomForestClassifier(n_estimators=100, random_state=42)

print("Models created successfully.")


# ## 13. Create the Voting Ensemble Model

# In[14]:


from sklearn.ensemble import VotingClassifier

ensemble_model = VotingClassifier(
    estimators=[
        ('logistic_regression', model1),
        ('svm', model2),
        ('random_forest', model3)
    ],
    voting='soft'
)

ensemble_model.fit(X_train_scaled, y_train)

print("Voting Ensemble Model trained successfully.")


# ## 14. Make Predictions Using the Ensemble Model

# In[15]:


y_pred = ensemble_model.predict(X_test_scaled)

print(y_pred)


# ## 15. Calculate the Accuracy of the Ensemble Model

# In[16]:


from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Ensemble Model Accuracy:", accuracy)
print("Ensemble Model Accuracy: {:.2f}%".format(accuracy * 100))


# ## 16. Generate the Confusion Matrix

# In[17]:


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)


# ## 17. Display the Classification Report

# In[18]:


from sklearn.metrics import classification_report

print(classification_report(
    y_test,
    y_pred,
    target_names=['Benign', 'Malignant']
))


# ## 18. Visualize the Confusion Matrix

# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    xticklabels=['Benign', 'Malignant'],
    yticklabels=['Benign', 'Malignant']
)

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Ensemble Model')
plt.show()


# ## 19. Compare Individual Models with the Ensemble Model

# In[20]:


model1.fit(X_train_scaled, y_train)
model2.fit(X_train_scaled, y_train)
model3.fit(X_train_scaled, y_train)

pred_lr = model1.predict(X_test_scaled)
pred_svm = model2.predict(X_test_scaled)
pred_rf = model3.predict(X_test_scaled)

acc_lr = accuracy_score(y_test, pred_lr)
acc_svm = accuracy_score(y_test, pred_svm)
acc_rf = accuracy_score(y_test, pred_rf)
acc_ensemble = accuracy_score(y_test, y_pred)

results = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'SVM',
        'Random Forest',
        'Voting Ensemble'
    ],
    'Accuracy': [
        acc_lr,
        acc_svm,
        acc_rf,
        acc_ensemble
    ]
})

results


# ## 20. Visualize Model Accuracy Comparison

# In[21]:


plt.figure(figsize=(9, 5))

plt.bar(results['Model'], results['Accuracy'])

plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')

plt.ylim(0, 1)
plt.xticks(rotation=20)
plt.show()


# ## 21. Display Accuracy in Percentage

# In[22]:


results['Accuracy (%)'] = results['Accuracy'] * 100

results[['Model', 'Accuracy (%)']]


# In[ ]:




