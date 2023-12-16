#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report



# In[9]:


data = pd.read_excel('Stack Overflow 1000 Samples.xlsx')
X = data['Comment']
y = data['Tech']


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
vectorizer = TfidfVectorizer()
X_train_features = vectorizer.fit_transform(X_train)
X_val_features = vectorizer.transform(X_val)
X_test_features = vectorizer.transform(X_test)
model = SVC()
model.fit(X_train_features, y_train)
y_val_pred = model.predict(X_val_features)
precision_macro = precision_score(y_val, y_val_pred, average='macro')
recall_macro = recall_score(y_val, y_val_pred, average='macro')
f1_macro = f1_score(y_val, y_val_pred, average='macro')
precision_micro = precision_score(y_val, y_val_pred, average='micro')
recall_micro = recall_score(y_val, y_val_pred, average='micro')
f1_micro = f1_score(y_val, y_val_pred, average='micro')

print("Macro Precision:", precision_macro)
print("Macro Recall:", recall_macro)
print("Macro F1 Score:", f1_macro)
print("Micro Precision:", precision_micro)
print("Micro Recall:", recall_micro)
print("Micro F1 Score:", f1_micro)


y_test_pred = model.predict(X_test_features)
precision_macro_test = precision_score(y_test, y_test_pred, average='macro')
recall_macro_test = recall_score(y_test, y_test_pred, average='macro')
f1_macro_test = f1_score(y_test, y_test_pred, average='macro')
precision_micro_test = precision_score(y_test, y_test_pred, average='micro')
recall_micro_test = recall_score(y_test, y_test_pred, average='micro')
f1_micro_test = f1_score(y_test, y_test_pred, average='micro')


print("Macro Precision (Test):", precision_macro_test)
print("Macro Recall (Test):", recall_macro_test)
print("Macro F1 Score (Test):", f1_macro_test)

print("Micro Precision (Test):", precision_micro_test)
print("Micro Recall (Test):", recall_micro_test)
print("Micro F1 Score (Test):", f1_micro_test)


# In[10]:


data = pd.read_excel('Stack Overflow 1000 Samples.xlsx')
X = data['Comment']
y = data['Subtopic']


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
vectorizer = TfidfVectorizer()
X_train_features = vectorizer.fit_transform(X_train)
X_val_features = vectorizer.transform(X_val)
X_test_features = vectorizer.transform(X_test)
model = SVC()
model.fit(X_train_features, y_train)
y_val_pred = model.predict(X_val_features)
precision_macro = precision_score(y_val, y_val_pred, average='macro')
recall_macro = recall_score(y_val, y_val_pred, average='macro')
f1_macro = f1_score(y_val, y_val_pred, average='macro')
precision_micro = precision_score(y_val, y_val_pred, average='micro')
recall_micro = recall_score(y_val, y_val_pred, average='micro')
f1_micro = f1_score(y_val, y_val_pred, average='micro')

print("Macro Precision:", precision_macro)
print("Macro Recall:", recall_macro)
print("Macro F1 Score:", f1_macro)
print("Micro Precision:", precision_micro)
print("Micro Recall:", recall_micro)
print("Micro F1 Score:", f1_micro)


y_test_pred = model.predict(X_test_features)
precision_macro_test = precision_score(y_test, y_test_pred, average='macro')
recall_macro_test = recall_score(y_test, y_test_pred, average='macro')
f1_macro_test = f1_score(y_test, y_test_pred, average='macro')
precision_micro_test = precision_score(y_test, y_test_pred, average='micro')
recall_micro_test = recall_score(y_test, y_test_pred, average='micro')
f1_micro_test = f1_score(y_test, y_test_pred, average='micro')


print("Macro Precision (Test):", precision_macro_test)
print("Macro Recall (Test):", recall_macro_test)
print("Macro F1 Score (Test):", f1_macro_test)

print("Micro Precision (Test):", precision_micro_test)
print("Micro Recall (Test):", recall_micro_test)
print("Micro F1 Score (Test):", f1_micro_test)


# In[ ]:




