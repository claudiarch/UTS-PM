#!/usr/bin/env python
# coding: utf-8

# ## Import Library

# In[1]:


#202431160_CLAUDIA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_auc_score,
    mean_squared_error, r2_score, ConfusionMatrixDisplay
)
pd.set_option('display.max_columns', None)

RANDOM_STATE = 42


# ## Dataset

# In[3]:


#202431160_CLAUDIA 
df = pd.read_csv('02_diabetes.csv')
print('Ukuran dataset:', df.shape)
display(df.head())
display(df.info())


# ## EDA

# In[4]:


#202431160_CLAUDIA 
print('Tipe data:')
display(df.dtypes.to_frame('Tipe Data'))
print('Jumlah missing value per kolom:')
display(df.isna().sum().to_frame('Missing Value'))
print('Jumlah duplikasi:', df.duplicated().sum())
print('Distribusi target diabetes:')
display(df['diabetes'].value_counts().rename(index={0:'Tidak diabetes', 1:'Diabetes'}).to_frame('Jumlah'))
display(df.describe(include='all').T)


# In[5]:


#202431160_CLAUDIA
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
counts = df['diabetes'].value_counts().sort_index()
axes[0].bar(['Tidak diabetes', 'Diabetes'], counts.values)
axes[0].set_title('Distribusi Target Diabetes')
axes[0].set_ylabel('Jumlah')
axes[1].hist(df.loc[df['diabetes'] == 0, 'age'], bins=25, alpha=0.6, label='Tidak diabetes')
axes[1].hist(df.loc[df['diabetes'] == 1, 'age'], bins=25, alpha=0.6, label='Diabetes')
axes[1].set_title('Distribusi Usia berdasarkan Target')
axes[1].legend()
plt.tight_layout()
plt.show()


# ## Preprocessing

# In[6]:


#202431160_CLAUDIA 
# Target dipisahkan dari fitur.
X = df.drop(columns='diabetes')
y = df['diabetes']

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('numerik', StandardScaler(), numerical_features),
        ('kategori', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print('Fitur numerik:', numerical_features)
print('Fitur kategorikal:', categorical_features)
print('Ukuran X_train:', X_train.shape, '| X_test:', X_test.shape)
print('Proporsi target pada data penuh:')
display(y.value_counts(normalize=True).rename('Proporsi').to_frame())


# ## Fungsi Evaluasi

# In[8]:


#202431160_CLAUDIA 
def evaluate_classifier(name, y_true, y_pred, scores=None):
    result = {
        'Algoritma': name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-score': f1_score(y_true, y_pred, zero_division=0)
    }
    if scores is not None:
        result['ROC-AUC'] = roc_auc_score(y_true, scores)
    return result


# ## K-Nearest Neighbors (KNN)

# In[10]:


#202431160_CLAUDIA
knn_model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', KNeighborsClassifier(n_neighbors=5, weights='distance'))
])
knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_scores = knn_model.predict_proba(X_test)[:, 1]

knn_result = evaluate_classifier('KNN (k=5)', y_test, knn_pred, knn_scores)
print('Hasil evaluasi KNN:')
display(pd.DataFrame([knn_result]).set_index('Algoritma'))
print(classification_report(y_test, knn_pred, target_names=['Tidak diabetes', 'Diabetes'], zero_division=0))
cm = confusion_matrix(y_test, knn_pred)
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion Matrix — KNN')
plt.colorbar()
plt.xticks([0, 1], ['Prediksi 0', 'Prediksi 1'])
plt.yticks([0, 1], ['Aktual 0', 'Aktual 1'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center')
plt.tight_layout()
plt.show()


# ## Linear Regression

# In[11]:


# WATERMARK: 202431160 - CLAUDIA SHERLY FEBRIANA DEWI
linear_model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('model', LinearRegression())
])
linear_model.fit(X_train, y_train)
linear_scores = linear_model.predict(X_test)
linear_pred = (linear_scores >= 0.5).astype(int)

linear_result = evaluate_classifier('Linear Regression (threshold=0.5)', y_test, linear_pred, linear_scores)
linear_result['RMSE'] = mean_squared_error(y_test, linear_scores) ** 0.5
linear_result['R2'] = r2_score(y_test, linear_scores)
print('Hasil evaluasi Linear Regression:')
display(pd.DataFrame([linear_result]).set_index('Algoritma'))
print(classification_report(y_test, linear_pred, target_names=['Tidak diabetes', 'Diabetes'], zero_division=0))
cm = confusion_matrix(y_test, linear_pred)
plt.imshow(cm, interpolation='nearest')
plt.title('Confusion Matrix — Linear Regression (threshold 0.5)')
plt.colorbar()
plt.xticks([0, 1], ['Prediksi 0', 'Prediksi 1'])
plt.yticks([0, 1], ['Aktual 0', 'Aktual 1'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center')
plt.tight_layout()
plt.show()


# ## Perbandingan Model

# In[13]:


#202431160_CLAUDIA
comparison = pd.DataFrame([knn_result, linear_result]).set_index('Algoritma')
display(comparison.round(4))
comparison[['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC']].plot(kind='bar', figsize=(12, 5))
plt.title('Perbandingan Kinerja Model')
plt.ylabel('Skor')
plt.ylim(0, 1)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.show()


# In[ ]:




