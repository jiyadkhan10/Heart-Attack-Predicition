#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[309]:


Blood_Pressure_data = pd.read_csv('Blood_Pressure_data.csv')
Blood_Pressure_data


# In[310]:


Blood_Pressure_data.head()


# In[311]:


Blood_Pressure_data.info()


# In[312]:


Blood_Pressure_data.describe()


# In[313]:


Blood_Pressure_data.isnull().sum()


# In[314]:


Blood_Pressure_data.drop('id', inplace = True, axis = 1)


# In[315]:


Blood_Pressure_data.drop('patient_no', inplace = True, axis = 1)


# In[316]:


Blood_Pressure_data.drop('cast', inplace = True, axis = 1)


# In[317]:


Blood_Pressure_data.drop('weight', inplace = True, axis = 1)


# In[318]:


Blood_Pressure_data.drop('payer_code', inplace = True, axis = 1)


# In[319]:


Blood_Pressure_data.drop('medical_specialty', inplace = True, axis = 1)


# In[320]:


Blood_Pressure_data.drop('examide', inplace = True, axis = 1)


# In[321]:


Blood_Pressure_data.drop('citoglipton', inplace = True, axis = 1)


# In[322]:


Blood_Pressure_data


# In[323]:


Blood_Pressure_data.info()


# In[324]:


dd = Blood_Pressure_data.drop("label", axis = 1)
y = Blood_Pressure_data["label"]
print('Shape of z: ',dd.shape)
print('Shape of y: ', y.shape)


# In[325]:


le = LabelEncoder()


# In[233]:


Blood_Pressure_data['gender'] = le.fit_transform(Blood_Pressure_data['gender'])


# In[234]:


Blood_Pressure_data['age group'] = le.fit_transform(Blood_Pressure_data['age group'])


# In[235]:


Blood_Pressure_data['max_glu_serum'] = le.fit_transform(Blood_Pressure_data['max_glu_serum'])


# In[236]:


Blood_Pressure_data['A1Cresult'] = le.fit_transform(Blood_Pressure_data['A1Cresult'])


# In[237]:


Blood_Pressure_data['metformin'] = le.fit_transform(Blood_Pressure_data['metformin'])


# In[238]:


Blood_Pressure_data['repaglinide'] = le.fit_transform(Blood_Pressure_data['repaglinide'])


# In[239]:


Blood_Pressure_data['nateglinide'] = le.fit_transform(Blood_Pressure_data['nateglinide'])


# In[240]:


Blood_Pressure_data['chlorpropamide'] = le.fit_transform(Blood_Pressure_data['chlorpropamide'])


# In[241]:


Blood_Pressure_data['glimepiride'] = le.fit_transform(Blood_Pressure_data['glimepiride'])


# In[242]:


Blood_Pressure_data['acetohexamide'] = le.fit_transform(Blood_Pressure_data['acetohexamide'])


# In[243]:


Blood_Pressure_data['glipizide'] = le.fit_transform(Blood_Pressure_data['glipizide'])


# In[244]:


Blood_Pressure_data['glyburide'] = le.fit_transform(Blood_Pressure_data['glyburide'])


# In[245]:


Blood_Pressure_data['pioglitazone'] = le.fit_transform(Blood_Pressure_data['pioglitazone'])


# In[246]:


Blood_Pressure_data['rosiglitazone'] = le.fit_transform(Blood_Pressure_data['rosiglitazone'])


# In[247]:


Blood_Pressure_data['acarbose'] = le.fit_transform(Blood_Pressure_data['acarbose'])


# In[248]:


Blood_Pressure_data['miglitol'] = le.fit_transform(Blood_Pressure_data['miglitol'])


# In[249]:


Blood_Pressure_data['tolazamide'] = le.fit_transform(Blood_Pressure_data['tolazamide'])


# In[250]:


Blood_Pressure_data['insulin'] = le.fit_transform(Blood_Pressure_data['insulin'])


# In[251]:


Blood_Pressure_data['glyburide-metformin'] = le.fit_transform(Blood_Pressure_data['glyburide-metformin'])


# In[252]:


Blood_Pressure_data['change'] = le.fit_transform(Blood_Pressure_data['change'])


# In[253]:


Blood_Pressure_data['Med'] = le.fit_transform(Blood_Pressure_data['Med'])


# In[254]:


Blood_Pressure_data['label'] = le.fit_transform(Blood_Pressure_data['label'])


# In[255]:


Blood_Pressure_data['admission_typeid'] = le.fit_transform(Blood_Pressure_data['admission_typeid'])


# In[256]:


Blood_Pressure_data['discharge_disposition_id'] = le.fit_transform(Blood_Pressure_data['discharge_disposition_id'])


# In[257]:


Blood_Pressure_data['admission_source_id'] = le.fit_transform(Blood_Pressure_data['admission_source_id'])


# In[258]:


Blood_Pressure_data['time_in_hospital'] = le.fit_transform(Blood_Pressure_data['time_in_hospital'])


# In[259]:


Blood_Pressure_data['num_lab_procedures'] = le.fit_transform(Blood_Pressure_data['num_lab_procedures'])


# In[260]:


Blood_Pressure_data['num_procedures'] = le.fit_transform(Blood_Pressure_data['num_procedures'])


# In[261]:


Blood_Pressure_data['num_medications'] = le.fit_transform(Blood_Pressure_data['num_medications'])


# In[262]:


Blood_Pressure_data['number_outpatient'] = le.fit_transform(Blood_Pressure_data['number_outpatient'])


# In[263]:


Blood_Pressure_data['number_emergency'] = le.fit_transform(Blood_Pressure_data['number_emergency'])


# In[264]:


Blood_Pressure_data['number_inpatient'] = le.fit_transform(Blood_Pressure_data['number_inpatient'])


# In[265]:


Blood_Pressure_data['number_diagnoses'] = le.fit_transform(Blood_Pressure_data['number_diagnoses'])


# In[266]:


Blood_Pressure_data['tolbutamide'] = le.fit_transform(Blood_Pressure_data['tolbutamide'])


# In[267]:


Blood_Pressure_data['troglitazone'] = le.fit_transform(Blood_Pressure_data['troglitazone'])


# In[268]:


Blood_Pressure_data['diag_1'] = le.fit_transform(Blood_Pressure_data['diag_1'])


# In[269]:


Blood_Pressure_data['diag_2'] = le.fit_transform(Blood_Pressure_data['diag_2'])


# In[270]:


Blood_Pressure_data['diag_3'] = le.fit_transform(Blood_Pressure_data['diag_3'])


# In[271]:


Blood_Pressure_data['glipizide-metformin'] = le.fit_transform(Blood_Pressure_data['glipizide-metformin'])


# In[272]:


Blood_Pressure_data['glimepiride-pioglitazone'] = le.fit_transform(Blood_Pressure_data['glimepiride-pioglitazone'])


# In[273]:


Blood_Pressure_data['metformin-rosiglitazone'] = le.fit_transform(Blood_Pressure_data['metformin-rosiglitazone'])


# In[274]:


Blood_Pressure_data['metformin-pioglitazone'] = le.fit_transform(Blood_Pressure_data['metformin-pioglitazone'])


# In[275]:


Blood_Pressure_data


# In[276]:


dd_train, dd_test, y_train, y_test = train_test_split(dd, y, test_size = 0.2, random_state = 60)
print('Shape of X_train: ', dd_train.shape)
print('Shape of y_train: ', y_train.shape)
print('Shape of X_test: ', dd_test.shape)
print('Shape of y_test: ', y_test.shape)


# In[277]:


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


# In[278]:


sc = StandardScaler()

sc.fit(dd_train)

dd_train_sc = sc.transform(dd_train)
dd_test_sc = sc.transform(dd_test)


# In[279]:


dd_test[0:5]


# In[280]:


dd_test_sc[0:5]


# In[281]:


dd_train_sc =  pd.DataFrame(dd_train_sc)
dd_test_sc =  pd.DataFrame(dd_test_sc)


# In[282]:


from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report 

clf1 = RandomForestClassifier()
clf1.fit(dd_train_sc,y_train)
pred = clf1.predict(dd_test_sc)

print ("Accuracy: " , accuracy_score(y_test,pred) * 100)  
print("Report: \n", classification_report(y_test, pred))
print("F1 Score: ", f1_score(y_test, pred, average = 'macro') * 100)


# In[283]:


from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report 

model2 = RandomForestClassifier()
model2.fit(dd_train,y_train)

predict = model2.predict(dd_test)
print("Accuracy: ", accuracy_score(y_test, predict) * 100)  
print("Report: \n", classification_report(y_test, pred))
print("F1 Score: ", f1_score(y_test, pred, average = 'macro') * 100)


# In[289]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report 

model1 = GaussianNB()
model1.fit(dd_train, y_train)
y_pred = model2.predict(dd_test)

print("Accuracy: ", accuracy_score(y_test, y_pred) * 100)  
print("Report: \n", classification_report(y_test, pred))
print("F1 Score: ", f1_score(y_test, pred, average = 'macro') * 100)


# GRAPH OF THE GIVEN DATA

# In[284]:


import matplotlib.pyplot as plt


# In[345]:


num = Blood_Pressure_data.groupby(['gender']).admission_typeid.value_counts()
num.plot(kind ='bar')


# In[344]:


num = Blood_Pressure_data.groupby(['gender']).discharge_disposition_id.value_counts()
num.plot(kind ='bar')


# In[343]:


num = Blood_Pressure_data.groupby(['gender']).admission_source_id.value_counts()
num.plot(kind ='bar')


# In[342]:


num = Blood_Pressure_data.groupby(['gender']).time_in_hospital.value_counts()
num.plot(kind ='bar')


# In[347]:


plt.figure(figsize=(16,9))
num = Blood_Pressure_data.groupby(['gender']).number_outpatient.value_counts()
num.plot(kind ='bar')


# In[348]:


plt.figure(figsize=(16,9))
num = Blood_Pressure_data.groupby(['gender']).number_inpatient.value_counts()
num.plot(kind ='bar')


# In[350]:


plt.figure(figsize=(16,9))
num = Blood_Pressure_data.groupby(['age group']).admission_typeid.value_counts()
num.plot(kind ='bar')


# In[351]:


plt.figure(figsize=(16,9))
num = Blood_Pressure_data.groupby(['age group']).admission_typeid.value_counts()
num.plot(kind ='bar')


# In[364]:


plt.figure(figsize=(32,16))
num = Blood_Pressure_data.groupby(['age group']).discharge_disposition_id.value_counts()
num.plot(kind ='bar')


# In[367]:


plt.figure(figsize=(32,16))
num = Blood_Pressure_data.groupby(['age group']).admission_source_id.value_counts()
num.plot(kind ='bar')


# In[369]:


plt.figure(figsize=(32,16))
num = Blood_Pressure_data.groupby(['age group']).time_in_hospital.value_counts()
num.plot(kind ='bar')


# In[379]:


plt.figure(figsize=(32,16))
num = Blood_Pressure_data.groupby(['age group']).number_outpatient.value_counts()
num.plot(kind ='bar')


# In[381]:


plt.figure(figsize=(32,16))
num = Blood_Pressure_data.groupby(['age group']).number_inpatient.value_counts()
num.plot(kind ='bar')


# In[382]:


num = Blood_Pressure_data.groupby(['gender']).label.value_counts()
num.plot(kind ='bar')


# In[383]:


num = Blood_Pressure_data.groupby(['age group']).label.value_counts()
num.plot(kind ='bar')


# In[384]:


num = Blood_Pressure_data.groupby(['max_glu_serum']).label.value_counts()
num.plot(kind ='bar')


# In[385]:


num = Blood_Pressure_data.groupby(['A1Cresult']).label.value_counts()
num.plot(kind ='bar')


# In[386]:


num = Blood_Pressure_data.groupby(['metformin']).label.value_counts()
num.plot(kind ='bar')


# In[387]:


num = Blood_Pressure_data.groupby(['repaglinide']).label.value_counts()
num.plot(kind ='bar')


# In[388]:


num = Blood_Pressure_data.groupby(['nateglinide']).label.value_counts()
num.plot(kind ='bar')


# In[389]:


num = Blood_Pressure_data.groupby(['chlorpropamide']).label.value_counts()
num.plot(kind ='bar')


# In[390]:


num = Blood_Pressure_data.groupby(['glimepiride']).label.value_counts()
num.plot(kind ='bar')


# In[391]:


num = Blood_Pressure_data.groupby(['acetohexamide']).label.value_counts()
num.plot(kind ='bar')


# In[392]:


num = Blood_Pressure_data.groupby(['glipizide']).label.value_counts()
num.plot(kind ='bar')


# In[393]:


num = Blood_Pressure_data.groupby(['glyburide']).label.value_counts()
num.plot(kind ='bar')


# In[394]:


num = Blood_Pressure_data.groupby(['tolbutamide']).label.value_counts()
num.plot(kind ='bar')


# In[395]:


num = Blood_Pressure_data.groupby(['pioglitazone']).label.value_counts()
num.plot(kind ='bar')


# In[396]:


num = Blood_Pressure_data.groupby(['rosiglitazone']).label.value_counts()
num.plot(kind ='bar')


# In[397]:


num = Blood_Pressure_data.groupby(['acarbose']).label.value_counts()
num.plot(kind ='bar')


# In[398]:


num = Blood_Pressure_data.groupby(['miglitol']).label.value_counts()
num.plot(kind ='bar')


# In[399]:


num = Blood_Pressure_data.groupby(['troglitazone']).label.value_counts()
num.plot(kind ='bar')


# In[400]:


num = Blood_Pressure_data.groupby(['tolazamide']).label.value_counts()
num.plot(kind ='bar')


# In[401]:


num = Blood_Pressure_data.groupby(['insulin']).label.value_counts()
num.plot(kind ='bar')


# In[402]:


num = Blood_Pressure_data.groupby(['glyburide-metformin']).label.value_counts()
num.plot(kind ='bar')


# In[403]:


num = Blood_Pressure_data.groupby(['glipizide-metformin']).label.value_counts()
num.plot(kind ='bar')


# In[404]:


num = Blood_Pressure_data.groupby(['glimepiride-pioglitazone']).label.value_counts()
num.plot(kind ='bar')


# In[405]:


num = Blood_Pressure_data.groupby(['metformin-rosiglitazone']).label.value_counts()
num.plot(kind ='bar')


# In[406]:


num = Blood_Pressure_data.groupby(['metformin-pioglitazone']).label.value_counts()
num.plot(kind ='bar')


# In[407]:


num = Blood_Pressure_data.groupby(['age group']).change.value_counts()
num.plot(kind ='bar')


# In[408]:


num = Blood_Pressure_data.groupby(['gender']).change.value_counts()
num.plot(kind ='bar')


# In[409]:


num = Blood_Pressure_data.groupby(['gender']).Med.value_counts()
num.plot(kind ='bar')


# In[410]:


num = Blood_Pressure_data.groupby(['age group']).Med.value_counts()
num.plot(kind ='bar')


# In[ ]:




