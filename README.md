<H3>Name : Prajin S</H3>
<H3>Register Number : 212223230151</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 26-08-2025</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```python
import pandas as pd
import io
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df=pd.read_csv("Churn_Modelling.csv")
print(df.isnull().sum())

df

print(df.duplicated().sum())

scaler = MinMaxScaler()
columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
df[columns] = scaler.fit_transform(df[columns])
df

X = df.iloc[:,:-1].values
X

Y = df.iloc[:,-1].values
Y

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

X_train
X_test

print("Length of X_train =",len(X_train))
print("Length of X_test =",len(X_test))
```


## OUTPUT:

<img width="290" height="324" alt="image" src="https://github.com/user-attachments/assets/002daba7-bec1-4ba2-a0eb-01a04bc17d86" />
<br>
<img width="1262" height="424" alt="image" src="https://github.com/user-attachments/assets/45400dde-ca0b-4bfc-91db-8a59263148c8" />
<br>

<img width="1268" height="435" alt="image" src="https://github.com/user-attachments/assets/6ac1f7a9-c8f3-4e87-8108-66b0e08294a7" />
<br>

<img width="685" height="176" alt="image" src="https://github.com/user-attachments/assets/297e93df-e700-43a8-af18-849a38d38e13" />
<br>

<img width="435" height="41" alt="image" src="https://github.com/user-attachments/assets/d2a67ecf-730b-4159-83e7-060d445fb7cb" />
<br>

<img width="675" height="182" alt="image" src="https://github.com/user-attachments/assets/375a96b7-e87c-42cc-b4b0-9b3b63a3109e" />
<br>

<img width="676" height="178" alt="image" src="https://github.com/user-attachments/assets/05d2c5e8-4424-401b-b953-2bf9fe54a2b0" />
<br>

<img width="269" height="29" alt="image" src="https://github.com/user-attachments/assets/ce86aace-d318-4fd0-82b0-0a521d7c074a" />
<br>

<img width="230" height="25" alt="image" src="https://github.com/user-attachments/assets/06b5c88c-cb0b-454e-825f-30424a599a78" />
<br>



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


