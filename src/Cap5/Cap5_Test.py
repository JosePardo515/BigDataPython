import os

import joblib
import numpy as np
import pandas as pd
import sklearn
from numpy import linalg
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (calinski_harabasz_score, mean_absolute_error,
                             mean_squared_error, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn import svm


ruta_csv = f'BigDataPython/data/Cap5/titanic.csv'
ruta1_csv = f'BigDataPython/data/Cap5/titanic_ml.csv'
ruta_xls = f'BigDataPython/data/Cap5/titanic_ml.xls'
ruta_xlsx = f'BigDataPython/data/Cap5/titanic_ml.xlsx'
ruta2_xlsx = f'BigDataPython/data/Cap5/titanic_ml.xlsx'
ruta_pkl = f'BigDataPython/data/Cap5/kmeans.pkl'
ruta1_pkl = f'BigDataPython/data/Cap5/kmeans_pipeline.pkl'

# Función para partir un DataFrame en train+test y separar también la columna clase
def split_label(df, test_size, label):
    train, test = train_test_split(df, test_size=test_size)
    features = df.columns.drop(label)
    train_X = train[features]
    train_y = train[label]
    test_X = test[features]
    test_y = test[label]
    
    return train_X, train_y, test_X, test_y  

def ejemplo1():
    print("NumPy version: " + np.__version__)
    print("Pandas version: " + pd.__version__)
    print("sklearn version: " + sklearn.__version__)

def ejemplo2():
    # Ejemplo mínimo de NumPy
    v = np.array([1,2,3])
    print(v)
    print(v[1])

    m = np.array([[1,2,3],[0,1,4],[5,6,0]])
    print(m)
    print(m[2,1])

    # Multiplicación vector-matriz
    print(v @ m)

    # Inversa de una matriz
    m_inv = linalg.inv(m)
    print(m_inv)

    # Multiplicación matriz-matriz
    print(m @ m_inv)

def ejemplo3():
    df = pd.read_csv(ruta_csv)
    print (df)
    print (df.columns)
    print (df.shape)
    print (type(df.iloc))
    print (df.iloc[5])   # Fila en la posición 5
    print (df.iloc[:2])  # Filas en el rango [0,2)
    print (df.iloc[0,0]) # Celda (0,0)
    print (df.iloc[[0,10,12],3:4]) # Filas 0, 10 y 12, columnas 3:4

    # .iloc para seleccionar usando los índices
    print (type(df.loc))
    print (df.loc[0])                # Fila con índice 0
    print (df.loc[0,'Fare'])         # celda de la fila 0 y columna 'Fare'
    print (df.loc[:3, 'Sex':'Fare']) # Filas 0:3 (incluidas) en las columnas 'Sex':'Fare' (incluidas)
    print (df.loc[:3, ['Sex','Fare','Embarked']]) # Filas 0:3 (incluidas) en las columnas 'Sex','Fare' y 'Embarked'
    print (df.loc[df['Age']> 70])    # Filas con 'Age' > 70
    print (df.loc[df['Age']> 70, ['Age','Sex']])    # Filas con 'Age' > 70, mostrar la columna 'Sex'

    print (df.dtypes)
    print(df.describe(include='all'))

    # Código alternativo para calcular valores nulos y únicos

    # Valores nulos
    for c in df.columns:
        print("Missing values [{0}]:".format(c), df[c].isna().sum())
    print()

    # Valores únicos    
    for c in df.columns:
        print("Unique values [{0}]:".format(c), df[c].unique().size)

    # Elimina columnas no relevantes y filas con valores nulos
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket','Cabin'])
    df = df.dropna()

    # Traduce los valores categóricos de 'Sex' y 'Embarked' a número enteros
    df['Sex'] = df['Sex'].astype('category').cat.codes
    df['Embarked'] = df['Embarked'].astype('category').cat.codes

    print (df)

    # A formato CSV
    df.to_csv(ruta1_csv, index=False)

    '''
    # A formato Excel (XLS y XLSX)
    # La hoja se llamará 'Sheet1'
    df.to_excel(ruta_xls, index=False)
    df.to_excel(ruta_xlsx, index=False)

    # Insertar varias hojas en un fichero Excel

    writer = pd.ExcelWriter(ruta2_xlsx)
    df.to_excel(writer, sheet_name='Hoja 1', index=False)
    df.to_excel(writer, sheet_name='Hoja 2', index=False)
    writer.close()
    '''

def ejemplo4():
    # División en train (80%) y test (20%) para clasificación, con clase 'Survived'
    titanic = pd.read_csv(ruta1_csv)
    train_X, train_y, test_X, test_y = split_label(titanic, 0.2, 'Survived')

    # train_X y test_X son DataFrames
    # train_y y test_y son Series

    print (train_X, train_y, test_X, test_y)

    # Aplica el transformador OneHotEncoder a la columna 'Embarked', dejando el resto sin 
    # modificar ('passthrough'). Detecta automáticamente el número de categorías diferentes
    ohe = ColumnTransformer( [("embarked_ohe", OneHotEncoder(categories='auto'), ['Embarked'])], 
                            remainder='passthrough')
    train_X_1 = ohe.fit_transform(train_X)

    # train_X_1 es un objeto ndarray de tamaño (569, 9) y tipo float64
    print(type(train_X_1))
    print(train_X_1.shape)
    print(train_X_1.dtype)
    print(train_X_1)

    # Escalado al rango [0,1] de todos los atributos

    min_max_scaler = MinMaxScaler()
    train_X_2 = min_max_scaler.fit_transform(train_X_1)

    print(type(train_X_2))
    print(train_X_2.shape)
    print(train_X_2.dtype)

    # Muestra las 3 primeras entradas de train_X_2
    for i in range(3): 
        print(train_X_2[i])

    # Entrenamiento
    clf = SVC(gamma='scale')
    clf.fit(train_X_2, train_y)

    # Transformación del conjunto de test (one hot encoding y escalado)
    print(test_X.shape)
    test_X_2 = min_max_scaler.transform(ohe.transform(test_X))
    print(test_X_2.shape)

    # Evaluación del modelo mediante precisión
    print("Precisión sobre test:", clf.score(test_X_2, test_y)) 

    # Uso del modelo
    clf.predict(test_X_2)

    # Clasificación usando kNN
    clf = KNeighborsClassifier()
    clf.fit(train_X_2, train_y)

    # Evaluación del modelo mediante precisión
    print("Precisión sobre test:", clf.score(test_X_2, test_y)) 

# Regresión lineal
def ejemplo5():
    # separación train-test con clase 'Fare'
    titanic = pd.read_csv(ruta1_csv)
    train_X, train_y, test_X, test_y = split_label(titanic, 0.2, 'Fare')

    # one hot encoding
    index_Embarked = train_X.columns.get_loc('Embarked')
    ohe = ColumnTransformer( [("embarked_ohe", OneHotEncoder(categories='auto'), ['Embarked'])], 
                            remainder='passthrough')
    train_X_1 = ohe.fit_transform(train_X)

    # Escalado de atributos al rango [0,1]
    min_max_scaler = MinMaxScaler()
    train_X_2 = min_max_scaler.fit_transform(train_X_1)

    # Entrenamiento
    reg = LinearRegression()
    reg.fit(train_X_2, train_y)

    # Transformación del conjunto de test (one hot encoding y escalado)
    test_X_2 = min_max_scaler.transform(ohe.transform(test_X))

    # Evaluación del modelo mediante métrica R^2 y MSE
    print("R^2:", reg.score(test_X_2, test_y)) 

    # Uso y evaluación del modelo con MSE
    pred = reg.predict(test_X_2)
    print("MSE:", mean_squared_error(test_y, pred))
    print("MAE:", mean_absolute_error(test_y, pred))

    # Regresión usando kNN
    reg = LinearRegression()
    reg.fit(train_X_2, train_y)

    # Evaluación del modelo mediante métrica R^2
    print("R^2:", reg.score(test_X_2, test_y)) 

    # Uso y evaluación del modelo con MSE y MAE
    pred = reg.predict(test_X_2)
    print("MSE:", mean_squared_error(test_y, pred))
    print("MAE:", mean_absolute_error(test_y, pred))

# Clustering
def ejemplo6():
    titanic = pd.read_csv(ruta1_csv)

    # one hot encoding
    ohe = ColumnTransformer( [("embarked_ohe", OneHotEncoder(categories='auto'), ['Embarked'])], 
                            remainder='passthrough')
    titanic_1 = ohe.fit_transform(titanic)

    # Escalado de atributos al rango [0,1]
    min_max_scaler = MinMaxScaler()
    titanic_2 = min_max_scaler.fit_transform(titanic_1)

    # Clustering
    clu = KMeans(n_clusters=3)
    clu.fit(titanic_2)
    print("Centros de los clústeres:\n", clu.cluster_centers_)

    # Evaluación de los clústeres
    print('silhouette_score:', silhouette_score(titanic_2, clu.labels_))
    print('calinski_harabasz:', calinski_harabasz_score(titanic_2, clu.labels_))

    # Comprobar distancia a cada centroide para las instancias de titanic_2 (podrían ser otro conjunto de datos)
    clu.transform(titanic_2)

# Pipeline
def ejemplo7():
    ## Preprocesado

    # División en train (80%) y test (20%) para clasificación, con clase 'Survived'
    titanic = pd.read_csv(ruta1_csv)
    train_X, train_y, test_X, test_y = split_label(titanic, 0.2, 'Survived')

    # Etapa one hot encoding
    ohe = ColumnTransformer( [("embarked_ohe", OneHotEncoder(categories='auto'), ['Embarked'])], 
                            remainder='passthrough')

    # Etapa de escalado de atributos al rango [0,1]
    min_max_scaler = MinMaxScaler()

    # Etapa de clasificación
    svm = SVC(gamma='scale')

    # Creación del pipeline
    pipe = Pipeline([('ohe', ohe), ('sca', min_max_scaler), ('clf', svm)])

    # Entranamiento del pipeline
    pipe.fit(train_X, train_y)

    # Evaluación del pipeline
    print('precisión:', pipe.score(test_X, test_y))

    # Uso del modelo
    print(pipe.predict(test_X))

# Regresión
def ejemplo8():
    ## Preprocesado

    # División en train (80%) y test (20%) para clasificación, con clase 'Survived'
    titanic = pd.read_csv(ruta1_csv)
    train_X, train_y, test_X, test_y = split_label(titanic, 0.2, 'Fare')

    # Etapa de one hot encoding
    ohe = ColumnTransformer( [("embarked_ohe", OneHotEncoder(categories='auto'), ['Embarked'])], 
                            remainder='passthrough')

    # Etapa de escalado de atributos al rango [0,1]
    min_max_scaler = MinMaxScaler()

    # Etapa de regresión
    lin = LinearRegression()

    # Creación del pipeline
    pipe = Pipeline([('ohe', ohe), ('sca', min_max_scaler), ('reg', lin)])

    # Entranamiento del pipeline
    pipe.fit(train_X, train_y)

    # Evaluación R^2 del pipeline
    print('R^2:', pipe.score(test_X, test_y))

    # Uso del modelo y evaluación MSE
    pred = pipe.predict(test_X)
    print(pred)
    print('MSE:', mean_squared_error(test_y, pred))
    print("MAE:", mean_absolute_error(test_y, pred))

# Clustering
def ejemplo9():
    titanic = pd.read_csv(ruta1_csv)
    # Etapa de one hot encoding
    ohe = ColumnTransformer( [("embarked_ohe", OneHotEncoder(categories='auto'), ['Embarked'])], 
                            remainder='passthrough')

    # Etapa de escalado en rango [0,1]
    sca = MinMaxScaler()

    # Etapa de clustering
    clu = KMeans(n_clusters=3)

    # Creación del pipeline
    pipe = Pipeline([('ohe', ohe), ('sca', sca), ('clu',clu)])

    # Entrenamiento del pipeline
    pipe.fit(titanic)
    print("Centros de los clústeres:\n", pipe.named_steps['clu'].cluster_centers_)

    # Evaluación de los clústeres
    print('silhouette_score:', silhouette_score(titanic, pipe.named_steps['clu'].labels_))
    print('calinski_harabasz:', calinski_harabasz_score(titanic, pipe.named_steps['clu'].labels_))

# Persistencia de modelos
def ejemplo10():
    titanic = pd.read_csv(ruta1_csv)
    ohe = ColumnTransformer( [("embarked_ohe", OneHotEncoder(categories='auto'), ['Embarked'])], 
                            remainder='passthrough')
    titanic_1 = ohe.fit_transform(titanic)
    min_max_scaler = MinMaxScaler()
    titanic_2 = min_max_scaler.fit_transform(titanic_1)

    clu = KMeans(n_clusters=3)
    clu.fit(titanic_2)

    print("Centros de los clústeres:\n", clu.cluster_centers_)
    print('silhouette_score:', silhouette_score(titanic_2, clu.labels_))
    print('calinski_harabasz:', calinski_harabasz_score(titanic_2, clu.labels_))

    joblib.dump(clu, ruta_pkl)

    # Cargar y utilizar un modelo de clustering k-means
    loaded_clu = joblib.load(ruta_pkl) 

    print("Centros de los clústeres:\n", clu.cluster_centers_)
    print('silhouette_score:', silhouette_score(titanic_2, clu.labels_))
    print('calinski_harabasz:', calinski_harabasz_score(titanic_2, clu.labels_))
    print(clu.transform(titanic_2))

def ejemplo11():
    titanic = pd.read_csv(ruta1_csv)
    ohe = ColumnTransformer( [("embarked_ohe", OneHotEncoder(categories='auto'), ['Embarked'])], 
                            remainder='passthrough')
    sca = MinMaxScaler()
    clu = KMeans(n_clusters=3)

    pipe = Pipeline([('ohe', ohe), ('sca', sca), ('clu',clu)])
    pipe.fit(titanic)

    print("Centros de los clústeres:\n", pipe.named_steps['clu'].cluster_centers_)
    print('silhouette_score:', silhouette_score(titanic, pipe.named_steps['clu'].labels_))
    print('calinski_harabasz:', calinski_harabasz_score(titanic, pipe.named_steps['clu'].labels_))

    joblib.dump(pipe, ruta1_pkl)

    # Cargar y utilizar un pipeline de clustering k-means
    loaded_pipe = joblib.load(ruta1_pkl) 

    print("Centros de los clústeres:\n", loaded_pipe.named_steps['clu'].cluster_centers_)
    print('silhouette_score:', silhouette_score(titanic, loaded_pipe.named_steps['clu'].labels_))
    print('calinski_harabasz:', calinski_harabasz_score(titanic, loaded_pipe.named_steps['clu'].labels_))

# Optimización de hiperparámetros
def ejemplo12():

    # División en train (80%) y test (20%) para clasificación, con clase 'Survived'
    titanic = pd.read_csv(ruta1_csv)
    train_X, train_y, test_X, test_y = split_label(titanic, 0.2, 'Survived')

    ohe = ColumnTransformer( [("embarked_ohe", OneHotEncoder(categories='auto'), ['Embarked'])], 
                            remainder='passthrough')
    train_X_1 = ohe.fit_transform(train_X)

    min_max_scaler = MinMaxScaler()
    train_X_2 = min_max_scaler.fit_transform(train_X_1)

    svc = svm.SVC(gamma='scale')

    parameters = {'kernel': ['linear', 'rbf'], 'C':[1,2] }
    clf = GridSearchCV(svc, parameters, n_jobs=4, cv=3)
    clf.fit(train_X, train_y)

    print(clf.best_params_)
    print(clf.best_score_)
    print(clf.best_estimator_)

    print(clf.score(test_X, test_y))
    clf.predict(test_X)

if __name__ == '__main__':
    os.system('cls')
    Test = 'ej12'

    if Test == 'ej1':
        ejemplo1()
    if Test == 'ej2':
        ejemplo2()
    if Test == 'ej3':
        ejemplo3()
    if Test == 'ej4':
        ejemplo4()
    if Test == 'ej5':
        ejemplo5()
    if Test == 'ej6':
        ejemplo6()
    if Test == 'ej7':
        ejemplo7()
    if Test == 'ej8':
        ejemplo8()
    if Test == 'ej9':
        ejemplo9()
    if Test == 'ej10':
        ejemplo10()
    if Test == 'ej11':
        ejemplo11()
    if Test == 'ej12':
        ejemplo12()