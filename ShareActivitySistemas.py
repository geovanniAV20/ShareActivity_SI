import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Cristian Geovanni Aguilar Valencia - A01066467
'''Para mi share activity decidí leer el artículo Machine learning applications in cancer prognosis and prediction, pero me pareció muy interesante
por lo que decidí realizar puramente un ejemplo para predecir si una célula de un tumor es maligna o no utilizando un dataset que
contiene información de tumores.

También leí el articulo  Using Machine Learning Algorithms for Breast Cancer Risk Prediction and Diagnosis en el cual
realizan una comparación de varias técnicas de machine learning para predecir el cancer de mama, en el cual llegan
a la conclusión de que Naive Bayes en una de las mejores técnicas para la predicción de cancer de mama y me gustaría hacerlo por mí mismo,
por ello estaré utilizando esta técnica para este ejemplo.

'''

#Se realiza la preparación de los datos.
#Con la ayuda de Pandas se carga el dataset
data = pd.read_csv('cancerTest.csv')
#Se imprime información básica para analizar el dataset.
print("\n \t El set de datos tiene {0[0]} lineas y {0[1]} columnas. \n".format(data.shape))
data.info()
data.head(3)
#Gracias a que se analizaron los datos se puede observar como se agrega una nueva columna por lo que hay que eliminarla.
data.drop(data.columns[[-1, 0]], axis=1, inplace=True)
data.info()
#Se obtiene cuantos de los registros contienen tumores maliciosos(M) y cuantos son benignos(B).
diagnostics = list(data.shape)[0]
categoriasDiagnosticos = list(data['diagnosis'].value_counts())
print("\n \t El set de datos tiene {} diagnosticos, {} malignos y {} banignos.".format(diagnostics, categoriasDiagnosticos[0], categoriasDiagnosticos[1]))
#Hay valores que no son tan utiles como otros por lo que se selaccionan las columnas mas utiles.
columnsT= list(data.columns[1:11])
#Se ultiliza un mapa de calor para ver la correlacion entre columnas utilizando Seaborn
plt.figure(figsize=(10,10))
sns.heatmap(data[columnsT].corr(), annot=True, square=True, cmap='coolwarm')
plt.show()
#Tambien se puede utilizar una matriz de dispersion para ver el comportamiento de la información. Los puntos rojos son tumores malignos y los azules son benignos
#En algunos casos los puntos azules y rojos estan en diferentes regiones de la grafica.
colorDic = {'M':'red', 'B':'blue'}
colors = data['diagnosis'].map(lambda x: colorDic.get(x))
sm = pd.scatter_matrix(data[columnsT], c=colors, alpha=0.4, figsize=((15,15)));
plt.show()
#Se seleccionan la columnas con los mejores resultados en las anteriores gráficas.
featuresSelection = ['radius_mean', 'perimeter_mean', 'area_mean', 'concavity_mean', 'concave points_mean']
#Mediante Scikit-learn se identificará si un tumor es maligno o benigno con base en las columnas seleccionadas utilizando Naive Bayes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import time
#Debido a que solo se pueden utilizar valores numericos es necesario transformar los valores de M y B en 0 y 1, respectivamente.
diagMap = {'M':1, 'B':0}
data['diagnosis'] = data['diagnosis'].map(diagMap)
#Se divide el data set de forma aleatoria en tesst y train con un test del 20% y un train de 80%
X = data.loc[:,columnsT]
y = data.loc[:, 'diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
accuracyAll = []
cvsAll = []
#Se aplica Naive bayes a todas las columnas
from sklearn.naive_bayes import GaussianNB
start = time.time()
clf = GaussianNB()
#Se entrena el modelo
clf.fit(X_train, y_train)
#Se realiza la predicción
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=5)
end = time.time()
accuracyAll.append(accuracy_score(prediction, y_test))
cvsAll.append(np.mean(scores))
#Se imprimen resultados
print("Precisión: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Puntaje de la validación: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Tiempo de ejecución: {0:.5} seconds \n".format(end-start))
#Se realiza el mismo proceso anterior pero solo con las columnas con mejores resultados
X = data.loc[:,featuresSelection]
y = data.loc[:, 'diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
accuracySelection = []
cvsSelection = []
start = time.time()
clf = GaussianNB()
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
scores = cross_val_score(clf, X, y, cv=5)
end = time.time()
accuracySelection.append(accuracy_score(prediction, y_test))
cvsSelection.append(np.mean(scores))
print("Precisión: {0:.2%}".format(accuracy_score(prediction, y_test)))
print("Puntaje de la validación: {0:.2%} (+/- {1:.2%})".format(np.mean(scores), np.std(scores)*2))
print("Tiempo de ejecución: %s seconds \n" % "{0:.5}".format(end-start))
#Se comparan ambos resultados
diffAccuracy = list(np.array(accuracySelection) - np.array(accuracyAll))
diffCvs = list(np.array(cvsSelection) - np.array(cvsAll))

d = {'Precision_Todas':accuracyAll, 'Precision_Seleccionadas':accuracySelection, 'Diferencia_Precision':diffAccuracy,
     'cvs_Todas':cvsAll, 'cvs_Seleccionadas':cvsSelection, 'Diferencia_cvs':diffCvs,}
print(d)
#Como se puede observar, si se reduce la muestra, a pesar de ser los mejores resultados, la precision se ve afectada, pero el rendimiento mejora.
'''Como conclusión y tomando como referencia los artículos que leí, pude comprobar que sí, Naive Bayes es una excelente técnica para 
predecir si un tumor es maligno o benigno, no solo para el cancer de mama, por lo menos con el dataset utilizado.
'''