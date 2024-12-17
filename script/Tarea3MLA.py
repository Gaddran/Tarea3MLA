#!/usr/bin/env python
# coding: utf-8

# <div>
# <img src="https://i.ibb.co/v3CvVz9/udd-short.png" width="150"/>
#     <br>
#     <strong>Universidad del Desarrollo</strong><br>
#     <em>Magíster en Data Science</em><br>
#     <em>Profesor: Tomás Fontecilla </em><br>
# 
# </div>
# 
# # Machine Learning Avanzado
# *17 de Diciembre de 2024*
# 
# #### Integrantes: 
# `Jocelyn Cáceres, Kurt Castro, Giuseppe Lavarello, Carlos Saquel`

# 1) LSTM producción de leche
#     - Preparación de Datos
#     - Visualización serie de tiempo
#     - Análisis serie de tiempo? tendencia, estacionalidad, etc
#     - Modelo LSTM
#         - Separar datos (train, test, ¿val?)
#         - Diseño arquitectura modelo
#         - Entrenar modelo
#         - Validacion y calculo de metricas
#         - predicciones (pide 3 meses)
#             - Visualizar predicciones
#             - Comparar resultados
# 
# 2) LSTM producción IPSA
#     - Obtencion de datos (probablemente directo de la pagina de la bolsa de santiago)
#     - Preparación de Datos
#     - Visualización serie de tiempo
#     - Análisis serie de tiempo?
#     - Modelo LSTM
#         - Separar datos (train, test, ¿val?)
#         - Diseño arquitectura modelo
#         - Entrenar modelo
#         - Validacion y calculo de metricas
#         - predicciones (pide 3 meses)
#             - Visualizar predicciones
#             - Comparar resultados
# 3) Autoencoder, limpieza de imágenes y extracción de texto
#     - Preparación de Datos
#     - Visualización de muestra de datos
#     - Preprosesamiento de imagenes
#     - Autoencoder
#         - Separar imagenes (train, test, val)
#         - Diseño arquitectura modelo
#         - Entrenar modelo
#         - Limpiar imagenes y visualizar comparacion de 2 o 3 ejemplos
#         - Extraer texto y validar 
#         - Usar imagen propia "mostrar imagen inicial y después de correr el modelo"
# 
# 4) Conclusiones y dar formato de Informe
#     - Ejemplo: 
#         1. Objetivo
#         2. Introducción
#         3. Metodología 
#             - Aca el codigo y las validaciones
#         4. Conclusiones    
# 

# 2) LSTM producción IPSA
#     - Obtencion de datos (probablemente directo de la pagina de la bolsa de santiago)
#     - Preparación de Datos
#     - Visualización serie de tiempo
#     - Análisis serie de tiempo?
#     - Modelo LSTM
#         - Separar datos (train, test, ¿val?)
#         - Diseño arquitectura modelo
#         - Entrenar modelo
#         - Validacion y calculo de metricas
#         - predicciones (pide 3 meses)
#             - Visualizar predicciones
#             - Comparar resultados

# ### Importe de paquetes

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ### Preparación y Carga de la Data

df_IPSA = pd.read_excel(r'./data/Cuadro_13122024183130.xlsx', header=2)


df_IPSA.head()


df_IPSA.info()


# **Importante** No hay valores nulos que necesiten evaluación.

# ### Visualización de la data

# Seteo de fecha como indice
df_IPSA.set_index('Periodo', inplace=True)

# renombre de columna
df_IPSA.rename(columns={'1.IPSA  (índice enero 2003=1000)                      ':'IPSA'}, inplace=True)

# Escalamiento para modelo
scaler = MinMaxScaler(feature_range=(0, 1))
df_IPSA['Escalado'] = scaler.fit_transform(df_IPSA[['IPSA']])
df_IPSA.plot(y='IPSA')


# ### Análisis de la serie de tiempo

# La data presenta una clara tendencia alcista, pero también muestra una volatilidad significativa, que podría estar relacionada con eventos mundiales como la crisis de 2008, el estallido social y la pandemia de COVID-19, considerando los períodos en los que se observan cambios abruptos.  
# Si bien no hay una estacionalidad inmediatamente evidente, podrían existir patrones anuales al analizar el espectro diferencial.
# 

# ### Elección de parametros y diseño del modelo 

retrasos = 3 #Numero de columnas de retrazo a crear

for i in range(1, retrasos + 1):
    df_IPSA[f'lag_{i}'] = df_IPSA['Escalado'].shift(i)
    
df_IPSA['Objetivo'] = df_IPSA['Escalado']
df_IPSA.dropna(inplace=True)  # Drop Nans


X = df_IPSA[[col for col in df_IPSA.columns if col.startswith('lag')]].values
y = df_IPSA['Objetivo'].values


# Se le da el formato requerido por el modelo LSTM a los datos.  
# 

X = X.reshape((X.shape[0], X.shape[1], 1))


# Se separan los datos en entrenamiento y testeo

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) #Shuffle false es importante para mantener el orden de la serie de tiempo


# Diseñamos la arquitectura del modelo; en este caso, es un modelo LSTM simple.  
# 

# Crear el LSTM modelo
model = Sequential()
model.add(Input((X_train.shape[1], 1)))
model.add(LSTM(units=500, activation='relu', return_sequences=False))
model.add(Dense(units=1))

# Compilar el Modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Resumen de modelo
model.summary()


history = model.fit(X_train, y_train, epochs=500, batch_size=256, verbose=0)


# Graficar la pérdida de entrenamiento y validación
plt.plot(history.history['loss'])


# Añadir etiquetas y título
plt.title('Pérdida del Modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.show()


# ### Validación y cálculo de métricas.  
# 

predicciones = model.predict(X_test)


predicciones_reescalado = scaler.inverse_transform(predicciones)
y_test_rescalado = scaler.inverse_transform(y_test.reshape(-1, 1))

dates = list(df_IPSA[int(len(df_IPSA)*0.8):].index)

plt.xticks(rotation=45)
plt.title('Comparación entre valores reales y predichos')
plt.plot(dates,y_test_rescalado, label='Real')
plt.plot(dates,predicciones_reescalado, label='Predicho')
plt.legend()
plt.show()


mae_IPSA = mean_absolute_error(y_test_rescalado,predicciones_reescalado)
mse_IPSA = mean_squared_error(y_test_rescalado,predicciones_reescalado)
r2_IPSA = r2_score(y_test_rescalado,predicciones_reescalado)

n = len(y_test)
p = X_train.shape[1] # numero de features (en este caso lags)

adjusted_r2_IPSA = 1 - ((1 - r2_IPSA) * (n - 1)) / (n - p - 1)

print(f"Mean Absolute Error (MAE): {mae_IPSA:.1f}")
print(f"Mean Squared Error (MSE): {mse_IPSA:.1f}")
print(f"Root Mean Squared Error (RMSE): {mse_IPSA**0.5:.1f}")
print(f"R² ajustado: {adjusted_r2_IPSA:.3f}")


#Obtener el valor del ultimo dato y darle formato que espera el modelo
last_data = df_IPSA[[col for col in df_IPSA.columns if col.startswith('lag')]].iloc[-1:].values.reshape(1, X.shape[1], 1) 


#prediccion de los siguientes n meses
n_meses = 3
prediccion_n_meses = []
for i in range(n_meses):
    predicciones2 = model.predict(last_data)
    last_data = np.roll(last_data, -1)
    last_data[0][-1] = predicciones2
    prediccion_n_meses.append(predicciones2[0])

print("Los valores de los proximos 3 meses seran:", scaler.inverse_transform(prediccion_n_meses).flatten())


dates_extended = pd.date_range(dates[-1] + pd.DateOffset(months=1), periods=3, freq='MS')
dates_extended = [pd.Timestamp(date) for date in dates_extended]
dates_extended=dates+dates_extended


plt.plot(dates_extended,scaler.inverse_transform(np.concat((predicciones,last_data[0]))), label='Predicho extendido', )
plt.plot(dates,scaler.inverse_transform((predicciones)), label='Predicho con test')

plt.legend()
plt.title("Valores predichos")
plt.xticks(rotation=45)
plt.show()


# ## Produccion de Leche

# ### Preparación y Carga de la Data

df_leche = pd.read_csv(r'./data/monthly_milk_production.csv')


df_leche.head()


df_leche.tail()


df_leche.info()


# **Importante** No hay valores nulos que necesiten evaluación.

# ### Visualización de la data

df_leche['Date'] = pd.to_datetime(df_leche['Date'])
# Renombre de columnas
df_leche.rename(columns={'Production': 'Produccion', 'Date': 'Fecha'}, inplace=True)

# Seteo de fecha como indice
df_leche.set_index('Fecha', inplace=True)


# Escalamiento para modelo
scaler = MinMaxScaler(feature_range=(0, 1))
df_leche['Escalado'] = scaler.fit_transform(df_leche[['Produccion']])



df_leche.plot(y='Produccion', title='Producción de leche entre 1962 y 1974', rot=45, ylabel='Libras' )


# ### Análisis de la serie de tiempo

# La data presenta una clara tendencia creciente antes de 1971 y parece estabilizarse desde entonces. Además, se puede observar que existe una fuerte componente estacional, que domina la producción dentro de cada año.
# 

# ### Elección de parametros y diseño del modelo 

retrasos = 7 #Numero de columnas de retrazo a crear

for i in range(1, retrasos + 1):
    df_leche[f'lag_{i}'] = df_leche['Escalado'].shift(i)
    
df_leche['Objetivo'] = df_leche['Escalado']
df_leche.dropna(inplace=True)  # Drop Nans


X_leche = df_leche[[col for col in df_leche.columns if col.startswith('lag')]].values
y_leche = df_leche['Objetivo'].values


# Se le da el formato requerido por el modelo LSTM a los datos.  
# 

X_leche = X_leche.reshape((X_leche.shape[0], X_leche.shape[1], 1))


# Se separan los datos en entrenamiento y testeo

X_leche_train, X_leche_test, y_leche_train, y_leche_test = train_test_split(X_leche, y_leche, test_size=0.3, shuffle=False, random_state=123) #Shuffle false es importante para mantener el orden de la serie de tiempo


# Diseñamos la arquitectura del modelo; en este caso, es un modelo LSTM simple.  
# 

# Crear el LSTM modelo
model = Sequential()
model.add(Input((X_leche_train.shape[1], 1)))
model.add(LSTM(units=500, activation='relu', return_sequences=True))
model.add(LSTM(units=500, activation='relu', return_sequences=False))
model.add(Dense(units=1))

# Compilar el Modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Resumen de modelo
model.summary()


history_leche = model.fit(X_leche_train, y_leche_train, epochs=500, batch_size=256, verbose=0)


# Graficar la pérdida de entrenamiento y validación
plt.plot(history_leche.history['loss'])


# Añadir etiquetas y título
plt.title('Pérdida del Modelo')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.show()


# ### Validación y cálculo de métricas.  
# 

predicciones_leche = model.predict(X_leche_test)


predicciones_leche_reescalado = scaler.inverse_transform(predicciones_leche)
y_leche_test_rescalado = scaler.inverse_transform(y_leche_test.reshape(-1, 1))

dates_leche = list(df_leche[int(len(df_leche)*0.7):].index)

fig, ax = plt.subplots()

plt.xticks(rotation=45)
plt.title('Comparación entre valores reales y predichos')
plt.plot(dates_leche,y_leche_test_rescalado, label='Real')
plt.plot(dates_leche,predicciones_leche_reescalado, label='Predicho')
plt.legend()

plt.show()


mae_leche = mean_absolute_error(y_leche_test_rescalado,predicciones_leche_reescalado)
mse_leche = mean_squared_error(y_leche_test_rescalado,predicciones_leche_reescalado)
r2_leche = r2_score(y_leche_test_rescalado,predicciones_leche_reescalado)

n = len(y_leche_test)
p = X_leche_train.shape[1] # numero de features (en este caso lags)

adjusted_r2_leche = 1 - ((1 - r2_leche) * (n - 1)) / (n - p - 1)

print(f"Mean Absolute Error (MAE): {mae_leche:.1f}")
print(f"Mean Squared Error (MSE): {mse_leche:.1f}")
print(f"Root Mean Squared Error (RMSE): {mse_leche**0.5:.1f}")
print(f"R² ajustado: {adjusted_r2_leche:.3f}")


#Obtener el valor del ultimo dato y darle formato que espera el modelo
last_data_leche = df_leche[[col for col in df_leche.columns if col.startswith('lag')]].iloc[-1:].values.reshape(1, X_leche.shape[1], 1) 


#prediccion de los siguientes n meses
n_meses = 3
prediccion_n_meses_leche = []
for i in range(n_meses):
    predicciones2_leche = model.predict(last_data_leche)
    last_data_leche = np.roll(last_data_leche, -1)
    last_data_leche[0][-1] = predicciones2_leche
    prediccion_n_meses_leche.append(predicciones2_leche[0])

print("Los valores de los proximos 3 meses seran:", scaler.inverse_transform(prediccion_n_meses_leche).flatten())


dates_extended = pd.date_range(dates_leche[-1] + pd.DateOffset(months=1), periods=3, freq='MS')
dates_extended = [pd.Timestamp(date) for date in dates_extended]
dates_extended=dates_leche+dates_extended


prediccion_n_meses_leche[0]


plt.plot(dates_extended,scaler.inverse_transform(np.concat((predicciones_leche,prediccion_n_meses_leche))), label='Predicho extendido', )
plt.plot(dates_leche,predicciones_leche_reescalado, label='Predicho con test')

plt.legend()
plt.title("Valores predichos")
plt.xticks(rotation=45)
plt.show()

