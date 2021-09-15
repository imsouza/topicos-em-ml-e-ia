#!/usr/bin/env python
# coding: utf-8

# # Importação das bibliotecas

# In[ ]:


from matplotlib import pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from math import sqrt


# # Download do Dataset

# In[ ]:


get_ipython().system('pip3 install wget')
get_ipython().system('wget https://raw.githubusercontent.com/diogocortiz/Crash-Course-IA/master/RegressaoLinear/FuelConsumptionCo2.csv')


# # Carregar o dataset para um Dataframe (Pandas)

# In[ ]:


# Cria um dataset chamado 'df' que receberá os dados do csv
df = pd.read_csv("FuelConsumptionCo2.csv")

#EXIBE A ESTRUTURA DO DATAFRAME
print(df.head())


# # Exibe o resumo do Dataset

# In[ ]:


print(df.describe())


# # Selecionar apenas as features do Motor e CO2

# In[ ]:


motores = df[['ENGINESIZE']]
co2 = df[['CO2EMISSIONS']]
print(motores.head())


# # Dividir o dataset em dados de treinamento e dados de teste (20%) neste casos vamos usar o train_test_split do scikitlearn

# In[ ]:


motores_treino, motores_test, co2_treino, co2_teste = train_test_split(motores, co2, test_size=0.2, random_state=42)
print(type(motores_treino))


# # Exibir a correlação entre as features do dataset de treinamento

# In[ ]:


plt.scatter(motores_treino, co2_treino, color='blue')
plt.xlabel("Motor")
plt.ylabel("Emissão de CO2")
plt.show()


# # Treinar o modelo de regressão linear

# In[ ]:


# cria um modelo do tipo Regressão Linear
modelo =  linear_model.LinearRegression()

# Treina o modelo usando o dataset de teste
# para encontrar o valor de A E B (Y = A + B.X)
# aplicando o método dos mínimos quadrados
modelo.fit(motores_treino, co2_treino)


# # Exibir os coeficientes (A e B)

# In[ ]:


print('(A) Intercepto: ', modelo.intercept_) # de onde a reta irá partir do eixo Y
print('(B) Inclinação: ', modelo.coef_) # inclinação da reta


# # Exibir a reta de regressão no dataset de treino

# In[ ]:


plt.scatter(motores_treino, co2_treino, color='blue')
plt.plot(motores_treino, modelo.coef_[0][0] * motores_treino + modelo.intercept_[0], '-r')
plt.ylabel("Emissão de C02")
plt.xlabel("Motores")
plt.show()


# # Executar o modelo no dataset de teste

# In[ ]:


# Primeiro será feito as predições usando o modelo e base de teste
predicoesCo2 = modelo.predict(motores_test)


# # Exibir a reta de regressão no dataset de teste

# In[ ]:


plt.scatter(motores_test, co2_teste, color='blue')
plt.plot(motores_test, modelo.coef_[0][0] * motores_test + modelo.intercept_[0], '-r')
plt.ylabel("Emissão de C02")
plt.xlabel("Motores")
plt.show()


# # Avaliar o modelo

# In[ ]:


# Exibindo métricas de erros
print("Soma dos Erros ao Quadrado (SSE): %2.f " % np.sum((predicoesCo2 - co2_teste) ** 2))
print("Erro Quadrático Médio (MSE): %.2f" % mean_squared_error(co2_teste, predicoesCo2))
print("Erro Médio Absoluto (MAE): %.2f" % mean_absolute_error(co2_teste, predicoesCo2))
print("Raiz do Erro Quadrático Médio (RMSE): %.2f " % sqrt(mean_squared_error(co2_teste, predicoesCo2)))
print("R2-score: %.2f" % r2_score(predicoesCo2 , co2_teste) )

