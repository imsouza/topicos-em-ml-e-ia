#!/usr/bin/env python
# coding: utf-8

# # Importando bibliotecas

# In[19]:


import numpy as np 
import itertools
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from IPython.display import Image 


# # Fazendo o download do dataset

# In[ ]:


get_ipython().system('pip install wget')
get_ipython().system('wget https://raw.githubusercontent.com/diogocortiz/Crash-Course-IA/master/ArvoreDecis%C3%A3o/dataset_einstein.csv')


# # Importando o Dataset para o Dataframe

# In[20]:


df = pd.read_csv('dataset_einstein.csv', delimiter=';')


# # Mostrando as primeiras cinco linhas

# In[21]:


print(df.head(5))


# In[22]:


count_row = df.shape[0]  # Pegando os números de registros (linhas)
count_col = df.shape[1]  # Pegando os números de colunas

print(count_row)
print(count_col)


# É preciso deixar o dataset somente com os registros que tenham todos os campos (para evitar ruídos e distorções)

# In[23]:


df = df.dropna() # Removendo os registros nos quais pelo menos um campo está em branco (nan) 

print(df.head(5))

print('Quantidade de campos (colunas): ', df.shape[1])
print('Total de registros:', df.shape[0])


# # Verificando se o banco de dados está balanceado ou desbalanceado

# In[24]:


print ('Total de registros negativos: ', df[df['SARS-Cov-2 exam result'] =='negative'].shape[0])
print ('Total de registros positivos: ', df[df['SARS-Cov-2 exam result'] =='positive'].shape[0])


# É necessário converter o Dataframe para um Array Numpy, que é o tipo de dados que será usado no treinamento. Também é necessário separar o Dataset em dois. Um com as features de entrada, e outro com os labels (etiquetas, rótulos do registro).   
# 
# Neste caso, é esperado criar um classificador para o teste do Covid, ou seja, o objetivo é treinar o modelo com a etiqueta presente no campo 'SARS-Cov-2 exam result'

# In[26]:


# Jogando as etiquetas para Y
Y = df['SARS-Cov-2 exam result'].values # Converte o dataframe do pandas pra um array do numpy
print(Y)

# X será a matriz com as features
# Campos de treinamento: (Hemoglobin, Leukocytes, Basophils, Proteina C reativa mg/dL)
X = df[['Hemoglobin', 'Leukocytes', 'Basophils','Proteina C reativa mg/dL']].values

# Exibindo X
print(X)



# Divisão do Dataset em dois: um para o treino (80% dos dados) e outro para o teste (20% dos dados)

# In[28]:


X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.2, random_state=3)


# # Criando e treinando o modelo

# O algortimo de treinamento (que neste caso é o de árvore de decisão) irá exportar um modelo treinado (que também é um algoritmo).

# In[29]:


# Cria um algortimo que será do tipo 'Árvore de Decisão'
algortimo_arvore = DecisionTreeClassifier(criterion='entropy', max_depth=5)

# Treinando o algoritmo
modelo = algortimo_arvore.fit(X_treino, Y_treino)


# A árvore de decisão pode ser considerada um modelo White Box, ou seja, um modelo que podemos entender melhor o que ele aprendeu e como ele decide.

# In[31]:


# Exibindo a feature mais importante (white box?)
print(modelo.feature_importances_)

nome_features = ['Hemoglobin', 'Leukocytes', 'Basophils','Proteina C reativa mg/dL']
nome_classes = modelo.classes_

# Montando a imagem da árvore
dot_data = StringIO()
#dot_data = tree.export_graphviz(my_tree_one, out_file=None, feature_names=featureNames)
export_graphviz(modelo, out_file=dot_data, filled=True, feature_names=nome_features, class_names=nome_classes, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())
graph.write_png("arvore.png")
Image('arvore.png')


# É possível entender também quais as features de maior importância para o modelo treinado

# In[32]:


importances = modelo.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
f, ax = plt.subplots(figsize=(11, 9))
plt.title("Feature ranking", fontsize = 20)
plt.bar(range(X.shape[1]), importances[indices], color="b", align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.ylabel("importance", fontsize = 18)
plt.xlabel("index of the feature", fontsize = 18)
plt.show()

# Indice das features
# 0 - 'Hemoglobin', 
# 1 - 'Leukocytes'
# 2 - 'Basophils',
# 3 - 'Proteina C reativa mg/dL']


# # Testando o modelo e fazendo predições no dataset de teste

# In[35]:


# Aplicando o modelo na base de testes e armazendo o resultado em y_predicoes
Y_predicoes = modelo.predict(X_teste)

# Avaliação do modelo: avalia o valor real do dataset y_teste com as predições
print("ACURÁCIA DA ÁRVORE: ", accuracy_score(Y_teste, Y_predicoes))
print(classification_report(Y_teste, Y_predicoes))

# Precisão: das classificações que o modelo fez para uma determinada classe, quantas efetivamente eram corretas?
# Recall: dos possíveis datapoints pertecentes a uma determinada classe, quantos o modelo conseguiu classificar corretamente?


# # Matriz de Confusão

# In[37]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de Confusão Normalizada")
    else:
        print('Matriz de Confusão sem normalizacão ')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Rótulo real')
    plt.xlabel('Rótulo prevista')

matrix_confusao = confusion_matrix(Y_teste, Y_predicoes)
plt.figure()
plot_confusion_matrix(matrix_confusao, classes=nome_classes,
                      title='Matrix de Confusão')

