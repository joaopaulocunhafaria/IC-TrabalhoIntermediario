import pandas as pd
import numpy as np
import skfuzzy as fuzzy
from sklearn.metrics import accuracy_score, classification_report


def s_norm_maximo(a, b):
    """
    S-Norma máximo. 
    Preserva a força da regra que disparou com maior intensidade para a classe
    """
    return np.maximum(a, b)


# Carregamento de dados
centros = np.load('output/centros.npy')
treino = pd.read_csv('output/base_treino.csv')
validacao = pd.read_csv('output/base_teste_validacao.csv')

colunas_entrada = ['atributo_1', 'atributo_2', 'atributo_3', 'atributo_4', 'atributo_5', 'atributo_6']
X_treino = treino[colunas_entrada].values
y_treino = treino['classe'].values

X_val = validacao[colunas_entrada].values
y_val = validacao['classe'].values

# constantes
m_fuzzy = 1.5
tol_erro = 0.005
iter_max = 1000


print("\nConstruindo a Base de Regras")
# Obtemos as pertinências para o conjunto de treino
u_treino, _, _, _, _, _ = fuzzy.cluster.cmeans_predict(
    X_treino.T, centros, m=m_fuzzy, error=tol_erro, maxiter=iter_max
)
pesos_treino = u_treino.T  # Pertinências das amostras aos clusters

n_clusters = centros.shape[0]
classes_unicas = np.unique(y_treino)

# Dicionário que mapeará cada Cluster (Regra) para uma Classe Consequente
regra_para_classe = {}

for i in range(n_clusters):

    # Encontra as amostras que têm maior pertinência a este cluster
    amostras_do_cluster = np.argmax(pesos_treino, axis=1) == i
    
    if np.any(amostras_do_cluster):
        # A classe que mais aparece dentro deste cluster define o consequente da regra
        classes_no_cluster = y_treino[amostras_do_cluster]
        classe_predominante = np.bincount(classes_no_cluster).argmax()
        regra_para_classe[i] = classe_predominante
    else:
        regra_para_classe[i] = classes_unicas[0]
        
    print(f"REGRA {i+1}: SE (x é Cluster {i+1}) ENTÃO (Classe é {regra_para_classe[i]})")



# composição max-min e inferência na validação 
print("\n Executando Inferência Mamdani")
# Obtemos as pertinências para o conjunto de validação
u_val, _, _, _, _, _ = fuzzy.cluster.cmeans_predict(
    X_val.T, centros, m=m_fuzzy, error=tol_erro, maxiter=iter_max
)
pesos_val = u_val.T # Força de disparo de cada regra para as amostras de validação

n_amostras_val = X_val.shape[0]

# Matriz para armazenar o grau de pertinência final de cada amostra a cada classe
pertinencia_final_classes = np.zeros((n_amostras_val, np.max(classes_unicas) + 1))

# Processo de Implicação e Agregação
for i in range(n_clusters):
    classe_consequente = regra_para_classe[i]
    forca_disparo_regra = pesos_val[:, i] # \alpha_i (Composição Min já implícita via FCM)
    
    # Agregação via S-Norma máximo
    pertinencia_atual = pertinencia_final_classes[:, classe_consequente]
    
    pertinencia_final_classes[:, classe_consequente] = s_norm_maximo(
        pertinencia_atual, 
        forca_disparo_regra
    )


# Defuzzificação por Máximo
predicoes_finais = np.argmax(pertinencia_final_classes[:, 1:], axis=1) + 1


print("\n" + "="*60)
print("AVALIAÇÃO DO MODELO MAMDANI (S-NORMA MÁXIMO)")
print("="*60)

acuracia = accuracy_score(y_val, predicoes_finais)
print(f"Acurácia: {acuracia:.4f}")

print("\nRelatório de Classificação:")
print(classification_report(y_val, predicoes_finais))

# Exemplo detalhado de composição para a primeira amostra
amostra_idx = 0
print(f"\nExemplo de Composição para a amostra {amostra_idx}:")
for i in range(n_clusters):
    print(f"  Força da Regra {i+1} (-> Classe {regra_para_classe[i]}): {pesos_val[amostra_idx, i]:.4f}")
print(f"Pertinências Finais Agregadas (Classes 1 a 4): {np.round(pertinencia_final_classes[amostra_idx, 1:], 4)}")
print(f"Classe Predita (Defuzzificação Max): {predicoes_finais[amostra_idx]}")
print(f"Classe Real: {y_val[amostra_idx]}")
