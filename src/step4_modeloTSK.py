import pandas as pd
import numpy as np
import skfuzzy as fuzzy
from sklearn.metrics import accuracy_score, classification_report

# --- Preparação dos Dados ---
centros = np.load('output/centros.npy')
treino = pd.read_csv('output/base_treino.csv')
validacao = pd.read_csv('output/base_teste_validacao.csv')

colunas_entrada = ['atributo_1', 'atributo_2', 'atributo_3', 'atributo_4', 'atributo_5', 'atributo_6']
X_treino = treino[colunas_entrada].values
y_treino = treino['classe'].values

X_val = validacao[colunas_entrada].values
y_val = validacao['classe'].values

# Parâmetros do Fuzzy C-Means (devem ser os mesmos usados na clusterização)
m_fuzzy = 1.5
tol_erro = 0.005
iter_max = 1000

print("\n Calculando os graus de pertinência (pesos) para o treino...")
# Obtemos a matriz de pertinência 'u' para saber o quanto cada amostra pertence a cada cluster
u_treino, _, _, _, _, _ = fuzzy.cluster.cmeans_predict(
    X_treino.T, centros, m=m_fuzzy, error=tol_erro, maxiter=iter_max
)
pesos_treino = u_treino.T 

print(" Calculando os coeficientes das regras (Mínimos Quadrados)...")
# Para o TS de Ordem 1, cada regra i tem uma saída: y_i = a_i0 + a_i1*x1 + ... + a_in*xn
# O modelo final é a média ponderada: y = sum(w_i * y_i)
n_amostras = X_treino.shape[0]
n_regras = centros.shape[0]
n_atributos = X_treino.shape[1]

# Construção da matriz de design para resolver o sistema linear
matriz_design = []
for k in range(n_amostras):
    linha_equacao = []
    for i in range(n_regras):
        w = pesos_treino[k, i]
        # Termo constante (intercepto) da regra i ponderado pelo peso
        linha_equacao.append(w) 
        # Termos dos atributos da regra i ponderados pelo peso
        for f in range(n_atributos):
            linha_equacao.append(w * X_treino[k, f])
    matriz_design.append(linha_equacao)

matriz_design = np.array(matriz_design)

# Resolve o sistema: Matriz_Design * Coeficientes = Saída_Real
# O vetor 'coeficientes' terá todos os a_ij organizados por regra
coeficientes, _, _, _ = np.linalg.lstsq(matriz_design, y_treino, rcond=None)

# Salvando os coeficientes para serem usados no passo de visualização (Step 6)
np.save('output/coeficientes_ts.npy', coeficientes)
print(f"   Sistema resolvido e coeficientes salvos em 'output/coeficientes_ts.npy'")

# --- Exibição das Regras ---
print("\n" + "-"*60)
print("EQUAÇÕES DO MODELO TAKAGI-SUGENO (ORDEM 1)")
print("-"*60)
ponteiro_coef = 0
for i in range(n_regras):
    desc_regra = f"REGRA {i+1}: IF (Dados no Cluster {i+1}) THEN y = {coeficientes[ponteiro_coef]:.4f}"
    ponteiro_coef += 1
    for f in range(n_atributos):
        valor_a = coeficientes[ponteiro_coef]
        desc_regra += f" + ({valor_a:.4f} * {colunas_entrada[f]})"
        ponteiro_coef += 1
    print(desc_regra)
print("-" * 60)

print("\nTestando o modelo na base de validação...")
u_val, _, _, _, _, _ = fuzzy.cluster.cmeans_predict(
    X_val.T, centros, m=m_fuzzy, error=tol_erro, maxiter=iter_max
)
pesos_val = u_val.T

saidas_continuas = []
for k in range(X_val.shape[0]):
    valor_y = 0
    idx_c = 0
    for i in range(n_regras):
        # Calcula a parte linear da regra i: y_i = a0 + a1*x1 + ...
        saida_regra_i = coeficientes[idx_c]
        idx_c += 1
        for f in range(n_atributos):
            saida_regra_i += coeficientes[idx_c] * X_val[k, f]
            idx_c += 1
        
        # Pondera pelo grau de pertinência (w)
        valor_y += pesos_val[k, i] * saida_regra_i
    saidas_continuas.append(valor_y)

saidas_continuas = np.array(saidas_continuas)

# Converte a saída contínua para classe 
predicoes_classes = np.round(saidas_continuas).astype(int)
predicoes_classes = np.clip(predicoes_classes, 1, 4) 

print("\n" + "="*60)
print("DESEMPENHO DO MODELO TAKAGI-SUGENO")
print("="*60)

acuracia = accuracy_score(y_val, predicoes_classes)
print(f"Acurácia Geral: {acuracia:.4f}")

erro_medio = np.mean((y_val - saidas_continuas)**2)
print(f"Erro Quadrático Médio (Saída Contínua): {erro_medio:.4f}")

print("\nRelatório de Classificação Detalhado:")
print(classification_report(y_val, predicoes_classes))

# Exemplo rápido para conferência
idx_exemplo = 0
print(f"\nTeste  - Amostra {idx_exemplo}:")
print(f"   Pesos dos Clusters: {np.round(pesos_val[idx_exemplo], 3)}")
print(f"   Saída do Modelo (Contínua): {saidas_continuas[idx_exemplo]:.4f}")
print(f"   Classe Predita: {predicoes_classes[idx_exemplo]} | Classe Real: {y_val[idx_exemplo]}")
