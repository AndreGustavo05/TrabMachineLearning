import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Bibliotecas de Machine Learning (Scikit-Learn)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Configuração visual dos gráficos
sns.set_theme(style="whitegrid")

# ==========================================
# 0. GERAÇÃO DO ARQUIVO DE TESTE (Caso não exista)
# ==========================================
arquivo_csv = 'telecom_chamadas_curtas.csv'

if not os.path.exists(arquivo_csv):
    print(f"⚠️ Arquivo '{arquivo_csv}' não encontrado. Gerando dados de teste automaticamente...")
    np.random.seed(42)
    tam = 5000
    naps = np.random.choice(['ASP_A', 'ASP_B', 'ASP_H', 'ASPECT_MULTI_TENANT_SISTEMA_C'], tam)
    ddds = np.random.choice(['11', '21', '31', '41', '85', '91'], tam)
    blocos = np.random.choice(['99999', '98888', '97777', '96666'], tam)
    horas = np.random.randint(8, 20, tam)
    dias = np.random.randint(1, 6, tam)
    
    decisoes = []
    for i in range(tam):
        # Injetando padrões lógicos para a IA encontrar
        if naps[i] == 'ASP_H' and ddds[i] == '91':
            decisoes.append(np.random.choice(['ALERTA_ERRO_REDE', 'SUCESSO'], p=[0.90, 0.10]))
        elif blocos[i] == '96666':
            decisoes.append(np.random.choice(['ALERTA_ERRO_REDE', 'SUCESSO'], p=[0.60, 0.40]))
        else:
            decisoes.append(np.random.choice(['ALERTA_ERRO_REDE', 'SUCESSO'], p=[0.05, 0.95]))
            
    df_teste = pd.DataFrame({
        'NAP_Origem': naps, 'Codigo_Area_DDD': ddds, 'Bloco_Operadora': blocos,
        'Hora_do_Dia': horas, 'Dia_da_Semana': dias, 'Decisao_IA': decisoes
    })
    df_teste.to_csv(arquivo_csv, index=False)
    print(f"✅ Arquivo '{arquivo_csv}' criado com sucesso!\n")

# ==========================================
# 1. CARREGAMENTO E INSPEÇÃO DOS DADOS
# ==========================================
print("--- CARREGANDO DADOS ---")
df = pd.read_csv(arquivo_csv)
# Usando .to_string() para o terminal não cortar as colunas
print(df.head().to_string())

print("\n--- INFORMAÇÕES DO DATASET ---")
df.info()

print("\n--- VALORES NULOS ---")
print(df.isnull().sum())

# ==========================================
# 2. ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
# ==========================================
print("\n📊 Gerando gráfico de distribuição... (Feche a janela do gráfico para o script continuar)")
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Decisao_IA', palette='Set2')
plt.title('Distribuição de Sucessos vs Erros de Rede', fontsize=14)
plt.ylabel('Quantidade de Chamadas')
plt.show()

# ==========================================
# 3. PREPARAÇÃO DOS DADOS E MACHINE LEARNING
# ==========================================
features = ['NAP_Origem', 'Codigo_Area_DDD', 'Bloco_Operadora', 'Hora_do_Dia', 'Dia_da_Semana']
X = df[features]
y = df['Decisao_IA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Pré-processamento Inteligente (Tradutor de Texto para Matemática)
categorical_features = ['NAP_Origem', 'Codigo_Area_DDD', 'Bloco_Operadora']
numeric_features = ['Hora_do_Dia', 'Dia_da_Semana']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Baseline (chuta a classe mais frequente)
baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)
y_pred_base = baseline.predict(X_test)

# 3.1 Treinamento de Modelos Avançados
print("\n🧠 Treinando os modelos preditivos (Aguarde)...")
modelo_rf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced'))])
modelo_tree = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced'))])
modelo_log = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000, class_weight='balanced'))])

modelo_rf.fit(X_train, y_train)
modelo_tree.fit(X_train, y_train)
modelo_log.fit(X_train, y_train)

y_pred_rf = modelo_rf.predict(X_test)
y_pred_tree = modelo_tree.predict(X_test)
y_pred_log = modelo_log.predict(X_test)

# ==========================================
# 4. AVALIAÇÃO DE MÉTRICAS E COMPARAÇÃO
# ==========================================
def avaliar_modelo(nome, y_real, y_prev):
    return {
        "Modelo": nome,
        "Acurácia (%)": round(accuracy_score(y_real, y_prev) * 100, 2)
    }

resultados = pd.DataFrame([
    avaliar_modelo("Baseline (Chute Cego)", y_test, y_pred_base),
    avaliar_modelo("Random Forest", y_test, y_pred_rf),
    avaliar_modelo("Árvore de Decisão", y_test, y_pred_tree),
    avaliar_modelo("Regressão Logística", y_test, y_pred_log)
])

print("\n--- COMPARAÇÃO DE MODELOS ---")
print(resultados.sort_values(by="Acurácia (%)", ascending=False).to_string(index=False))

# ==========================================
# 5. VISUALIZAÇÃO DE RESULTADOS (MATRIZ DE CONFUSÃO)
# ==========================================
print("\n--- RAIO-X DO MELHOR MODELO (RANDOM FOREST) ---")
print(classification_report(y_test, y_pred_rf, zero_division=0))

print("\n📈 Gerando Matriz de Confusão... (Feche a janela do gráfico para finalizar)")
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_rf, labels=modelo_rf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=modelo_rf.classes_)
disp.plot(cmap='Blues', ax=ax)
plt.title('Matriz de Confusão (Random Forest)', fontsize=12)
plt.grid(False)
plt.show()

print("\n✅ Script finalizado com sucesso!")