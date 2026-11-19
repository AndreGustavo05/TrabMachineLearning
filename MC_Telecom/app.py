import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ==========================================
# CONFIGURAÇÃO INICIAL DA PÁGINA
# ==========================================
st.set_page_config(page_title="App Analítico - Chamadas Curtas", layout="wide")
st.title("📊 Painel Preditivo: Risco de Chamadas Curtas e Falhas")
st.markdown("Faça o upload do histórico de discagem para analisar padrões de rede e treinar modelos preditivos.")

# ==========================================
# 1. UPLOAD DE DADOS
# ==========================================
uploaded_file = st.file_uploader("Envie seu arquivo de dados (.csv, .json, .xml)", type=["csv", "json", "xml"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        elif uploaded_file.name.endswith('.xml'):
            df = pd.read_xml(uploaded_file)
            
        st.success(f"Arquivo '{uploaded_file.name}' carregado com sucesso!")
        
        aba1, aba2 = st.tabs(["📈 Análise Exploratória (EDA)", "🤖 Machine Learning (Classificação)"])
        
        # ==========================================
        # ABA 1: ANÁLISE EXPLORATÓRIA
        # ==========================================
        with aba1:
            st.header("Visão Geral do Mailing/Discagem")
            st.dataframe(df.head())
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Valores Nulos por Coluna")
                st.dataframe(df.isnull().sum().reset_index().rename(columns={'index': 'Coluna', 0: 'Qtd Nulos'}))
            with col2:
                st.subheader("Estatísticas Básicas")
                st.dataframe(df.describe(include='all'))

            st.divider()
            
            # Gráfico focado na variável alvo de falhas
            st.subheader("Distribuição de Variáveis")
            var_dist = st.selectbox("Escolha uma variável para analisar:", df.columns)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            # Ajuste dinâmico: Se for texto/categoria, usa CountPlot; se for número, Histograma
            if df[var_dist].dtype == 'object' or df[var_dist].nunique() < 20:
                sns.countplot(data=df, x=var_dist, palette='viridis', ax=ax)
                plt.xticks(rotation=45)
            else:
                sns.histplot(df[var_dist], kde=True, color='#2563eb', ax=ax)
            st.pyplot(fig)

        # ==========================================
        # ABA 2: MACHINE LEARNING (CLASSIFICAÇÃO)
        # ==========================================
        with aba2:
            st.header("Treinamento de Detecção de Falhas")
            st.markdown("Configure a IA para prever a estabilidade da rota (Ex: SUCESSO vs ALERTA_ERRO_REDE).")
            
            if len(df.columns) > 1:
                col_target, col_features = st.columns(2)
                with col_target:
                    target = st.selectbox("Variável Alvo (O que prever?):", df.columns, index=len(df.columns)-1)
                with col_features:
                    features = st.multiselect("Variáveis Preditoras (Pistas):", [col for col in df.columns if col != target], default=[col for col in df.columns if col != target][:3])
                
                if features:
                    st.subheader("Configuração do Algoritmo")
                    
                    modelo_escolhido = st.selectbox(
                        "Escolha o Motor da IA:", 
                        ["Random Forest", "Árvore de Decisão", "Regressão Logística", "KNN Classifier"]
                    )
                    
                    hiperparametro = None
                    if modelo_escolhido == "Random Forest":
                        hiperparametro = st.slider("Número de Árvores (n_estimators):", 10, 100, 50)
                    elif modelo_escolhido == "Árvore de Decisão":
                        hiperparametro = st.slider("Profundidade Máxima (max_depth):", 1, 20, 10)
                    elif modelo_escolhido == "KNN Classifier":
                        hiperparametro = st.slider("Número de Vizinhos (n_neighbors):", 1, 15, 5)

                    if st.button("Treinar Modelo Agora", type="primary"):
                        X = df[features].dropna()
                        y = df[target].loc[X.index]
                        
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                        
                        # Baseline (Chuta a categoria mais frequente)
                        baseline = DummyClassifier(strategy='most_frequent')
                        baseline.fit(X_train, y_train)
                        acuracia_base = accuracy_score(y_test, baseline.predict(X_test))
                        
                        # Pipeline de Pré-processamento Automático (O Segredo para lidar com texto e número misturado)
                        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
                        categorical_features = X.select_dtypes(include=['object']).columns
                        
                        preprocessor = ColumnTransformer(
                            transformers=[
                                ('num', StandardScaler(), numeric_features),
                                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
                            ])
                        
                        if modelo_escolhido == "Random Forest":
                            motor = RandomForestClassifier(n_estimators=hiperparametro, random_state=42, class_weight='balanced')
                        elif modelo_escolhido == "Árvore de Decisão":
                            motor = DecisionTreeClassifier(max_depth=hiperparametro, random_state=42, class_weight='balanced')
                        elif modelo_escolhido == "Regressão Logística":
                            motor = LogisticRegression(class_weight='balanced', max_iter=1000)
                        elif modelo_escolhido == "KNN Classifier":
                            motor = KNeighborsClassifier(n_neighbors=hiperparametro)
                        
                        modelo = Pipeline(steps=[('preprocessor', preprocessor), ('classificador', motor)])
                        
                        modelo.fit(X_train, y_train)
                        y_pred = modelo.predict(X_test)
                        
                        acuracia_modelo = accuracy_score(y_test, y_pred)
                        
                        st.success(f"Modelo {modelo_escolhido} validado com sucesso!")
                        
                        col_res1, col_res2 = st.columns(2)
                        with col_res1:
                            st.metric("Acurácia do Modelo", f"{acuracia_modelo * 100:.2f}%")
                        with col_res2:
                            st.metric("Baseline (Chute Cego)", f"{acuracia_base * 100:.2f}%")
                            
                        st.subheader("Matriz de Confusão e Precisão")
                        st.text(classification_report(y_test, y_pred, zero_division=0))
                        
                else:
                    st.info("Selecione pelo menos uma variável preditora (X).")
            else:
                st.warning("O dataset precisa de variáveis para rodar.")
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
else:
    st.info("Aguardando o upload do arquivo para começar.")