import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from collections import Counter
import re
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
from io import BytesIO
import base64

# Configuração da página
st.set_page_config(
    page_title="Reddit Clustering Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def preprocessar_texto(texto):
    """Limpa e preprocessa o texto"""
    if pd.isna(texto):
        return ""

    texto = str(texto).lower()
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()

    return texto


class KMeansAnimado:
    """Classe KMeans com capacidade de gerar animações"""

    def __init__(self, n_clusters=3, max_iter=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, random_state=random_state)
        self.historico_centroides = []
        self.historico_labels = []
        self.X_data = None

    def fit(self, X):
        """Treina o modelo e coleta histórico para animação"""
        self.X_data = X.copy()

        # Inicialização dos centroides
        np.random.seed(self.random_state)
        centroides = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        self.historico_centroides.append(centroides.copy())

        for iteracao in range(self.max_iter):
            # Atribuir pontos aos clusters
            distancias = np.sqrt(((X - centroides[:, np.newaxis]) ** 2).sum(axis=2))
            labels = np.argmin(distancias, axis=0)
            self.historico_labels.append(labels.copy())

            # Atualizar centroides
            novos_centroides = centroides.copy()
            for k in range(self.n_clusters):
                if np.sum(labels == k) > 0:
                    novos_centroides[k] = X[labels == k].mean(axis=0)

            self.historico_centroides.append(novos_centroides.copy())

            # Verificar convergência
            if np.allclose(centroides, novos_centroides, rtol=1e-4):
                break

            centroides = novos_centroides

        # Treinar o modelo final do scikit-learn
        self.kmeans.fit(X)
        return self

    def predict(self, X):
        """Predição usando o modelo do scikit-learn"""
        return self.kmeans.predict(X)

    def gerar_animacao(self, figsize=(12, 8), cores_vibrantes=None):
        """Gera animação do processo de clustering"""
        if cores_vibrantes is None:
            cores_vibrantes = ['#FF0000', '#0000FF', '#00FF00', '#FFFF00', '#FF00FF', '#00FFFF', '#FF8000']

        if self.X_data is None or len(self.historico_centroides) == 0:
            raise ValueError("Modelo precisa ser treinado antes de gerar animação")

        # Configurar matplotlib para backend não interativo
        plt.ioff()

        fig, ax = plt.subplots(figsize=figsize, dpi=100)

        # Definir limites fixos para todos os frames
        x_min, x_max = -3, 25
        y_min, y_max = -3, 25

        def animar(frame):
            ax.clear()

            # Usar labels da iteração atual (se disponível)
            if frame < len(self.historico_labels):
                labels = self.historico_labels[frame]
            else:
                labels = self.historico_labels[-1] if self.historico_labels else np.zeros(len(self.X_data))

            # Plotar pontos
            for k in range(self.n_clusters):
                mask = labels == k
                if np.sum(mask) > 0:
                    ax.scatter(self.X_data[mask, 0], self.X_data[mask, 1],
                               c=cores_vibrantes[k % len(cores_vibrantes)], s=80, alpha=0.7,
                               label=f'Cluster {k}', edgecolors='black', linewidth=0.5)

            # Plotar centroides
            if frame < len(self.historico_centroides):
                centroides = self.historico_centroides[frame]
                for k, centroide in enumerate(centroides):
                    ax.scatter(centroide[0], centroide[1],
                               c='black', s=300, marker='X',
                               edgecolors='white', linewidth=2)

            ax.set_title(f'K-Means Clustering - Iteração {frame}',
                         fontsize=16, fontweight='bold', color='darkblue')
            ax.set_xlabel('Primeiro Componente Principal', fontsize=12, fontweight='bold')
            ax.set_ylabel('Segundo Componente Principal', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # Definir limites consistentes
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        # Criar animação
        n_frames = len(self.historico_centroides)
        animacao = FuncAnimation(fig, animar, frames=n_frames, interval=1500, repeat=True, blit=False)

        plt.tight_layout()
        return animacao, fig


@st.cache_data
def processar_clustering_sklearn(df, n_clusters, n_componentes, max_features):
    """Aplica clustering nos textos do Reddit usando scikit-learn"""

    # Criar uma cópia explícita para evitar warnings
    df_processed = df.copy()

    # Pré-processamento
    df_processed['body_clean'] = df_processed['body'].apply(preprocessar_texto)
    df_processed = df_processed[df_processed['body_clean'].str.len() > 0].copy()

    # Vetorização TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )

    X_tfidf = vectorizer.fit_transform(df_processed['body_clean'])
    X_dense = X_tfidf.toarray()

    # Padronização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dense)

    # PCA com scikit-learn
    pca = PCA(n_components=n_componentes, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # K-means com scikit-learn (para análise final)
    kmeans_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels_final = kmeans_final.fit_predict(X_pca)

    # K-means animado (para gerar animação)
    kmeans_animado = KMeansAnimado(n_clusters=n_clusters, random_state=42)
    kmeans_animado.fit(X_pca)

    # Adicionar clusters ao dataframe usando .loc para evitar warnings
    df_processed = df_processed.copy()  # Garantir que temos uma cópia
    df_processed.loc[:, 'cluster'] = labels_final

    # Análise de palavras-chave por cluster
    feature_names = vectorizer.get_feature_names_out()
    clusters_info = {}

    for cluster_id in range(n_clusters):
        mask = labels_final == cluster_id
        cluster_docs = X_tfidf[mask]

        if cluster_docs.shape[0] > 0:
            mean_tfidf = cluster_docs.mean(axis=0).A1
            top_indices = mean_tfidf.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_indices]

            clusters_info[cluster_id] = {
                'palavras_chave': top_words,
                'n_documentos': mask.sum(),
                'exemplos': df_processed.loc[df_processed['cluster'] == cluster_id, 'body_clean'].head(3).tolist()
            }

    return df_processed, X_pca, pca, kmeans_final, kmeans_animado, clusters_info, X_tfidf, vectorizer


def salvar_animacao_gif(animacao, filename='animacao_kmeans.gif'):
    """Salva a animação como GIF e retorna os bytes"""
    try:
        # Criar um arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix='.gif') as tmp_file:
            temp_filename = tmp_file.name

        # Salvar a animação
        animacao.save(temp_filename, writer='pillow', fps=0.7, dpi=80)

        # Ler o arquivo
        with open(temp_filename, 'rb') as f:
            gif_bytes = f.read()

        # Limpar o arquivo temporário
        try:
            os.unlink(temp_filename)
        except:
            pass

        return gif_bytes
    except Exception as e:
        st.error(f"Erro ao salvar animação: {str(e)}")
        return None


# --- APLICAÇÃO STREAMLIT ---
st.title("Análise de Clustering com Scikit-Learn e Animação")

st.sidebar.header("Configurações da Análise")

# Upload de arquivo
arquivo_carregado = st.sidebar.file_uploader("Escolha um arquivo CSV", type="csv")
if arquivo_carregado is not None:
    try:
        df_original = pd.read_csv(arquivo_carregado, chunksiz = 100000)
        if 'body' not in df_original.columns:
            st.error("O arquivo deve conter a coluna 'body' com os textos")
        else:
            st.session_state['df_original'] = df_original
            st.sidebar.success(f"Arquivo carregado: {df_original.shape[0]} textos")
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")

if 'df_original' in st.session_state:
    df_original = st.session_state['df_original']

    # Parâmetro para limitar o número de linhas
    st.sidebar.subheader("Configurações dos Dados")
    max_linhas = st.sidebar.slider(
        "Número máximo de linhas para análise",
        min_value=100,
        max_value=len(df_original),
        value=min(2000, len(df_original)),
        step=100
    )

    # Limitar o dataframe
    if max_linhas < len(df_original):
        df = df_original.head(max_linhas).copy()
        st.sidebar.info(f"Usando {max_linhas} linhas de {len(df_original)} disponíveis")
    else:
        df = df_original.copy()

    st.header("1. Visualização dos Dados")

    # Mostrar informações básicas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Textos", len(df))
    with col2:
        st.metric("Colunas", len(df.columns))
    with col3:
        st.metric("Linhas Selecionadas", f"{len(df)} / {len(df_original)}")

    # Mostrar amostra dos dados
    st.dataframe(df.head())

    st.header("2. Configuração dos Parâmetros")

    # Parâmetros da análise
    st.sidebar.subheader("Parâmetros dos Modelos")
    n_componentes_pca = st.sidebar.selectbox("Número de Componentes PCA", [2, 3], index=0)
    n_clusters = st.sidebar.slider("Número de Clusters K-Means", 2, 7, 3)
    max_features = st.sidebar.slider("Máximo de Features TF-IDF", 500, 3000, 1000, step=100)

    if st.button("Executar Análise"):
        with st.spinner('Executando análise de clustering com scikit-learn...'):

            try:
                # Processar clustering
                df_result, X_pca, pca, kmeans_final, kmeans_animado, clusters_info, X_tfidf, vectorizer = processar_clustering_sklearn(
                    df, n_clusters, n_componentes_pca, max_features
                )

                st.header("3. Matriz TF-IDF")

                # Mostrar informações sobre a matriz TF-IDF
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Número de Documentos", X_tfidf.shape[0])
                with col2:
                    st.metric("Número de Features", X_tfidf.shape[1])
                with col3:
                    st.metric("Densidade da Matriz", f"{X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]):.2%}")

                # Mostrar uma amostra da matriz TF-IDF
                st.write("**Amostra da Matriz TF-IDF (primeiros 10 documentos e 10 features):**")

                # Converter para array denso para visualização
                X_tfidf_dense = X_tfidf.toarray()
                feature_names = vectorizer.get_feature_names_out()

                # Criar DataFrame para visualização
                df_tfidf_sample = pd.DataFrame(
                    X_tfidf_dense[:10, :10],
                    columns=feature_names[:10],
                    index=[f"Doc {i + 1}" for i in range(10)]
                )

                st.dataframe(df_tfidf_sample.round(4))

                st.header("4. Resultados do PCA")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Variância Explicada", f"{pca.explained_variance_ratio_.sum():.2%}")
                with col2:
                    st.metric("Features Originais → Componentes", f"{max_features} → {n_componentes_pca}")

                # Mostrar componentes principais
                st.write("**Variância explicada por componente:**")
                for i, var_ratio in enumerate(pca.explained_variance_ratio_):
                    st.write(f"PC{i + 1}: {var_ratio:.2%}")

                st.header("5. Visualização dos Clusters")

                # Preparar dados para visualização
                df_viz = pd.DataFrame(X_pca, columns=[f'PC{i + 1}' for i in range(n_componentes_pca)])
                df_viz['Cluster'] = kmeans_final.labels_.astype(str)

                # Cores elegantes
                cores_elegantes = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22']

                if n_componentes_pca == 2:
                    # Gráfico 2D interativo
                    fig_2d = px.scatter(
                        df_viz,
                        x='PC1',
                        y='PC2',
                        color='Cluster',
                        title='Clusters K-Means após PCA - 2D Interativo',
                        color_discrete_sequence=cores_elegantes[:n_clusters],
                        hover_data={'PC1': ':.3f', 'PC2': ':.3f'}
                    )

                    # Adicionar centroides como marcadores especiais
                    centroides_df = pd.DataFrame(
                        kmeans_final.cluster_centers_,
                        columns=['PC1', 'PC2']
                    )
                    centroides_df['Cluster'] = [f'Centroide {i}' for i in range(n_clusters)]

                    fig_2d.add_scatter(
                        x=centroides_df['PC1'],
                        y=centroides_df['PC2'],
                        mode='markers',
                        marker=dict(
                            size=20,
                            symbol='star',
                            color='black',
                            line=dict(width=2, color='white')
                        ),
                        name='Centroides',
                        hovertemplate='<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>',
                        text=centroides_df['Cluster']
                    )

                    fig_2d.update_traces(
                        marker=dict(size=8, line=dict(width=1, color='white')),
                        selector=dict(mode='markers', name__ne='Centroides')
                    )

                    fig_2d.update_layout(
                        height=600,
                        title_font_size=16,
                        xaxis_title='Primeira Componente Principal',
                        yaxis_title='Segunda Componente Principal',
                        showlegend=True
                    )

                    st.plotly_chart(fig_2d, use_container_width=True)

                elif n_componentes_pca == 3:
                    # Gráfico 3D
                    fig_3d = px.scatter_3d(
                        df_viz,
                        x='PC1',
                        y='PC2',
                        z='PC3',
                        color='Cluster',
                        title='Clusters K-Means após PCA (Scikit-Learn) - 3D',
                        color_discrete_sequence=cores_elegantes[:n_clusters]
                    )
                    fig_3d.update_traces(marker=dict(size=8, line=dict(width=1, color='black')))
                    fig_3d.update_layout(height=700)
                    st.plotly_chart(fig_3d, use_container_width=True)

                st.header("6. Estatísticas dos Clusters")

                # Tabela de estatísticas
                estatisticas_clusters = []
                for cluster_id in range(n_clusters):
                    info = clusters_info[cluster_id]
                    estatisticas_clusters.append({
                        'Cluster': cluster_id,
                        'Quantidade': info['n_documentos'],
                        'Percentual': f"{(info['n_documentos'] / len(df_result) * 100):.1f}%",
                        'Principais Palavras': ', '.join(info['palavras_chave'][:5])
                    })

                df_stats = pd.DataFrame(estatisticas_clusters)
                st.dataframe(df_stats, use_container_width=True)

                st.header("7. Análise Qualitativa dos Clusters")

                # Expandir cada cluster
                for cluster_id in range(n_clusters):
                    info = clusters_info[cluster_id]

                    with st.expander(f"Cluster {cluster_id} - {info['n_documentos']} documentos"):

                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("**Palavras-chave:**")
                            for palavra in info['palavras_chave'][:8]:
                                st.write(f"• {palavra}")

                        with col2:
                            st.write("**Exemplos de textos:**")
                            for i, exemplo in enumerate(info['exemplos']):
                                texto_truncado = exemplo[:150] + "..." if len(exemplo) > 150 else exemplo
                                st.write(f"{i + 1}. {texto_truncado}")

                # Animação K-Means (apenas para 2D)
                if n_componentes_pca == 2:
                    st.header("8. Animação do Processo K-Means")

                    with st.spinner("Gerando animação K-Means... Aguarde alguns instantes."):
                        try:
                            # Gerar animação
                            animacao, fig_anim = kmeans_animado.gerar_animacao(cores_vibrantes=cores_elegantes)

                            # Salvar como GIF
                            gif_bytes = salvar_animacao_gif(animacao)

                            if gif_bytes:
                                st.success("Animação gerada com sucesso!")
                                st.info(
                                    "A animação mostra como os centroides se movem a cada iteração do algoritmo K-Means.")

                                # Botão de download
                                st.download_button(
                                    label="📥 Baixar Animação (GIF)",
                                    data=gif_bytes,
                                    file_name="animacao_kmeans_clustering.gif",
                                    mime="image/gif",
                                    help="Clique para baixar a animação do processo K-Means"
                                )

                            # Fechar a figura para liberar memória
                            plt.close(fig_anim)

                        except Exception as e:
                            st.error(f"Erro ao gerar animação: {str(e)}")
                            st.write("Detalhes do erro:", str(e))
                else:
                    st.info("A animação K-Means está disponível apenas para análises com 2 componentes principais.")

                st.success("Análise concluída com sucesso!")

            except Exception as e:
                st.error(f"Erro durante a análise: {str(e)}")
                st.info("Verifique se o arquivo contém textos válidos na coluna 'body'")

else:
    st.info("Por favor, faça upload de um arquivo CSV para começar a análise")
    st.markdown("""
    ### Requisitos do arquivo:

    **Upload de Arquivo CSV:**
    - Deve ser um arquivo CSV
    - Deve conter uma coluna chamada **'body'** com os textos
    - Utilize o parâmetro de limite de linhas para controlar o tamanho da análise

    ### Funcionalidades:
    - **Visualização da Matriz TF-IDF**: Mostra como os textos são representados numericamente
    - **PCA com Scikit-Learn**: Redução de dimensionalidade precisa e eficiente
    - **K-Means com Scikit-Learn**: Clustering robusto e otimizado
    - **Animação GIF**: Visualização do processo de convergência do K-Means
    - **Análise de Texto**: Extração de palavras-chave por cluster usando TF-IDF
    """)