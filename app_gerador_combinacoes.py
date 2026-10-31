# =========================================================
# GERADOR DE COMBINAÇÕES - LOTOFÁCIL
# =========================================================
# Projeto desenvolvido com Streamlit, Pandas, NumPy e Random.
#
# Este projeto é baseado no app_v.03.1.py.
# O objetivo é servir também como material de aprendizado:
# por isso, as principais partes do código possuem comentários
# explicando o que está sendo feito.
# =========================================================

import random

import numpy as np
import pandas as pd
import streamlit as st

# O XGBoost é utilizado de forma semelhante ao script base:
# cada número recebe uma probabilidade estimada a partir do
# histórico dos sorteios.
from xgboost import XGBClassifier


# =========================================================
# CONFIGURAÇÃO DA PÁGINA
# =========================================================

st.set_page_config(
    page_title="Bonomo - Gerador de Combinações",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Bonomo - Gerador de Combinações Lotofácil")
st.caption("Projeto para aprendizado")


# =========================================================
# CONSTANTES
# =========================================================

# A Lotofácil possui 25 números disponíveis.
NUMEROS_DISPONIVEIS = list(range(1, 26))

# Números primos existentes entre 1 e 25.
PRIMOS = {2, 3, 5, 7, 11, 13, 17, 19, 23}

# Números de Fibonacci existentes entre 1 e 25.
FIBONACCI = {1, 2, 3, 5, 8, 13, 21}

# Limites definidos no arquivo de instruções.
SOMA_MIN = 170
SOMA_MAX = 221


# =========================================================
# FUNÇÕES AUXILIARES
# =========================================================

def contar_pares(jogo):
    """Retorna a quantidade de números pares de um jogo."""
    return sum(1 for numero in jogo if numero % 2 == 0)


def contar_primos(jogo):
    """Retorna quantos números do jogo são primos."""
    return sum(1 for numero in jogo if numero in PRIMOS)


def contar_fibonacci(jogo):
    """Retorna quantos números do jogo pertencem à sequência de Fibonacci."""
    return sum(1 for numero in jogo if numero in FIBONACCI)


def contar_multiplos_3(jogo):
    """Retorna quantos números do jogo são múltiplos de 3."""
    return sum(1 for numero in jogo if numero % 3 == 0)


def contar_anterior(jogo, sorteio_anterior):
    """
    Retorna quantos números do jogo também apareceram no sorteio anterior.
    """
    return sum(1 for numero in jogo if numero in sorteio_anterior)


def jogo_valido(
    jogo,
    pares_min,
    pares_max,
    anterior_min,
    anterior_max,
    primos_min,
    primos_max,
    fibonacci_min,
    fibonacci_max,
    soma_min,
    soma_max,
    multiplos_min,
    multiplos_max,
):
    """
    Verifica todos os filtros definidos pelo usuário.

    A quantidade de ímpares não precisa de um filtro separado:
    ela é automaticamente determinada pela quantidade de pares.
    """

    pares = contar_pares(jogo)
    primos = contar_primos(jogo)
    fibonacci = contar_fibonacci(jogo)
    multiplos_3 = contar_multiplos_3(jogo)
    anterior = contar_anterior(jogo, sorteio_anterior)

    return (
        pares_min <= pares <= pares_max
        and anterior_min <= anterior <= anterior_max
        and primos_min <= primos <= primos_max
        and fibonacci_min <= fibonacci <= fibonacci_max
        and soma_min <= sum(jogo) <= soma_max
        and multiplos_min <= multiplos_3 <= multiplos_max
    )


def criar_matriz_historica(bolas):
    """
    Converte o histórico de sorteios para uma matriz 0/1.

    Cada linha representa um sorteio.
    Cada coluna representa um número de 1 a 25.
    1 significa que o número apareceu no sorteio.
    0 significa que não apareceu.
    """

    matriz = pd.DataFrame(
        0,
        index=bolas.index,
        columns=NUMEROS_DISPONIVEIS,
        dtype=int,
    )

    for indice, linha in bolas.iterrows():
        for numero in linha:
            # Garantimos que o Excel não introduza valores inválidos.
            if pd.notna(numero):
                numero = int(numero)
                if numero in NUMEROS_DISPONIVEIS:
                    matriz.at[indice, numero] = 1

    return matriz


@st.cache_resource
def treinar_modelos(matriz):
    """
    Treina um modelo XGBoost para cada número da Lotofácil.

    O modelo utiliza o sorteio anterior como entrada e tenta
    estimar a probabilidade de cada número aparecer no próximo sorteio.

    O cache evita treinar novamente os modelos a cada interação
    com a interface.
    """

    modelos = {}

    # Shift(1) faz com que cada linha utilize o sorteio anterior
    # como entrada. A primeira linha não possui sorteio anterior,
    # por isso usamos zero.
    X = matriz.shift(1).fillna(0)

    for numero in NUMEROS_DISPONIVEIS:
        y = matriz[numero]

        # Se o histórico não possuir variação suficiente para um
        # classificador, não interrompemos o aplicativo.
        if y.nunique() < 2:
            modelos[numero] = None
            continue

        modelo = XGBClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )

        modelo.fit(X, y)
        modelos[numero] = modelo

    return modelos


def calcular_probabilidades(modelos, matriz):
    """
    Calcula a probabilidade estimada para cada número de 1 a 25.

    Caso um modelo não possa ser utilizado, atribuímos uma
    probabilidade neutra de 0.5.
    """

    ultimo_sorteio = matriz.iloc[-1].values.reshape(1, -1)
    probabilidades = {}

    for numero in NUMEROS_DISPONIVEIS:
        modelo = modelos.get(numero)

        if modelo is None:
            probabilidades[numero] = 0.5
            continue

        probabilidades[numero] = float(
            modelo.predict_proba(ultimo_sorteio)[0][1]
        )

    return probabilidades


def selecionar_jogo_ponderado(tamanho, probabilidades):
    """
    Cria uma combinação utilizando as probabilidades calculadas
    pelo modelo.

    O random.choices permite que números com maior probabilidade
    tenham maior chance de serem escolhidos, mas sem tornar o
    resultado determinístico.
    """

    pesos = np.array(
        [max(probabilidades[numero], 0.001) for numero in NUMEROS_DISPONIVEIS],
        dtype=float,
    )

    # Normalização dos pesos para facilitar o entendimento.
    pesos = pesos / pesos.sum()

    jogo = np.random.choice(
        NUMEROS_DISPONIVEIS,
        size=tamanho,
        replace=False,
        p=pesos,
    )

    return tuple(sorted(int(numero) for numero in jogo))


def gerar_combinacoes(
    quantidade_jogos,
    tamanho_jogo,
    probabilidades,
    historico,
    sorteio_anterior,
    filtros,
    max_tentativas=200000,
):
    """
    Gera combinações até atingir a quantidade solicitada.

    A função rejeita:
      1. jogos já sorteados;
      2. jogos repetidos na geração atual;
      3. jogos que não atendam aos filtros escolhidos.
    """

    jogos = []
    usados = set()

    tentativas = 0

    while len(jogos) < quantidade_jogos and tentativas < max_tentativas:
        tentativas += 1

        jogo = selecionar_jogo_ponderado(
            tamanho_jogo,
            probabilidades,
        )

        # Não permite jogos já sorteados no histórico.
        if jogo in historico:
            continue

        # Não permite duplicar jogos na mesma geração.
        if jogo in usados:
            continue

        # Verifica os filtros definidos na interface.
        if not jogo_valido(
            jogo=jogo,
            pares_min=filtros["pares_min"],
            pares_max=filtros["pares_max"],
            anterior_min=filtros["anterior_min"],
            anterior_max=filtros["anterior_max"],
            primos_min=filtros["primos_min"],
            primos_max=filtros["primos_max"],
            fibonacci_min=filtros["fibonacci_min"],
            fibonacci_max=filtros["fibonacci_max"],
            soma_min=filtros["soma_min"],
            soma_max=filtros["soma_max"],
            multiplos_min=filtros["multiplos_min"],
            multiplos_max=filtros["multiplos_max"],
        ):
            continue

        jogos.append(jogo)
        usados.add(jogo)

    return jogos, tentativas


def criar_dataframe_resultado(jogos, probabilidades, sorteio_anterior):
    """
    Organiza as combinações geradas em um DataFrame para exibição.
    """

    registros = []

    for indice, jogo in enumerate(jogos, start=1):
        registros.append(
            {
                "Jogo": indice,
                "Números": " - ".join(f"{n:02d}" for n in jogo),
                "Quantidade": len(jogo),
                "Pares": contar_pares(jogo),
                "Ímpares": len(jogo) - contar_pares(jogo),
                "Anterior": contar_anterior(jogo, sorteio_anterior),
                "Primos": contar_primos(jogo),
                "Fibonacci": contar_fibonacci(jogo),
                "Múltiplos de 3": contar_multiplos_3(jogo),
                "Soma": sum(jogo),
                "Score IA": round(
                    sum(probabilidades[n] for n in jogo),
                    4,
                ),
            }
        )

    return pd.DataFrame(registros)


# =========================================================
# SIDEBAR - CONFIGURAÇÕES
# =========================================================

st.sidebar.header("⚙️ Configurações")

uploaded_file = st.sidebar.file_uploader(
    "Envie o Excel com os sorteios anteriores",
    type=["xlsx", "xls"],
    help="O arquivo deve possuir as colunas Bola1 até Bola15.",
)

# A quantidade padrão é 15, conforme o arquivo de instruções.
tamanho_jogo = st.sidebar.slider(
    "Quantidade de números por combinação",
    min_value=15,
    max_value=20,
    value=15,
    help="Padrão: 15 números.",
)

quantidade_jogos = st.sidebar.slider(
    "Quantidade de combinações",
    min_value=1,
    max_value=100,
    value=5,
)

st.sidebar.subheader("🎯 Filtros")

pares_min, pares_max = st.sidebar.slider(
    "Quantidade de pares",
    min_value=6,
    max_value=8,
    value=(6, 8),
)

# Como a quantidade de ímpares é complementar ao tamanho do jogo,
# não criamos um segundo controle independente.
st.sidebar.info(
    "Ímpares: calculados automaticamente como "
    "quantidade do jogo − quantidade de pares."
)

anterior_min, anterior_max = st.sidebar.slider(
    "Números do sorteio anterior",
    min_value=0,
    max_value=15,
    value=(0, 9),
    help="O limite 9 é o padrão indicado nas instruções.",
)

primos_min, primos_max = st.sidebar.slider(
    "Quantidade de primos",
    min_value=4,
    max_value=7,
    value=(4, 7),
)

fibonacci_min, fibonacci_max = st.sidebar.slider(
    "Quantidade de Fibonacci",
    min_value=3,
    max_value=6,
    value=(3, 6),
)

multiplos_min, multiplos_max = st.sidebar.slider(
    "Múltiplos de 3",
    min_value=3,
    max_value=6,
    value=(3, 6),
)

soma_min, soma_max = st.sidebar.slider(
    "Intervalo da soma",
    min_value=170,
    max_value=221,
    value=(170, 221),
)


# =========================================================
# PROCESSAMENTO DO ARQUIVO EXCEL
# =========================================================

if uploaded_file is None:
    st.info(
        "📂 Envie o arquivo Excel na barra lateral para carregar "
        "o histórico dos sorteios."
    )

    st.markdown(
        """
        ### Como utilizar

        1. Envie o arquivo Excel com os sorteios anteriores.
        2. Escolha entre **15 e 20 números** por combinação.
        3. Escolha a quantidade de combinações.
        4. Ajuste os filtros desejados.
        5. Clique em **Gerar combinações**.
        """
    )

    st.stop()


try:
    # Lemos o Excel enviado pelo usuário.
    df = pd.read_excel(uploaded_file)

    # Procuramos as colunas esperadas pelo script base.
    colunas_bolas = [f"Bola{i}" for i in range(1, 16)]

    colunas_faltantes = [
        coluna for coluna in colunas_bolas if coluna not in df.columns
    ]

    if colunas_faltantes:
        st.error(
            "O arquivo Excel não possui as colunas esperadas: "
            + ", ".join(colunas_faltantes)
        )
        st.stop()

    bolas = df[colunas_bolas].copy()

    # Converte as dezenas para números, descartando valores inválidos.
    for coluna in colunas_bolas:
        bolas[coluna] = pd.to_numeric(
            bolas[coluna],
            errors="coerce",
        )

    bolas = bolas.dropna().astype(int)

except Exception as erro:
    st.error(f"Não foi possível ler o arquivo Excel: {erro}")
    st.stop()


# =========================================================
# VALIDAÇÃO DO HISTÓRICO
# =========================================================

if len(bolas) < 2:
    st.error(
        "É necessário possuir pelo menos dois sorteios no histórico "
        "para utilizar o modelo."
    )
    st.stop()


# Converte o histórico para uma matriz de presença/ausência.
matriz = criar_matriz_historica(bolas)

# Guarda os jogos históricos em um set.
# Isso permite verificar rapidamente se uma combinação já foi sorteada.
historico = {
    tuple(sorted(int(numero) for numero in linha))
    for linha in bolas.values
}

# O último sorteio é utilizado para calcular a quantidade de
# números repetidos na combinação gerada.
sorteio_anterior = set(
    int(numero) for numero in bolas.iloc[-1].values
)


# =========================================================
# RESUMO DO HISTÓRICO
# =========================================================

col1, col2, col3 = st.columns(3)

col1.metric("📊 Sorteios carregados", len(bolas))
col2.metric("🎯 Último sorteio", " - ".join(
    f"{n:02d}" for n in sorted(sorteio_anterior)
))
col3.metric("🧾 Jogos históricos", len(historico))


with st.expander("🔎 Ver último sorteio"):
    st.write(sorted(sorteio_anterior))


# =========================================================
# GERAÇÃO
# =========================================================

st.subheader("🚀 Gerar combinações")

st.write(
    f"Serão geradas combinações com **{tamanho_jogo} números**, "
    f"sem repetir jogos existentes no histórico."
)

if st.button(
    "🚀 Gerar combinações",
    type="primary",
    use_container_width=True,
):

    with st.spinner("Treinando IA e procurando combinações válidas..."):

        # Treina os modelos semelhantes ao script app_v.03.1.py.
        modelos = treinar_modelos(matriz)

        # Obtém as probabilidades para cada número.
        probabilidades = calcular_probabilidades(
            modelos,
            matriz,
        )

        filtros = {
            "pares_min": pares_min,
            "pares_max": pares_max,
            "anterior_min": anterior_min,
            "anterior_max": anterior_max,
            "primos_min": primos_min,
            "primos_max": primos_max,
            "fibonacci_min": fibonacci_min,
            "fibonacci_max": fibonacci_max,
            "soma_min": soma_min,
            "soma_max": soma_max,
            "multiplos_min": multiplos_min,
            "multiplos_max": multiplos_max,
        }

        jogos, tentativas = gerar_combinacoes(
            quantidade_jogos=quantidade_jogos,
            tamanho_jogo=tamanho_jogo,
            probabilidades=probabilidades,
            historico=historico,
            sorteio_anterior=sorteio_anterior,
            filtros=filtros,
        )

    # =====================================================
    # RESULTADO
    # =====================================================

    if len(jogos) < quantidade_jogos:
        st.warning(
            f"Foram encontradas apenas {len(jogos)} combinações válidas "
            f"após {tentativas:,} tentativas. "
            "Os filtros podem estar muito restritivos para a quantidade "
            "de números escolhida."
        )

    if not jogos:
        st.error(
            "Nenhuma combinação foi encontrada. "
            "Tente ampliar algum dos filtros."
        )
        st.stop()

    resultado = criar_dataframe_resultado(
        jogos,
        probabilidades,
        sorteio_anterior,
    )

    st.success(
        f"{len(jogos)} combinação(ões) encontrada(s) "
        f"em {tentativas:,} tentativas."
    )

    st.dataframe(
        resultado,
        use_container_width=True,
        hide_index=True,
    )

    # =====================================================
    # DOWNLOAD
    # =====================================================

    # Também disponibilizamos o resultado em CSV.
    # Isso facilita guardar as combinações geradas.
    csv = resultado.to_csv(
        index=False,
        sep=";",
        encoding="utf-8-sig",
    )

    st.download_button(
        "📥 Baixar combinações em CSV",
        data=csv,
        file_name="combinacoes_lotofacil.csv",
        mime="text/csv",
        use_container_width=True,
    )


# =========================================================
# RODAPÉ
# =========================================================

st.markdown("---")
st.caption(
    "🎯 Bono - Gerador de Combinações | "
    "Projeto educacional baseado no app_v.03.1.py"
)
