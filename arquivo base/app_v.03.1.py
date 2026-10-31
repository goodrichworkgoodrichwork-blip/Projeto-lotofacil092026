# =========================================================
# LOTERIAS IA - CENTRAL
# Mega-Sena + Lotofácil
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import random

from xgboost import XGBClassifier

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Bono-Loterias IA",
    layout="wide"
)

st.sidebar.title("🎯 Bono-Loterias IA")

opcao = st.sidebar.radio(
    "Escolha o sistema",
    [
        "Mega-Sena IA",
        "Lotofácil IA"
    ]
)

# =========================================================
# FUNÇÕES
# =========================================================

def sequencias(jogo):

    jogo = sorted(jogo)

    seq = 1
    max_seq = 1

    for i in range(1, len(jogo)):

        if jogo[i] == jogo[i - 1] + 1:
            seq += 1
            max_seq = max(max_seq, seq)

        else:
            seq = 1

    return max_seq

# =========================================================
# MEGA-SENA
# =========================================================

if opcao == "Mega-Sena IA":

    st.title("🎯 Mega-Sena IA")

    uploaded_file = st.file_uploader(
        "Envie Mega-Sena.xlsx",
        type=["xlsx"],
        key="mega"
    )

    if uploaded_file:

        df = pd.read_excel(uploaded_file)

        bolas = df[[f"Bola{i}" for i in range(1, 7)]]

        primos = {
            2,3,5,7,11,13,17,19,23,
            29,31,37,41,43,47,53,59
        }

        def features_jogo(jogo):

            return {
                "soma": sum(jogo),
                "pares": sum(1 for n in jogo if n % 2 == 0),
                "primos": sum(1 for n in jogo if n in primos),
                "sequencias": sequencias(jogo)
            }

        def matriz():

            m = pd.DataFrame(
                0,
                index=bolas.index,
                columns=range(1, 61)
            )

            for i, row in bolas.iterrows():

                for n in row:
                    m.at[i, n] = 1

            return m

        mat = matriz()

        @st.cache_resource
        def treinar():

            modelos = {}

            X = mat.shift(1).fillna(0)

            for n in range(1, 61):

                model = XGBClassifier(
                    eval_metric='logloss'
                )

                model.fit(X, mat[n])

                modelos[n] = model

            return modelos

        def probabilidades(modelos):

            ult = mat.iloc[-1].values.reshape(1, -1)

            return {
                n: modelos[n].predict_proba(ult)[0][1]
                for n in range(1, 61)
            }

        historico = set(
            tuple(sorted(r))
            for r in bolas.values
        )

        def gerar(qtd, probs):

            jogos = []
            usados = set()

            while len(jogos) < qtd:

                jogo = tuple(
                    sorted(
                        random.sample(range(1, 61), 6)
                    )
                )

                if jogo in historico:
                    continue

                if jogo in usados:
                    continue

                f = features_jogo(jogo)

                if not (120 <= f["soma"] <= 240):
                    continue

                score = sum(probs[n] for n in jogo)

                jogos.append((jogo, score))
                usados.add(jogo)

            jogos.sort(
                key=lambda x: x[1],
                reverse=True
            )

            return jogos

        qtd = st.sidebar.slider(
            "Quantidade Mega-Sena",
            1,
            100,
            5
        )

        if st.button("🚀 Gerar Mega-Sena"):

            modelos = treinar()

            probs = probabilidades(modelos)

            jogos = gerar(qtd * 3, probs)

            # =================================================
            # CORREÇÃO
            # =================================================
            jogos = jogos[:qtd]

            dfj = pd.DataFrame({

                "Jogo":
                    [i + 1 for i in range(len(jogos))],

                "Números":
                    [list(j) for j, _ in jogos],

                "Score":
                    [round(s, 4) for _, s in jogos]
            })

            st.dataframe(
                dfj,
                use_container_width=True
            )

# =========================================================
# LOTOFÁCIL
# =========================================================

if opcao == "Lotofácil IA":

    st.title("🚀 Lotofácil IA")

    uploaded_file = st.file_uploader(
        "Envie Lotofacil.xlsx",
        type=["xlsx"],
        key="lotofacil"
    )

    if uploaded_file:

        df = pd.read_excel(uploaded_file)

        bolas = df[[f"Bola{i}" for i in range(1, 16)]]

        def features_jogo(jogo):

            return {
                "soma": sum(jogo),
                "pares": sum(1 for n in jogo if n % 2 == 0),
                "sequencias": sequencias(jogo)
            }

        def matriz():

            m = pd.DataFrame(
                0,
                index=bolas.index,
                columns=range(1, 26)
            )

            for i, row in bolas.iterrows():

                for n in row:
                    m.at[i, n] = 1

            return m

        mat = matriz()

        @st.cache_resource
        def treinar():

            modelos = {}

            X = mat.shift(1).fillna(0)

            for n in range(1, 26):

                model = XGBClassifier(
                    eval_metric='logloss'
                )

                model.fit(X, mat[n])

                modelos[n] = model

            return modelos

        def probabilidades(modelos):

            ult = mat.iloc[-1].values.reshape(1, -1)

            return {
                n: modelos[n].predict_proba(ult)[0][1]
                for n in range(1, 26)
            }

        historico = set(
            tuple(sorted(r))
            for r in bolas.values
        )

        def gerar(qtd, probs):

            jogos = []
            usados = set()

            while len(jogos) < qtd:

                jogo = tuple(
                    sorted(
                        random.sample(range(1, 26), 15)
                    )
                )

                if jogo in historico:
                    continue

                if jogo in usados:
                    continue

                f = features_jogo(jogo)

                if not (180 <= f["soma"] <= 220):
                    continue

                score = sum(probs[n] for n in jogo)

                jogos.append((jogo, score))
                usados.add(jogo)

            jogos.sort(
                key=lambda x: x[1],
                reverse=True
            )

            return jogos

        qtd = st.sidebar.slider(
            "Quantidade Lotofácil",
            1,
            100,
            5
        )

        if st.button("🚀 Gerar Lotofácil"):

            modelos = treinar()

            probs = probabilidades(modelos)

            jogos = gerar(qtd * 3, probs)

            # =================================================
            # CORREÇÃO
            # =================================================
            jogos = jogos[:qtd]

            dfj = pd.DataFrame({

                "Jogo":
                    [i + 1 for i in range(len(jogos))],

                "Números":
                    [list(j) for j, _ in jogos],

                "Score":
                    [round(s, 4) for _, s in jogos]
            })

            st.dataframe(
                dfj,
                use_container_width=True
            )

st.markdown("---")
st.markdown("🎯 Loterias IA Corrigido")
