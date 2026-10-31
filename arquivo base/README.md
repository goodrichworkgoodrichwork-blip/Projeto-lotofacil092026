# app_v0.3.3
Sistema de geração inteligente de jogos da Lotofácil com Machine Learning e otimização
# 🎯 Lotofácil IA - Gerador Inteligente com Machine Learning

Sistema desenvolvido em Python utilizando Streamlit para análise de dados e geração inteligente de jogos da Lotofácil.

---

## 🚀 Objetivo

Criar um modelo híbrido combinando:

- 📊 Análise estatística
- 🧠 Machine Learning (XGBoost)
- 🧬 Algoritmo Genético
- 🎲 Simulação Monte Carlo
- ⚡ Otimização heurística

---

## 🧠 Tecnologias Utilizadas

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- XGBoost

---

## 📊 Como o sistema funciona

1. Carrega dados históricos dos concursos
2. Converte para matriz binária
3. Treina um modelo para cada número (1 a 25)
4. Calcula probabilidade de ocorrência
5. Gera jogos candidatos
6. Aplica algoritmo genético
7. Otimiza com Monte Carlo
8. Filtra diversidade
9. Retorna os melhores jogos

---

## 🎯 Funcionalidades

- ✔ Geração de jogos inteligentes
- ✔ Controle de quantidade de jogos
- ✔ Evita jogos já sorteados
- ✔ Score baseado em probabilidade
- ✔ Diversidade entre jogos
- ✔ Gráfico de probabilidade

---

## ⚙️ Como executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
