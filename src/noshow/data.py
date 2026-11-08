"""Geração de dataset sintético de consultas médicas.

O dataset imita o problema real (ex.: Medical Appointment No Shows, Kaggle),
mas é reprodutível e inclui, de propósito, colunas com vazamento
(`checkin_realizado`, `duracao_atendimento_min`) para exercitar a detecção.
O histórico do paciente é calculado de forma estritamente causal.
"""
import numpy as np
import pandas as pd

from noshow.history import historico_causal

TARGET = "faltou"


def gerar_dataset(n_pacientes: int = 6000, n_consultas: int = 25000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    pac = pd.DataFrame({
        "paciente_id": np.arange(n_pacientes),
        "idade": np.clip(rng.normal(38, 20, n_pacientes), 0, 95).astype(int),
        "sexo": rng.choice(["F", "M"], n_pacientes, p=[0.64, 0.36]),
        "bolsa_familia": rng.binomial(1, 0.10, n_pacientes),
        "hipertensao": rng.binomial(1, 0.20, n_pacientes),
        "diabetes": rng.binomial(1, 0.07, n_pacientes),
        "distancia_km": np.round(rng.gamma(2.0, 4.0, n_pacientes), 1),
        # propensão latente (não observada): correlaciona as consultas do mesmo paciente
        "_propensao": rng.normal(0, 1, n_pacientes),
    })

    df = pd.DataFrame({
        "paciente_id": rng.integers(0, n_pacientes, n_consultas),
        "data_consulta": pd.Timestamp("2024-01-01")
        + pd.to_timedelta(rng.integers(0, 547, n_consultas), unit="D"),
        "antecedencia_dias": np.minimum(rng.exponential(12, n_consultas).astype(int), 120),
        "hora_consulta": rng.choice(np.arange(7, 18), n_consultas),
        "tipo_consulta": rng.choice(["clinica_geral", "cardiologia", "pediatria", "exame", "retorno"],
                                    n_consultas, p=[0.35, 0.15, 0.15, 0.2, 0.15]),
    })
    df = df.merge(pac, on="paciente_id", how="left")
    df["dia_semana"] = df["data_consulta"].dt.dayofweek
    df["sms_enviado"] = ((df["antecedencia_dias"] >= 3) & (rng.random(len(df)) < 0.7)).astype(int)

    logit = (
        -1.55
        + 0.022 * df["antecedencia_dias"].clip(upper=60)
        + 0.85 * df["_propensao"]
        + 0.012 * df["distancia_km"]
        + 0.35 * df["bolsa_familia"]
        - 0.008 * (df["idade"] - 38)
        - 0.25 * df["sms_enviado"]
        + 0.30 * (df["tipo_consulta"] == "exame")
        - 0.25 * (df["tipo_consulta"] == "retorno")
        + 0.18 * (df["dia_semana"] == 0)
        + 0.12 * (df["hora_consulta"] >= 16)
    )
    p = 1 / (1 + np.exp(-logit))
    df[TARGET] = (rng.random(len(df)) < p).astype(int)

    # Histórico causal: apenas consultas de DIAS ANTERIORES do mesmo paciente.
    df = df.sort_values(["data_consulta", "paciente_id"], kind="stable").reset_index(drop=True)
    df = historico_causal(df, TARGET)

    # Colunas com vazamento (só existem APÓS o horário da consulta).
    df["checkin_realizado"] = ((1 - df[TARGET]) * (rng.random(len(df)) > 0.01)).astype(int)
    df["duracao_atendimento_min"] = np.where(df[TARGET] == 1, 0, rng.integers(10, 60, len(df)))

    df.insert(0, "consulta_id", np.arange(len(df)))
    return df.drop(columns=["_propensao"])
