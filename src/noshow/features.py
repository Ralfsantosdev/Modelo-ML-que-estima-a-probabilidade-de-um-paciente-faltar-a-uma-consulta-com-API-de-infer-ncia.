"""Especificações de features por dataset e pré-processamento."""
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class Spec:
    nome: str
    numeric: tuple
    categorical: tuple
    leaky: tuple = ()
    ids: tuple = ("consulta_id", "paciente_id", "data_consulta")
    segmentos: tuple = ()
    min_freq: int | None = None  # agrupa categorias raras (ex.: bairro) como "infrequent"

    @property
    def features(self) -> list:
        return list(self.numeric) + list(self.categorical)


SINTETICO = Spec(
    nome="sintetico",
    numeric=("idade", "antecedencia_dias", "hora_consulta", "dia_semana", "distancia_km",
             "bolsa_familia", "hipertensao", "diabetes", "sms_enviado",
             "historico_consultas", "historico_faltas", "taxa_faltas_previa"),
    categorical=("sexo", "tipo_consulta"),
    leaky=("checkin_realizado", "duracao_atendimento_min"),
    segmentos=("tipo_consulta", "dia_semana", "sms_enviado", "bolsa_familia", "faixa_antecedencia"),
)

# Kaggle "Medical Appointment No Shows" (Joni Hoppen), 110.527 consultas, abr–jun/2016.
KAGGLE = Spec(
    nome="kaggle",
    numeric=("idade", "antecedencia_dias", "dia_semana", "dia_semana_agendamento", "hora_agendamento",
             "bolsa_familia", "hipertensao", "diabetes", "alcoolismo", "deficiencia", "sms_enviado",
             "historico_consultas", "historico_faltas", "taxa_faltas_previa"),
    categorical=("sexo", "bairro"),
    segmentos=("sms_enviado", "bolsa_familia", "deficiencia", "dia_semana", "faixa_antecedencia"),
    min_freq=300,
)
SPECS = {s.nome: s for s in (SINTETICO, KAGGLE)}

# Atalhos de compatibilidade (dataset sintético)
NUMERIC, CATEGORICAL, FEATURES = list(SINTETICO.numeric), list(SINTETICO.categorical), SINTETICO.features
LEAKY, IDS = list(SINTETICO.leaky), list(SINTETICO.ids)


def build_preprocessor(scale: bool = False, spec: Spec = SINTETICO) -> ColumnTransformer:
    num_steps = [("imp", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    enc = OneHotEncoder(handle_unknown="infrequent_if_exist" if spec.min_freq else "ignore",
                        min_frequency=spec.min_freq)
    return ColumnTransformer([
        ("num", Pipeline(num_steps), list(spec.numeric)),
        ("cat", enc, list(spec.categorical)),
    ])


def assert_no_leakage(columns, spec: Spec = SINTETICO) -> None:
    bad = set(columns) & set(spec.leaky + spec.ids)
    if bad:
        raise ValueError(f"Features com vazamento ou identificadores: {sorted(bad)}")


def split_temporal(df: pd.DataFrame, val_start="2025-01-01", test_start="2025-04-01"):
    d = df["data_consulta"]
    return df[d < val_start], df[(d >= val_start) & (d < test_start)], df[d >= test_start]


def split_por_quantil(df: pd.DataFrame, q_val: float = 0.70, q_test: float = 0.85):
    """Split temporal por dias distintos (para janelas curtas, como o Kaggle: ~6 semanas).
    Um dia nunca é dividido entre conjuntos."""
    dias = df["data_consulta"].drop_duplicates().sort_values().reset_index(drop=True)
    val_start, test_start = dias[int(len(dias) * q_val)], dias[int(len(dias) * q_test)]
    return split_temporal(df, val_start, test_start)
