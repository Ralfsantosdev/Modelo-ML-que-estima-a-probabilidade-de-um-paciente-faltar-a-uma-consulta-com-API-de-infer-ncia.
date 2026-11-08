"""Seleção de dataset: sintético ou Kaggle."""
from noshow.data import gerar_dataset
from noshow.features import KAGGLE, SINTETICO, split_por_quantil, split_temporal
from noshow.kaggle import carregar_kaggle

KAGGLE_PATH = "data/KaggleV2-May-2016.csv"


def carregar(nome: str, path: str | None = None):
    """Retorna (df, spec, split_fn)."""
    if nome == "sintetico":
        return gerar_dataset(), SINTETICO, split_temporal
    if nome == "kaggle":
        return carregar_kaggle(path or KAGGLE_PATH), KAGGLE, split_por_quantil
    raise ValueError(f"dataset desconhecido: {nome}")
