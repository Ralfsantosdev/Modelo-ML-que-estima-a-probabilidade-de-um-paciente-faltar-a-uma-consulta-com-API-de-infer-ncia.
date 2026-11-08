import numpy as np
import pandas as pd
import pytest

from noshow.data import TARGET
from noshow.datasets import carregar
from noshow.features import KAGGLE, assert_no_leakage, split_por_quantil
from noshow.kaggle import carregar_kaggle, historico_causal
from tests.fake_kaggle import make_fake_kaggle


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory):
    return make_fake_kaggle(tmp_path_factory.mktemp("k") / "KaggleV2-May-2016.csv")


@pytest.fixture(scope="module")
def dfk(csv_path):
    return carregar_kaggle(str(csv_path))


def test_esquema_invalido(tmp_path):
    p = tmp_path / "x.csv"; pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="colunas ausentes"):
        carregar_kaggle(str(p))


def test_limpeza_remove_invalidos(dfk):
    assert dfk.attrs["linhas_removidas"] == 3          # idade -1, idade 200, agendado após a consulta
    assert dfk.idade.between(0, 115).all() and (dfk.antecedencia_dias >= 0).all()
    assert dfk.consulta_id.is_unique


def test_alvo_e_mapeamento(dfk):
    assert set(dfk[TARGET].unique()) <= {0, 1} and 0.1 < dfk[TARGET].mean() < 0.5
    assert set(KAGGLE.features) <= set(dfk.columns)
    assert dfk.paciente_id.dtype == "int64"
    assert dfk.hora_agendamento.between(0, 23).all()


def test_sem_vazamento_nas_features(dfk):
    assert_no_leakage(KAGGLE.features, KAGGLE)
    assert not {"consulta_id", "paciente_id", "data_consulta", TARGET} & set(KAGGLE.features)


def test_historico_ignora_mesmo_dia():
    df = pd.DataFrame({
        "paciente_id": [1, 1, 1, 1], "consulta_id": [1, 2, 3, 4],
        "data_consulta": pd.to_datetime(["2016-05-02", "2016-05-02", "2016-05-09", "2016-05-16"]),
        TARGET: [1, 0, 1, 0]})
    h = historico_causal(df).sort_values("consulta_id")
    assert h.historico_consultas.tolist() == [0, 0, 2, 3]   # duas consultas no mesmo dia não se enxergam
    assert h.historico_faltas.tolist() == [0, 0, 1, 2]
    assert np.isnan(h.taxa_faltas_previa.iloc[0]) and h.taxa_faltas_previa.iloc[2] == 0.5


def test_split_nao_divide_dias(dfk):
    tr, va, te = split_por_quantil(dfk)
    assert tr.data_consulta.max() < va.data_consulta.min() <= va.data_consulta.max() < te.data_consulta.min()
    assert len(tr) + len(va) + len(te) == len(dfk) and len(te) > 0


def test_carregar_dispatch(csv_path):
    df, spec, fn = carregar("kaggle", str(csv_path))
    assert spec.nome == "kaggle" and fn is split_por_quantil and len(df) > 0
    with pytest.raises(ValueError):
        carregar("inexistente")


def test_treino_ponta_a_ponta_kaggle(dfk, tmp_path):
    from noshow.train import run
    res = run(dfk, models_dir=tmp_path / "m", report_dir=tmp_path / "r",
              tracking_uri=f"sqlite:///{tmp_path}/ml.db", experiment="k",
              spec=KAGGLE, split_fn=split_por_quantil)
    assert set(res.index) == {"logistic_regression", "random_forest", "xgboost"}
    assert (res.test_roc_auc > 0.5).all()
    import json
    assert json.loads((tmp_path / "m" / "metadata.json").read_text())["dataset"] == "kaggle"
