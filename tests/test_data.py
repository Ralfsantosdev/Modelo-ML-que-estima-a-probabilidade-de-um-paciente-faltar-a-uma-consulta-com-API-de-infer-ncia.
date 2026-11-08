import pytest
from noshow.data import TARGET
from noshow.eda import auditar_vazamento
from noshow.features import FEATURES, LEAKY, assert_no_leakage, split_temporal


def test_features_nao_contem_vazamento():
    assert_no_leakage(FEATURES)
    assert not set(FEATURES) & set(LEAKY)


def test_assert_no_leakage_levanta_erro():
    with pytest.raises(ValueError):
        assert_no_leakage(FEATURES + ["checkin_realizado"])


def test_auditoria_detecta_colunas_vazadas(df_small):
    assert set(auditar_vazamento(df_small)["suspeitas"]) == set(LEAKY)


def test_historico_e_causal(df_small):
    """Histórico usa só consultas de dias anteriores (inclusive com várias consultas no mesmo dia)."""
    assert (df_small.historico_faltas <= df_small.historico_consultas).all()
    for pid in df_small.paciente_id.value_counts().index[:15]:
        p = df_small[df_small.paciente_id == pid]
        for _, r in p.iterrows():
            ant = p[p.data_consulta < r.data_consulta]
            assert r.historico_consultas == len(ant)
            assert r.historico_faltas == ant[TARGET].sum()


def test_consultas_mesmo_dia_nao_se_enxergam(df_small):
    dup = df_small[df_small.duplicated(["paciente_id", "data_consulta"], keep=False)]
    assert len(dup) > 0
    g = dup.groupby(["paciente_id", "data_consulta"])[["historico_consultas", "historico_faltas"]].nunique()
    assert (g == 1).all().all()


def test_split_temporal_sem_sobreposicao(df_small):
    tr, va, te = split_temporal(df_small)
    assert tr.data_consulta.max() < va.data_consulta.min()
    assert va.data_consulta.max() < te.data_consulta.min()
    assert len(tr) + len(va) + len(te) == len(df_small)


def test_dataset_reprodutivel():
    from noshow.data import gerar_dataset
    a = gerar_dataset(200, 1000, seed=7); b = gerar_dataset(200, 1000, seed=7)
    assert a.equals(b)
