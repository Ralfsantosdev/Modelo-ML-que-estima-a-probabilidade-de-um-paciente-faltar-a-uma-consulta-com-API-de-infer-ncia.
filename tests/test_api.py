import pytest
from fastapi.testclient import TestClient

from app import main
from noshow.train import run

VALIDO = {"idade": 30, "sexo": "F", "tipo_consulta": "exame", "antecedencia_dias": 20,
          "hora_consulta": 9, "dia_semana": 0, "distancia_km": 12.5}


@pytest.fixture(scope="module")
def client(df_small, tmp_path_factory):
    d = tmp_path_factory.mktemp("art")
    run(df_small, models_dir=d / "m", report_dir=d / "r",
        tracking_uri=f"sqlite:///{d}/mlflow.db", experiment="api-test")
    main.load_artifacts(d / "m")
    with TestClient(main.app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_ok(client):
    r = client.post("/predict", json=VALIDO)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probabilidade_falta"] <= 1
    assert isinstance(body["previsao_falta"], bool)


def test_predict_risco_maior_com_historico_ruim(client):
    bom = client.post("/predict", json={**VALIDO, "historico_consultas": 5, "historico_faltas": 0}).json()
    ruim = client.post("/predict", json={**VALIDO, "historico_consultas": 5, "historico_faltas": 5}).json()
    assert ruim["probabilidade_falta"] >= bom["probabilidade_falta"]


@pytest.mark.parametrize("campo,valor", [("idade", -1), ("sexo", "X"), ("hora_consulta", 30),
                                         ("tipo_consulta", "invalido")])
def test_predict_valida_entrada(client, campo, valor):
    assert client.post("/predict", json={**VALIDO, campo: valor}).status_code == 422


def test_historico_inconsistente(client):
    r = client.post("/predict", json={**VALIDO, "historico_consultas": 1, "historico_faltas": 3})
    assert r.status_code == 422


def test_campos_obrigatorios(client):
    assert client.post("/predict", json={"idade": 30}).status_code == 422


# ---------------- API com modelo Kaggle ----------------
KAGGLE_OK = {"idade": 35, "sexo": "F", "bairro": "centro", "antecedencia_dias": 10, "dia_semana": 2,
             "dia_semana_agendamento": 0, "hora_agendamento": 14}


@pytest.fixture(scope="module")
def client_kaggle(tmp_path_factory):
    from noshow.features import KAGGLE, split_por_quantil
    from noshow.kaggle import carregar_kaggle
    from tests.fake_kaggle import make_fake_kaggle
    d = tmp_path_factory.mktemp("apik")
    df = carregar_kaggle(str(make_fake_kaggle(d / "k.csv")))
    run(df, models_dir=d / "m", report_dir=d / "r", tracking_uri=f"sqlite:///{d}/ml.db",
        experiment="apik", spec=KAGGLE, split_fn=split_por_quantil)
    antigo = dict(main.STATE)
    main.load_artifacts(d / "m")
    with TestClient(main.app) as c:
        yield c
    main.STATE.clear(); main.STATE.update(antigo)


def test_kaggle_health_e_predict(client_kaggle):
    assert client_kaggle.get("/health").json()["dataset"] == "kaggle"
    r = client_kaggle.post("/predict", json=KAGGLE_OK)
    assert r.status_code == 200 and 0 <= r.json()["probabilidade_falta"] <= 1


def test_kaggle_bairro_desconhecido_nao_quebra(client_kaggle):
    assert client_kaggle.post("/predict", json={**KAGGLE_OK, "bairro": "BAIRRO NOVO"}).status_code == 200


def test_kaggle_rejeita_payload_do_sintetico(client_kaggle):
    assert client_kaggle.post("/predict", json=VALIDO).status_code == 422
    assert client_kaggle.post("/predict", json={**KAGGLE_OK, "deficiencia": 9}).status_code == 422
