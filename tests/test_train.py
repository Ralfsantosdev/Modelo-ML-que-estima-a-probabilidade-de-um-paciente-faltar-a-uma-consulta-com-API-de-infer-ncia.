import pandas as pd
from noshow.train import best_threshold, metrics, run


def test_metricas_basicas():
    y = pd.Series([0, 0, 1, 1]); p = pd.Series([0.1, 0.4, 0.35, 0.8]).values
    m = metrics(y, p, 0.5)
    assert set(m) == {"precision", "recall", "f1", "roc_auc", "pr_auc"}
    assert m["precision"] == 1.0 and m["recall"] == 0.5


def test_limiar_no_intervalo():
    y = pd.Series([0, 0, 1, 1, 1]); p = pd.Series([0.1, 0.2, 0.6, 0.7, 0.9]).values
    assert 0 < best_threshold(y, p) < 1


def test_run_treina_tres_modelos_e_loga_mlflow(df_small, tmp_path):
    import mlflow
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    res = run(df_small, models_dir=tmp_path / "m", report_dir=tmp_path / "r",
              tracking_uri=uri, experiment="teste")
    assert set(res.index) == {"logistic_regression", "random_forest", "xgboost"}
    assert (res["test_roc_auc"] > 0.5).all()
    assert (tmp_path / "m" / "best_model.joblib").exists()
    runs = mlflow.search_runs(experiment_names=["teste"])
    assert len(runs) == 4  # 3 modelos + seleção
