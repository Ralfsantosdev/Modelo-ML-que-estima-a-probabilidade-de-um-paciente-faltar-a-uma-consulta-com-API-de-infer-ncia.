"""Treino e comparação de 3 modelos com rastreamento no MLflow."""
import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from noshow.data import TARGET, gerar_dataset
from noshow.datasets import carregar
from noshow.features import SINTETICO, Spec, assert_no_leakage, build_preprocessor, split_temporal

SEED = 42


def make_models(pos_weight: float, spec: Spec = SINTETICO) -> dict:
    return {
        "logistic_regression": Pipeline([
            ("prep", build_preprocessor(scale=True, spec=spec)),
            ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=SEED)),
        ]),
        "random_forest": Pipeline([
            ("prep", build_preprocessor(spec=spec)),
            ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features="sqrt",
                                           class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)),
        ]),
        "xgboost": Pipeline([
            ("prep", build_preprocessor(spec=spec)),
            ("clf", XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                                  colsample_bytree=0.8, min_child_weight=10, scale_pos_weight=pos_weight,
                                  eval_metric="logloss", tree_method="hist", random_state=SEED, n_jobs=2)),
        ]),
    }


def best_threshold(y, p) -> float:
    """Limiar que maximiza F1 — escolhido SOMENTE na validação."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    return float(thr[np.argmax(f1)])


def metrics(y, p, thr) -> dict:
    pred = (p >= thr).astype(int)
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
    }


def run(df: pd.DataFrame | None = None, models_dir="models", report_dir="reports",
        tracking_uri="sqlite:///mlflow.db", experiment="noshow-consultas",
        spec: Spec = SINTETICO, split_fn=split_temporal) -> pd.DataFrame:
    df = gerar_dataset() if df is None else df
    FEATURES = spec.features
    assert_no_leakage(FEATURES, spec)
    tr, va, te = split_fn(df)
    Xtr, ytr, Xva, yva, Xte, yte = tr[FEATURES], tr[TARGET], va[FEATURES], va[TARGET], te[FEATURES], te[TARGET]
    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    rows, fitted = [], {}
    for name, pipe in make_models(pos_weight, spec).items():
        with mlflow.start_run(run_name=name):
            pipe.fit(Xtr, ytr)
            thr = best_threshold(yva, pipe.predict_proba(Xva)[:, 1])
            m_val = metrics(yva, pipe.predict_proba(Xva)[:, 1], thr)
            m_te = metrics(yte, pipe.predict_proba(Xte)[:, 1], thr)
            mlflow.log_params({"model": name, "threshold": round(thr, 4), "features": len(FEATURES),
                               "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                               "split": "temporal", "seed": SEED, "dataset": spec.nome,
                               "treino_ate": str(tr["data_consulta"].max().date()),
                               "teste_a_partir": str(te["data_consulta"].min().date()),
                               **{f"clf__{k}": v for k, v in pipe.named_steps["clf"].get_params().items()
                                  if isinstance(v, (int, float, str, bool, type(None)))}})
            mlflow.log_metrics({**{f"val_{k}": v for k, v in m_val.items()},
                                **{f"test_{k}": v for k, v in m_te.items()}})
            mlflow.sklearn.log_model(pipe, name="model", input_example=Xte.head(3), serialization_format="cloudpickle")
            rows.append({"modelo": name, "limiar": thr, **{f"val_{k}": v for k, v in m_val.items()},
                         **{f"test_{k}": v for k, v in m_te.items()}})
            fitted[name] = (pipe, thr)

    res = pd.DataFrame(rows).set_index("modelo")
    best = res["val_roc_auc"].idxmax()          # seleção na validação, não no teste
    pipe, thr = fitted[best]
    Path(models_dir).mkdir(exist_ok=True, parents=True)
    Path(report_dir).mkdir(exist_ok=True, parents=True)
    joblib.dump(pipe, Path(models_dir) / "best_model.joblib")
    (Path(models_dir) / "metadata.json").write_text(json.dumps(
        {"modelo": best, "limiar": thr, "features": FEATURES, "dataset": spec.nome,
         "treino_ate": str(tr["data_consulta"].max().date())}, indent=2))
    with mlflow.start_run(run_name="selecao"):
        mlflow.log_params({"melhor_modelo": best, "criterio": "val_roc_auc", "dataset": spec.nome})
        mlflow.log_metric("test_roc_auc_melhor", res.loc[best, "test_roc_auc"])
    (Path(report_dir) / "comparacao_modelos.md").write_text(
        f"# Comparação (conjunto de TESTE, limiar da validação)\n\nMelhor por ROC-AUC na validação: **{best}**\n\n"
        + res[[c for c in res.columns if c.startswith("test_") or c == "limiar"]].round(4).to_markdown() + "\n",
        encoding="utf-8")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["sintetico", "kaggle"], default="sintetico")
    ap.add_argument("--path", default=None, help="CSV do Kaggle (padrão: data/KaggleV2-May-2016.csv)")
    a = ap.parse_args()
    df, spec, split_fn = carregar(a.dataset, a.path)
    out = run(df, spec=spec, split_fn=split_fn, experiment=f"noshow-{spec.nome}")
    pd.set_option("display.width", 200)
    print(out.filter(like="test_").round(3).join(out["limiar"].round(3)))
