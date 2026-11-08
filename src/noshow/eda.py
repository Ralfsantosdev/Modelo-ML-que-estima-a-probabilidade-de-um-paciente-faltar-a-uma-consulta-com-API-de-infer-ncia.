"""Análise exploratória + auditoria de vazamento. Gera reports/eda.md e gráfico."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score

from noshow.data import TARGET
from noshow.datasets import carregar
from noshow.features import SINTETICO, Spec


def auc_univariado(df: pd.DataFrame, cols) -> pd.Series:
    out = {}
    for c in cols:
        x = df[c]
        if not pd.api.types.is_numeric_dtype(x):
            x = x.astype("category").cat.codes
        x = x.fillna(x.median())
        a = roc_auc_score(df[TARGET], x)
        out[c] = max(a, 1 - a)
    return pd.Series(out).sort_values(ascending=False)


def auditar_vazamento(df: pd.DataFrame, limite: float = 0.90, spec: Spec = SINTETICO) -> dict:
    """Sinaliza colunas suspeitas (AUC univariado alto) e duplicidades."""
    aucs = auc_univariado(df, [c for c in df.columns if c not in list(spec.ids) + [TARGET, "faixa_antecedencia"]])
    return {
        "auc_univariado": aucs,
        "suspeitas": aucs[aucs >= limite].index.tolist(),
        "duplicadas_paciente_dia": int(df.duplicated(["paciente_id", "data_consulta"]).sum()),
        "pacientes_repetidos_pct": float(df["paciente_id"].duplicated(keep=False).mean()),
    }


def main(dataset="sintetico", path=None, out_dir="reports"):
    out = Path(out_dir); out.mkdir(exist_ok=True, parents=True)
    df, spec, _ = carregar(dataset, path)
    audit = auditar_vazamento(df, spec=spec)
    df["faixa_antecedencia"] = pd.cut(df.antecedencia_dias, [-1, 0, 3, 7, 14, 30, 200])
    L = [f"# Análise exploratória e auditoria de vazamento ({spec.nome})\n",
         f"- Linhas: **{len(df):,}** | Pacientes: **{df.paciente_id.nunique():,}**",
         f"- Taxa de faltas: **{df[TARGET].mean():.1%}**",
         f"- Período: {df.data_consulta.min().date()} a {df.data_consulta.max().date()}",
         f"- Nulos em `taxa_faltas_previa` (1ª consulta): {df.taxa_faltas_previa.isna().mean():.1%}",
         f"- Linhas com paciente repetido: {audit['pacientes_repetidos_pct']:.1%} "
         "(exige split temporal/por grupo)\n",
         "## Faltas por segmento\n"]
    for c in spec.segmentos:
        t = df.groupby(c, observed=True)[TARGET].mean().round(3).to_frame("taxa_faltas")
        L.append(f"**{c}**\n\n{t.to_markdown()}\n")
    L.append("## Auditoria de vazamento (AUC univariado)\n")
    L.append(audit["auc_univariado"].round(3).to_frame("auc").to_markdown() + "\n")
    L.append(f"**Suspeitas (AUC >= 0.90): {audit['suspeitas']}** → excluídas das features.\n")
    L.append(f"Consultas duplicadas (paciente+dia): {audit['duplicadas_paciente_dia']}\n")
    L.append("### Outros riscos tratados\n"
             "- `historico_faltas` e `taxa_faltas_previa` usam só consultas anteriores (cumsum − atual).\n"
             "- Split **temporal** (treino < 2025-01, validação 2025-T1, teste 2025-T2).\n"
             "- Limiar escolhido na validação, nunca no teste.\n")
    (out / f"eda_{spec.nome}.md").write_text("\n".join(L), encoding="utf-8")

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    df.groupby("faixa_antecedencia", observed=True)[TARGET].mean().plot.bar(ax=ax[0], title="Faltas x antecedência")
    audit["auc_univariado"].head(10).iloc[::-1].plot.barh(ax=ax[1], title="AUC univariado (top 10)")
    plt.tight_layout(); fig.savefig(out / f"eda_{spec.nome}.png", dpi=120)
    print("Suspeitas:", audit["suspeitas"])
    assert set(audit["suspeitas"]) <= set(spec.leaky), f"Vazamento inesperado: {audit['suspeitas']}"
    return audit


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["sintetico", "kaggle"], default="sintetico")
    ap.add_argument("--path", default=None)
    a = ap.parse_args()
    main(a.dataset, a.path)
