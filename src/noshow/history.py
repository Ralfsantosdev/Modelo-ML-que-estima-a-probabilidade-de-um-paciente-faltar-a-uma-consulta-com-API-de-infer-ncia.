"""Histórico causal do paciente (compartilhado entre datasets)."""
import pandas as pd


def historico_causal(df: pd.DataFrame, target: str = "faltou") -> pd.DataFrame:
    """Contagem de consultas/faltas do paciente em dias ESTRITAMENTE anteriores.
    Consultas no mesmo dia ocorrem em paralelo: o desfecho de uma é desconhecido para as outras."""
    dia = (df.groupby(["paciente_id", "data_consulta"])[target]
             .agg(n="size", f="sum").reset_index().sort_values(["paciente_id", "data_consulta"]))
    g = dia.groupby("paciente_id")
    dia["historico_consultas"] = g["n"].cumsum() - dia["n"]
    dia["historico_faltas"] = g["f"].cumsum() - dia["f"]
    out = df.merge(dia[["paciente_id", "data_consulta", "historico_consultas", "historico_faltas"]],
                   on=["paciente_id", "data_consulta"], how="left")
    out["taxa_faltas_previa"] = (out["historico_faltas"] / out["historico_consultas"]).where(
        out["historico_consultas"] > 0)
    return out
