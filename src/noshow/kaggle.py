"""Adaptador do dataset Kaggle 'Medical Appointment No Shows' (KaggleV2-May-2016.csv).

Colunas originais: PatientId, AppointmentID, Gender, ScheduledDay, AppointmentDay, Age,
Neighbourhood, Scholarship, Hipertension, Diabetes, Alcoholism, Handcap, SMS_received, No-show.

Particularidades tratadas:
- AppointmentDay vem sem horário (00:00) -> antecedência calculada em DIAS, por datas normalizadas.
- Idades inválidas (<0 ou >115) e antecedência negativa (agendamento posterior à consulta) são removidas.
- `Handcap` é 0–4 (nível), não binário -> `deficiencia`.
- A hora de ScheduledDay é a do AGENDAMENTO (conhecida na predição), não da consulta.
- Histórico do paciente só usa consultas de DIAS ANTERIORES (consultas no mesmo dia ocorrem em
  paralelo: seu desfecho ainda é desconhecido).
- `No-show` = "Yes" significa que o paciente FALTOU.
"""
import pandas as pd

from noshow.data import TARGET
from noshow.history import historico_causal  # noqa: F401  (re-exportado)

COLUNAS = ["PatientId", "AppointmentID", "Gender", "ScheduledDay", "AppointmentDay", "Age",
           "Neighbourhood", "Scholarship", "Hipertension", "Diabetes", "Alcoholism",
           "Handcap", "SMS_received", "No-show"]


def carregar_kaggle(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    faltando = set(COLUNAS) - set(raw.columns)
    if faltando:
        raise ValueError(f"CSV fora do esquema Kaggle; colunas ausentes: {sorted(faltando)}")

    ag = pd.to_datetime(raw["ScheduledDay"], utc=True).dt.tz_localize(None)
    co = pd.to_datetime(raw["AppointmentDay"], utc=True).dt.tz_localize(None).dt.normalize()
    df = pd.DataFrame({
        "consulta_id": raw["AppointmentID"],
        "paciente_id": raw["PatientId"].astype("int64"),
        "data_consulta": co,
        "idade": raw["Age"],
        "sexo": raw["Gender"].astype("string").str.upper(),
        "bairro": raw["Neighbourhood"].astype("string").str.strip().str.upper(),
        "bolsa_familia": raw["Scholarship"],
        "hipertensao": raw["Hipertension"],
        "diabetes": raw["Diabetes"],
        "alcoolismo": raw["Alcoholism"],
        "deficiencia": raw["Handcap"],
        "sms_enviado": raw["SMS_received"],
        "antecedencia_dias": (co - ag.dt.normalize()).dt.days,
        "dia_semana": co.dt.dayofweek,
        "dia_semana_agendamento": ag.dt.dayofweek,
        "hora_agendamento": ag.dt.hour,
        TARGET: (raw["No-show"].astype(str).str.strip().str.lower() == "yes").astype(int),
    })
    n0 = len(df)
    df = df[df["idade"].between(0, 115) & (df["antecedencia_dias"] >= 0)]
    df = df.drop_duplicates("consulta_id")
    df = df.sort_values(["data_consulta", "paciente_id", "consulta_id"], kind="stable").reset_index(drop=True)
    df = historico_causal(df, TARGET)
    df.attrs["linhas_removidas"] = n0 - len(df)
    return df
