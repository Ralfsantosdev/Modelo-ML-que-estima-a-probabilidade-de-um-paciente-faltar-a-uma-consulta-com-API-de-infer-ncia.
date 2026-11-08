"""Gera CSV no esquema EXATO do Kaggle (com suas sujeiras) para testar o adaptador sem rede."""
import numpy as np
import pandas as pd


def make_fake_kaggle(path, n=9000, n_pac=3000, seed=0):
    rng = np.random.default_rng(seed)
    dia = pd.Series(pd.Timestamp("2016-04-29") + pd.to_timedelta(rng.integers(0, 40, n), unit="D"))
    lead = np.minimum(rng.exponential(10, n).astype(int), 100)
    sched = (dia - pd.to_timedelta(lead, unit="D")
             + pd.to_timedelta(rng.integers(7 * 3600, 18 * 3600, n), unit="s"))
    sched = pd.Series(sched)
    pac = rng.integers(0, n_pac, n)
    sms = ((lead >= 3) & (rng.random(n) < 0.6)).astype(int)
    lat = rng.normal(0, 1, n_pac)[pac]
    p = 1 / (1 + np.exp(-(-1.6 + 0.03 * np.minimum(lead, 40) + 0.7 * lat - 0.2 * sms)))
    nos = np.where(rng.random(n) < p, "Yes", "No")
    df = pd.DataFrame({
        "PatientId": (pac + 1) * 1e11 + 12345.0,          # float, como no original
        "AppointmentID": 5_000_000 + np.arange(n),
        "Gender": rng.choice(["F", "M"], n, p=[0.65, 0.35]),
        "ScheduledDay": sched.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "AppointmentDay": dia.dt.strftime("%Y-%m-%dT00:00:00Z"),
        "Age": np.clip(rng.normal(37, 22, n), 0, 100).astype(int),
        "Neighbourhood": rng.choice(["JARDIM DA PENHA", "MARIA ORTIZ", "ITARARÉ", "CENTRO", "RARO BAIRRO"],
                                    n, p=[0.3, 0.3, 0.2, 0.19, 0.01]),
        "Scholarship": rng.binomial(1, 0.1, n), "Hipertension": rng.binomial(1, 0.2, n),
        "Diabetes": rng.binomial(1, 0.07, n), "Alcoholism": rng.binomial(1, 0.03, n),
        "Handcap": rng.choice([0, 1, 2, 3, 4], n, p=[0.98, 0.015, 0.003, 0.001, 0.001]),
        "SMS_received": sms, "No-show": nos,
    })
    df.loc[0, "Age"] = -1                                   # idade inválida
    df.loc[1, "Age"] = 200                                  # idade absurda
    df.loc[2, "ScheduledDay"] = (pd.Timestamp(df.loc[2, "AppointmentDay"][:10]) + pd.Timedelta(days=3)
                                 ).strftime("%Y-%m-%dT10:00:00Z")   # agendado DEPOIS da consulta
    df.to_csv(path, index=False)
    return path
