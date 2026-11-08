"""API de inferência: probabilidade de falta em consulta (sintético ou Kaggle)."""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ValidationError

MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
INDEX_HTML = Path(__file__).parent / "static" / "index.html"
STATE: dict = {}


def load_artifacts(model_dir: Path = MODEL_DIR) -> None:
    STATE["model"] = joblib.load(model_dir / "best_model.joblib")
    STATE["meta"] = json.loads((model_dir / "metadata.json").read_text())


@asynccontextmanager
async def lifespan(app: FastAPI):
    if "model" not in STATE:
        load_artifacts()
    yield


app = FastAPI(title="No-show API", version="1.1.0", lifespan=lifespan)


class _Base(BaseModel):
    idade: int = Field(ge=0, le=120)
    sexo: Literal["F", "M"]
    antecedencia_dias: int = Field(ge=0, le=365)
    dia_semana: int = Field(ge=0, le=6, description="0=segunda")
    bolsa_familia: Literal[0, 1] = 0
    hipertensao: Literal[0, 1] = 0
    diabetes: Literal[0, 1] = 0
    sms_enviado: Literal[0, 1] = 0
    historico_consultas: int = Field(ge=0, default=0)
    historico_faltas: int = Field(ge=0, default=0)


class ConsultaSintetico(_Base):
    tipo_consulta: Literal["clinica_geral", "cardiologia", "pediatria", "exame", "retorno"]
    hora_consulta: int = Field(ge=0, le=23)
    distancia_km: float = Field(ge=0, le=500)


class ConsultaKaggle(_Base):
    bairro: str = Field(min_length=1, max_length=80)
    alcoolismo: Literal[0, 1] = 0
    deficiencia: int = Field(ge=0, le=4, default=0)
    dia_semana_agendamento: int = Field(ge=0, le=6)
    hora_agendamento: int = Field(ge=0, le=23)


SCHEMAS = {"sintetico": ConsultaSintetico, "kaggle": ConsultaKaggle}


class Predicao(BaseModel):
    probabilidade_falta: float
    previsao_falta: bool
    limiar: float
    modelo: str
    dataset: str


@app.get("/")
def home():
    """Tela web de previsão (formulário -> /predict)."""
    return FileResponse(INDEX_HTML)


@app.get("/health")
def health():
    meta = STATE.get("meta", {})
    return {"status": "ok", "modelo": meta.get("modelo"), "dataset": meta.get("dataset", "sintetico")}


@app.get("/schema")
def schema():
    """JSON Schema do payload esperado pelo modelo carregado."""
    return SCHEMAS[STATE["meta"].get("dataset", "sintetico")].model_json_schema()


@app.post("/predict", response_model=Predicao)
def predict(payload: dict):
    meta = STATE["meta"]
    ds = meta.get("dataset", "sintetico")
    try:
        c = SCHEMAS[ds].model_validate(payload)
    except ValidationError as e:
        raise HTTPException(422, detail=json.loads(e.json(include_url=False, include_context=False)))
    if c.historico_faltas > c.historico_consultas:
        raise HTTPException(422, "historico_faltas não pode exceder historico_consultas")
    row = c.model_dump()
    row["taxa_faltas_previa"] = (c.historico_faltas / c.historico_consultas) if c.historico_consultas else None
    if "bairro" in row:
        row["bairro"] = row["bairro"].strip().upper()
    X = pd.DataFrame([row])[meta["features"]]
    p = float(STATE["model"].predict_proba(X)[0, 1])
    return Predicao(probabilidade_falta=round(p, 4), previsao_falta=p >= meta["limiar"],
                    limiar=round(meta["limiar"], 4), modelo=meta["modelo"], dataset=ds)
