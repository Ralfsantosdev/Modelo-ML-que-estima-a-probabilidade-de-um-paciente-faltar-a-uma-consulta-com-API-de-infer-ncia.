# Previsão de não comparecimento (no-show) em consultas

Modelo que estima a probabilidade de um paciente faltar a uma consulta, com API de inferência.

## Execução
```bash
pip install -r requirements.txt
export PYTHONPATH=src:.
# Dataset sintético (padrão)
python -m noshow.eda            # EDA + auditoria de vazamento -> reports/eda_sintetico.md
python -m noshow.train          # 3 modelos + MLflow -> models/, reports/comparacao_modelos.md

# Dataset real (Kaggle: joniarroba/noshowappointments) — baixe o CSV para data/KaggleV2-May-2016.csv
python -m noshow.eda   --dataset kaggle
python -m noshow.train --dataset kaggle     # ou --path /caminho/outro.csv
mlflow ui --backend-store-uri sqlite:///mlflow.db
uvicorn app.main:app --reload   # POST /predict, GET /health
pytest                          # 30 testes
```
`GET /schema` mostra o payload do modelo carregado (sintético ou Kaggle). Exemplo (sintético): `curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"idade":30,"sexo":"F","tipo_consulta":"exame","antecedencia_dias":20,"hora_consulta":9,"dia_semana":0,"distancia_km":12.5}'`

## Dados
### Kaggle (real) — `src/noshow/kaggle.py`
Adaptador do *Medical Appointment No Shows* (110.527 consultas, Brasil, abr–jun/2016). Tratamentos:
`No-show="Yes"` → `faltou=1`; remoção de idade <0 ou >115 e de agendamento posterior à consulta;
antecedência em dias (AppointmentDay não tem horário); `Handcap` (0–4) como `deficiencia`; `hora_agendamento`
(hora do **agendamento**, não da consulta); `bairro` com categorias raras (<300) agrupadas, ajustado só no treino;
split temporal por quantil de dias (70/15/15), sem dividir um dia entre conjuntos.
O adaptador foi validado com um CSV no esquema oficial (`tests/fake_kaggle.py`); **métricas reais ainda precisam ser geradas na sua máquina**.

### Sintético
Dataset **sintético** e reprodutível (`src/noshow/data.py`, seed fixa), 25 mil consultas, ~25,6% de faltas.
Para dados reais (ex.: *Medical Appointment No Shows*, Kaggle), basta substituir `gerar_dataset()` mantendo o esquema de colunas.

## Vazamento de dados
| Risco | Tratamento |
|---|---|
| `checkin_realizado`, `duracao_atendimento_min` (existem só após a consulta; AUC univariado ≈ 1,0) | Detectadas por `auditar_vazamento`, listadas em `LEAKY`, bloqueadas por `assert_no_leakage` e por teste |
| Histórico do paciente incluindo a consulta atual **ou outras do mesmo dia** | `historico_causal` (`history.py`): só dias estritamente anteriores; testes recalculam manualmente e cobrem duplicidade no mesmo dia |
| Mesmo paciente em treino e teste (98,5% das linhas) | Split **temporal** (treino < 2025-01; validação 2025-T1; teste 2025-T2) |
| Limiar/modelo escolhidos no teste | Ambos escolhidos na **validação**; teste usado uma única vez |
| Identificadores (`paciente_id`, `consulta_id`) | Excluídos das features |

## Modelos e resultados — sintético (teste, limiar da validação)
| Modelo | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0,332 | 0,680 | 0,446 | **0,630** |
| Random Forest | 0,326 | 0,646 | 0,433 | 0,625 |
| XGBoost | 0,324 | 0,636 | 0,429 | 0,621 |

Seleção por ROC-AUC na validação → regressão logística. Os três modelos ficaram estatisticamente próximos:
o sinal do dataset é predominantemente linear, e a complexidade extra não compensa.
Classes desbalanceadas tratadas com `class_weight` / `scale_pos_weight`; PR-AUC também é registrada.

## Rastreabilidade
MLflow (`sqlite:///mlflow.db`, experimento `noshow-consultas`): parâmetros, métricas de validação e teste,
modelo serializado com exemplo de entrada, tamanhos dos splits e seed. `models/metadata.json` guarda
modelo escolhido, limiar e lista de features usada pela API.

## Limitações e riscos
1. **Dados sintéticos**: as métricas não representam desempenho em produção; o ROC-AUC de ~0,63 reflete o ruído injetado.
2. **Deriva temporal**: sazonalidade, mudança de políticas (SMS, teleconsulta) e pandemias alteram o padrão de faltas. Reavaliar mensalmente.
3. **Equidade**: `bolsa_familia`, `distancia_km`, `idade` e `sexo` correlacionam com condição socioeconômica. Usar o escore para **reforçar contato** (lembretes, transporte), nunca para negar ou penalizar atendimento. Auditar métricas por subgrupo antes de uso real.
4. **Overbooking**: uso do escore para sobreagendar pode prejudicar quem comparece; calibrar o custo de espera versus vaga ociosa.
5. **Calibração**: pesos de classe distorcem as probabilidades; recalibrar (Platt/isotônica) antes de interpretá-las como risco absoluto.
6. **Cold start**: pacientes sem histórico (≈24% das consultas) têm `taxa_faltas_previa` imputada; a API aceita histórico zero.
7. **Privacidade (LGPD)**: dados de saúde são sensíveis; exigem base legal, minimização, controle de acesso e logs de inferência anonimizados.
8. **Segurança da serialização**: `joblib`/`cloudpickle` executam código ao carregar; carregar apenas artefatos confiáveis.
9. **Kaggle — janela curta**: ~6 semanas; o histórico do paciente é censurado à esquerda (consultas anteriores a abr/2016 são invisíveis) e a maioria tem pouco ou nenhum histórico.
10. **Kaggle — generalização**: uma cidade, 2016, rede pública; não extrapolar para outros contextos sem revalidar. `bairro` pode atuar como proxy socioeconômico.
11. **Sem monitoramento**: não há detecção de drift nem retreino automático; faz parte do próximo passo.
