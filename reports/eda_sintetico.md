# Análise exploratória e auditoria de vazamento (sintetico)

- Linhas: **25,000** | Pacientes: **5,900**
- Taxa de faltas: **25.6%**
- Período: 2024-01-01 a 2025-06-30
- Nulos em `taxa_faltas_previa` (1ª consulta): 23.7%
- Linhas com paciente repetido: 98.5% (exige split temporal/por grupo)

## Faltas por segmento

**tipo_consulta**

| tipo_consulta   |   taxa_faltas |
|:----------------|--------------:|
| cardiologia     |         0.247 |
| clinica_geral   |         0.256 |
| exame           |         0.3   |
| pediatria       |         0.25  |
| retorno         |         0.217 |

**dia_semana**

|   dia_semana |   taxa_faltas |
|-------------:|--------------:|
|            0 |         0.274 |
|            1 |         0.24  |
|            2 |         0.265 |
|            3 |         0.232 |
|            4 |         0.263 |
|            5 |         0.259 |
|            6 |         0.261 |

**sms_enviado**

|   sms_enviado |   taxa_faltas |
|--------------:|--------------:|
|             0 |         0.26  |
|             1 |         0.253 |

**bolsa_familia**

|   bolsa_familia |   taxa_faltas |
|----------------:|--------------:|
|               0 |         0.253 |
|               1 |         0.285 |

**faixa_antecedencia**

| faixa_antecedencia   |   taxa_faltas |
|:---------------------|--------------:|
| (-1, 0]              |         0.241 |
| (0, 3]               |         0.226 |
| (3, 7]               |         0.223 |
| (7, 14]              |         0.243 |
| (14, 30]             |         0.295 |
| (30, 200]            |         0.376 |

## Auditoria de vazamento (AUC univariado)

|                         |   auc |
|:------------------------|------:|
| duracao_atendimento_min | 1     |
| checkin_realizado       | 0.996 |
| taxa_faltas_previa      | 0.561 |
| historico_faltas        | 0.556 |
| antecedencia_dias       | 0.553 |
| idade                   | 0.538 |
| distancia_km            | 0.524 |
| bolsa_familia           | 0.508 |
| hora_consulta           | 0.507 |
| tipo_consulta           | 0.506 |
| hipertensao             | 0.505 |
| sms_enviado             | 0.505 |
| sexo                    | 0.504 |
| historico_consultas     | 0.504 |
| diabetes                | 0.502 |
| dia_semana              | 0.501 |

**Suspeitas (AUC >= 0.90): ['duracao_atendimento_min', 'checkin_realizado']** → excluídas das features.

Consultas duplicadas (paciente+dia): 114

### Outros riscos tratados
- `historico_faltas` e `taxa_faltas_previa` usam só consultas anteriores (cumsum − atual).
- Split **temporal** (treino < 2025-01, validação 2025-T1, teste 2025-T2).
- Limiar escolhido na validação, nunca no teste.
