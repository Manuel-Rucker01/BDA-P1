# Plan de mejoras v3 — del modelo "que se vence en 9 meses" a un sistema robusto

**Fecha:** 2026-11-05 · **Estado:** en ejecución · **Producción mientras tanto:** `v2final` (features v2, reentrenado nov-2026)

## Diagnóstico (por qué hace falta)
| Problema | Evidencia |
|---|---|
| Historia corta (~1,5 años, un solo régimen) | El modelo pasó de IC útil a **IC ≈ −0,015** en jul–sep 2026 (holdout); el reentrenado con datos frescos vuelve a **+0,070** |
| Universo congelado y con sesgo de supervivencia | ~10% del universo de 1.942 nombres ya no existe; las empresas nuevas nunca son elegibles |
| Solo precio/volumen + grafo estático | Sin resultados, valuación ni calidad; el KG codifica sobre todo sector/tamaño y fue fuente de una fuga |
| Cartera desalineada con la señal | La habilidad está en el ranking amplio (decil 10 − decil 1 = +2,3 pp/mes) pero el top-10 no superó al promedio; la cuenta concentrada perdió 33% |
| Sin monitoreo ni reentrenamiento | La pérdida de señal se descubrió por una caída, no por una alerta |

## Principios
1. **Todo point-in-time y testeado** (tests de "no mirar el futuro" en cada fuente de datos).
2. **Un solo código de features** para entrenar y operar (`features/core.py` y extensiones).
3. **Adopción por evidencia (gate)**: v3 reemplaza a v2 solo si mejora en walk-forward fuera de muestra 2019→2026 con costos, con bootstrap y Deflated Sharpe, sin empeorar el drawdown.
4. **Seguridad intacta**: solo paper, breaker de drawdown, runner sin estado, pruebas antes de cada cambio en producción.

## Workstreams
### WS1 — Universo y precios sin sesgo de supervivencia (Alpaca SIP)
- Universo PIT: activos **activos + inactivos** de Alpaca en NASDAQ/NYSE/AMEX, solo acciones comunes (excluye ETFs, fondos, warrants, units, preferidas, OTC).
- Barras diarias SIP ajustadas **2016 → hoy**, store Parquet incremental, controles de calidad (duplicados, huecos, saltos de split, reutilización de símbolos).
- Elegibilidad por fecha: precio ≥ $3, liquidez mínima (ADV 20d) e historia mínima — la misma regla en entrenamiento y en vivo.

### WS2 — Fundamentales de la SEC (EDGAR XBRL)
- `companyfacts` (bulk) + mapeo ticker↔CIK; panel **por fecha de presentación** (filed + 1 día hábil): nada se usa antes de ser público.
- TTM a partir de 10-Q/10-K; factores: valor (E/P, B/P, S/P, CF/P), calidad (GP/A, ROA, ROE, accruals), crecimiento (ventas/ganancias YoY), apalancamiento, emisión neta de acciones, tamaño PIT.
- **Bloqueante:** la SEC exige un *User-Agent* con contacto real (`SEC_USER_AGENT`).

### WS3 — Pipeline de entrenamiento y evaluación v3 (reproducible, sin Spark en producción)
- Features = precio/volumen (core) + fundamentales + sector; **sin KG por defecto** (ablación con KG sobre el universo común) — sin KG cualquier acción listada es elegible.
- Muestreo semanal de fechas, walk-forward **purgado con embargo** 2018→2026, objetivo = ranking del retorno forward (21 y 30 sesiones; variante neutral por sector).
- Ensemble de GBDTs con hiperparámetros modestos, reentrenamiento final con todos los datos, manifiesto `feature_version: 3`.
- Evaluación: IC/IR por año y régimen, deciles, carteras (long-only amplia y long/short neutral), costos, bootstrap/DSR, vs SPY y equal-weight; **gate de promoción** vs `v2final`.

### WS4 — Cartera y ejecución alineadas con la señal
- **Long-only amplia**: K = 30–50, límite por sector **estricto** (sin relajar), máx. por nombre, colchón de cash.
- **Long/short neutral**: largo decil superior, corto decil inferior, dollar-neutral y balanceado por sector, chequeo de préstamo (ETB), límites de bruto/neto/corto.
- Ejecución: ventas → esperar fills → compras; tamaño mínimo de orden en % del equity.
- Régimen HMM: se elimina de los caminos donde no tiene efecto (o se usa solo si se valida).

### WS5 — Monitoreo y reentrenamiento automáticos
- Guardar predicciones de cada rebalanceo (`ops/predictions/`), IC realizado mes a mes, drift de features (PSI entrenamiento vs vivo), reporte `ops/monitoring.md`.
- Alertas: issue de GitHub / email si el IC rodante cae bajo cero 3 meses o el drift es alto.
- **Reentrenamiento trimestral** automático (refresco incremental → entrenamiento → gate) que abre un PR con el candidato y su reporte.

## Orden y dependencias
```
F0 (hoy)   deploy v2final ───────────────────────────────► producción
WS1 ─┐
WS2 ─┼─► WS3 (entrenar + gate) ─► promoción v3 (si pasa) ─► producción
WS4 ─┘                    ▲
WS5 ──────────────────────┘ (monitorea lo que esté en producción)
```
WS1, WS2, WS4 y WS5 arrancan en paralelo; WS3 cuando estén WS1/WS2 (y sin competir por CPU con el entrenamiento de F0).

## Riesgos y mitigaciones
| Riesgo | Mitigación |
|---|---|
| Sobreajuste por muchos experimentos | Deflated Sharpe con el número de pruebas; gate decidido de antemano |
| Datos de fundamentales sucios/reexpresados | Usar la primera versión presentada (filed) y tests PIT |
| Reutilización de símbolos | Identidad por activo de Alpaca/CIK, no solo por ticker |
| Memoria/cómputo con ~8k acciones × 10 años | Store Parquet por año, muestreo semanal para entrenar |
| Régimen futuro distinto | Monitoreo de IC + reentrenamiento trimestral |
