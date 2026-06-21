# Changelog

## [Unreleased] — 2026-06-21

### Adicionado

#### Benchmark vs BTC Buy-and-Hold

- **`engine/benchmark.py`** — módulo puro que constrói a curva buy-and-hold de BTC alinhada
  aos snapshots de equity do Soros (mesmo capital inicial, mesma janela, forward-fill para
  lacunas no histórico de preços). Inclui `load_equity_snapshots` e `load_btc_closes` para
  leitura read-only do SQLite.

- **`engine/metrics.py`** — métricas comparativas entre as duas curvas: retorno total,
  Sharpe anualizado (risk-free = 0, fator de anualização derivado da cadência mediana dos
  snapshots) e max drawdown. Sinaliza `sharpe_conclusive = False` para amostras com menos
  de 30 pontos.

- **`dashboard/app/api/benchmark/route.ts`** — endpoint `GET /api/benchmark` que lê o banco,
  monta as séries via `engine/benchmark.py` e retorna JSON com as duas curvas alinhadas e
  o objeto de métricas comparativas.

- **`dashboard/lib/benchmark.ts`** — espelho TypeScript de `engine/benchmark.py` e
  `engine/metrics.py`, usado pelo dashboard e pelos testes Bun.

- **Painel de benchmark no dashboard** — gráfico de overlay (Soros vs BTC buy-and-hold,
  mesmo eixo), tabela de métricas lado a lado, indicador de "batendo/perdendo o benchmark"
  em pt-BR. Tooltips explicam Sharpe, drawdown e o que é o benchmark.

### Testes

- Cobertura da matemática de benchmark e métricas em `dashboard/dashboard.test.ts` (73 testes):
  - Histórico curto/vazio: n=1 (ponto único) e n=2 (dois pontos, Sharpe null por falta de
    variância suficiente).
  - Lacunas no histórico de BTC: forward-fill corretamente propaga o último preço conhecido;
    múltiplas lacunas consecutivas; série inteira forward-filled a partir de um único preço.
  - Casos de erro: snapshots vazios, btcCloses vazio, nenhum BTC cobre a janela de equity.
  - Timestamps irregulares: mediana de intervalo reflete a cadência real.
  - Invariantes: `riskFreeRate = 0`, `sharpeConclusive` ligado a `n >= 30`, drawdown ≤ 0.
