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

#### A/B de sentimento: backtest histórico + forward shadow ao vivo

- **`sentiment/fear_greed_history.py`** — busca e indexa o histórico do Fear & Greed Index
  (alternative.me, limit=0 para cobertura máxima), com cache em disco e lookup por
  backward-fill para datas faltantes.

- **`backtest/ab_runner.py`** — runner A/B: roda o backtest engine duas vezes (OFF e ON com
  F&G histórico injetado por barra) e retorna `ABResult` com as duas curvas de equity, as
  métricas reusando `engine/benchmark.py`, e `fng_coverage_pct` (datas com leitura exata,
  sem contar as preenchidas por backward-fill).

- **`engine/shadow_tracker.py`** — forward shadow scoring: a cada ciclo do bot, computa as
  duas variantes do composite (real e shadow keyless), mantém duas trilhas de equity virtual
  em `forward_shadow_snapshots` e garante isolamento (falha do shadow não derruba o ciclo real).

- **`dashboard/app/api/backtest-ab/route.ts`** — endpoint on-demand `GET /api/backtest-ab`
  para o backtest A/B histórico (off vs on com F&G); nunca computado no page load.

- **`dashboard/app/api/forward-ab/route.ts`** — endpoint para as duas curvas ao vivo (real
  vs shadow) com flag de amostra pequena quando n < 30.

- **Views do dashboard** — overlay das duas curvas de equity (backtest histórico e forward
  shadow) com tabela de métricas lado a lado, caveat de F&G-only para o backtest, e
  indicador de amostra pequena para o forward.

### Testes

- Cobertura da matemática de benchmark e métricas em `dashboard/dashboard.test.ts` (83 testes):
  - Histórico curto/vazio: n=1 (ponto único) e n=2 (dois pontos, Sharpe null por falta de
    variância suficiente).
  - Lacunas no histórico de BTC: forward-fill corretamente propaga o último preço conhecido;
    múltiplas lacunas consecutivas; série inteira forward-filled a partir de um único preço.
  - Casos de erro: snapshots vazios, btcCloses vazio, nenhum BTC cobre a janela de equity.
  - Timestamps irregulares: mediana de intervalo reflete a cadência real.
  - Invariantes: `riskFreeRate = 0`, `sharpeConclusive` ligado a `n >= 30`, drawdown ≤ 0.

- Cobertura da matemática do A/B e casos de borda em `tests/test_ab_runner.py` (+20 novos):
  - **Histórico F&G com buraco**: lacuna no meio do período não causa crash; coverage conta
    apenas datas exatas (não as preenchidas por backward-fill); backward-fill de greed
    extremo diverge do comportamento neutral.
  - **Variante sem trades**: `signal_threshold > 1.0` garante 0 trades em ambas as variantes;
    equity curve permanece flat em `initial_capital`; métricas são zero; coverage ainda
    computada corretamente.
  - **Amostra curta**: n=1 e n=2 não crasham; equity curve tem comprimento correto; métricas
    retornam 0 graciosamente (momentum/volatility precisam de ≥26 candles).
