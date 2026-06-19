# soros

Bot de trading algoritmico pessoal e automatizado, focado em cripto (Binance). O suporte
a acoes (Alpaca/yfinance) existe mas vem desligado por default (cripto-only).
Duas camadas: sinais deterministicos (modelos validados academicamente) e analise de
sentimento por LLM (Claude). Storage local em SQLite. Paper trading obrigatorio antes
de qualquer capital real; execucao real toggleavel por classe (default desligada).

Construido com o ecossistema os-eco (seeds, mulch, canopy) orquestrado pelo warren.
O codigo e gerado pela esteira a partir do plano seeds: rode `sd ready` para ver o
trabalho desbloqueado.

---

## Configuracao rapida

Copie `.env.example` para `.env` e preencha os valores necessarios:

```bash
cp .env.example .env
```

---

## Variaveis de ambiente

### Credenciais de exchange

| Variavel | Descricao |
|---|---|
| `BINANCE_API_KEY` | Chave da API Binance (necessaria se `CRYPTO_LIVE=true`) |
| `BINANCE_SECRET` | Secret da API Binance |
| `ALPACA_API_KEY` | Chave da API Alpaca (necessaria se `STOCKS_LIVE=true`) |
| `ALPACA_SECRET` | Secret da API Alpaca |
| `ALPACA_BASE_URL` | Endpoint Alpaca (default: paper trading) |

### Toggles de execucao

| Variavel | Default | Descricao |
|---|---|---|
| `CRYPTO_LIVE` | `false` | Executa ordens reais na Binance. Exige 48h+ de paper trading validado. |
| `STOCKS_LIVE` | `false` | Executa ordens reais na Alpaca. Exige 48h+ de paper trading validado. |
| `SENTIMENT_ENABLED` | `false` | Ativa o runner de sentimento. Exige acesso a subscricao Claude. |

### Simbolos e universo autonomo

| Variavel | Default | Descricao |
|---|---|---|
| `CRYPTO_SYMBOLS` | _(vazio)_ | Override opcional: simbolos cripto sempre incluidos alem do universo autonomo. Vazio = universo 100% por regra (market cap + gems). |
| `STOCK_SYMBOLS` | _(vazio)_ | Simbolos de acoes pinned. Vazio = cripto-only; preencha para operar acoes. |
| `CRYPTO_WATCHLIST` | _(vazio)_ | Candidatos cripto adicionais avaliados pelo screener. |
| `STOCK_WATCHLIST` | _(vazio)_ | Candidatos de acoes adicionais avaliados pelo screener. |

### Universo autonomo — base por market cap

| Variavel | Default | Descricao |
|---|---|---|
| `MARKETCAP_TOP_N` | `20` | Numero de moedas top-N por market cap (CoinGecko, keyless). |
| `MARKETCAP_REFRESH_SECS` | `3600` | Intervalo (segundos) de refresh da lista de market cap. |

### Gem scanner — candidatos de ignicao

| Variavel | Default | Descricao |
|---|---|---|
| `GEM_VOLUME_SURGE_MULTIPLIER` | `2.0` | Multiplicador minimo de volume sobre a media para qualificar como gem. |
| `GEM_ROC_MIN_PCT` | `3.0` | Rate-of-change minimo (%) na janela curta para qualificar. |
| `GEM_TOP_N` | `5` | Maximo de gems surfacados por ciclo. |
| `GEM_MIN_VOLUME_USD` | `500000` | Piso de liquidez (USD 24h) para candidatos gem. |
| `IGNITION_WEIGHT` | `0.15` | Peso do sinal de ignicao no composite (0.0 = desabilitado). |
| `GEM_POSITION_SIZE_PCT` | `0.05` | Fracao do equity alocada por posicao gem (menor que base; deve ser <= POSITION_SIZE_PCT). |
| `GEM_TRAILING_STOP_PCT` | `0.05` | Trailing stop para posicoes gem (fracao, ex: 0.05 = 5%). 0.0 = desabilitado. |

### Screener

| Variavel | Default | Descricao |
|---|---|---|
| `SCREENER_ENABLED` | `false` | Ativa o screener. Quando `false`, opera apenas os pinned. |
| `SCREENER_TOP_N` | `3` | Maximo de simbolos da watchlist selecionados pelo screener. |
| `SCREENER_MIN_VOLUME_USD` | `1000000` | Volume notional 24h minimo (USD) para qualificar. |

### Fontes de sentimento

Sentimento cripto e totalmente keyless: Fear & Greed Index (alternative.me) e
votes de comunidade por moeda (CoinGecko `/coins/{id}`). Sem nenhuma chave
necessaria para cripto — o sinal sempre funciona.

| Variavel | Default | Descricao |
|---|---|---|
| `FINNHUB_API_KEY` | _(vazio)_ | Habilita scores de sentimento por ticker (acoes) via Finnhub. Sem chave → score neutro. |

### Coleta de dados

| Variavel | Default | Descricao |
|---|---|---|
| `WATCHLIST_OHLCV_LIMIT` | `50` | Candles coletados por simbolo da watchlist (janela curta). |
| `LOOP_INTERVAL_SECONDS` | `3600` | Intervalo entre ticks do loop principal (segundos). |

### Capital e custos

| Variavel | Default | Descricao |
|---|---|---|
| `INITIAL_CAPITAL` | `10000` | Capital inicial (USD) para calculo de P&L em paper mode. |
| `FEE_PCT` | `0.001` | Taxa por operacao (0.1%). |
| `SLIPPAGE_PCT` | `0.0005` | Slippage estimado por operacao (0.05%). |
| `POSITION_SIZE_PCT` | `0.10` | Fracao do equity alocada por posicao (10%). |

### Logging

| Variavel | Default | Descricao |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Nivel de log (DEBUG, INFO, WARNING, ERROR). |
| `DB_PATH` | `data/soros.db` | Caminho para o arquivo SQLite. |

---

## Backtest

Execute o backtest sobre precos historicos armazenados no SQLite:

```bash
# Simbolos explicitos
python -m backtest --symbols BTC/USDT:crypto,AAPL:stocks --start 2024-01-01 --end 2024-12-31

# Usando o screener para determinar os simbolos (reusa a mesma logica do loop live)
python -m backtest --screener --start 2024-01-01 --end 2024-12-31

# Universo autonomo completo: market-cap base ∪ gem candidates + screener
python -m backtest --assembler --start 2024-01-01 --end 2024-12-31

# Salvar resultados em JSON
python -m backtest --symbols BTC/USDT:crypto --start 2024-01-01 --end 2024-12-31 --output json --out resultado
```

O backtest replica o pipeline deterministico completo (momentum + volatility + funding + ignition).
Sentimento e excluido (sem replay de LLM). Quando `--screener` e usado, a lista de
simbolos vem de `engine.screener.screen()`. Quando `--assembler` e usado, o universo e
construido via `data.assembler.assemble_universe()` (market-cap base ∪ gem candidates com
DEX boost), passado ao screener — mesma logica do loop ao vivo com SCREENER_ENABLED=true.

---

## Sweep de parâmetros (robustez)

O sweep roda o backtest sobre um grid de valores configuravel e grava as metricas por valor
em `sweep_results`. E disparado on-demand — nunca no carregamento da pagina.

```bash
# Grid padrao de SIGNAL_THRESHOLD (0.15 / 0.20 / 0.25 / 0.30 / 0.35)
python -m backtest.sweep --symbols BTC/USDT:crypto --start 2024-01-01 --end 2024-12-31

# Grid customizado via argumento
python -m backtest.sweep --screener --start 2024-01-01 --end 2024-12-31 --thresholds 0.10,0.20,0.30

# Sweep de outro parametro (ex: position_size_pct)
python -m backtest.sweep --symbols BTC/USDT:crypto --start 2024-01-01 --end 2024-12-31 \
    --param position_size_pct --thresholds 0.05,0.10,0.15,0.20
```

O resultado fica disponivel no dashboard (pagina principal, secao de cenarios) marcando o
valor atualmente em uso. O enquadramento e de **robustez**: o objetivo e ver se o desempenho
e estavel na vizinhanca do valor escolhido — nao escolher o campiao do historico (overfitting).

### Variavel de ambiente

| Variavel | Default | Descricao |
|---|---|---|
| `SWEEP_THRESHOLDS` | `0.15,0.20,0.25,0.30,0.35` | Grid de valores para o sweep de `signal_threshold` (comma-separated). |

### Extensibilidade

O sweep usa `SweepSpec` para declarar qual campo do `BacktestConfig` varrer e sobre quais
valores.  Para varrer um novo parametro (ex: `fee_pct`, `position_size_pct`), basta
instanciar um novo spec:

```python
from backtest.sweep import SweepSpec, run_sweep

spec = SweepSpec(param="position_size_pct", values=[0.05, 0.10, 0.15, 0.20])
rows = run_sweep(cfg_template, spec=spec)
```

Qualquer campo numerico de `BacktestConfig` e suportado sem modificar o runner.

---

## Configuracao em tempo real (Settings)

Alem das variaveis de ambiente, o bot expoe uma camada de override runtime via tabela
SQLite (`settings`). Isso permite ajustar tunables sem reiniciar o processo.

### Precedencia de configuracao

```
env var  >  tabela settings  >  valor default hardcoded
```

O metodo `config.reload_runtime_overrides()` e chamado no inicio de cada ciclo do bot
e aplica essa precedencia para todas as chaves da ALLOWLIST. As variaveis de ambiente
e os valores default sao reavaliados a cada ciclo — sem restart necessario.

### Chaves editaveis (ALLOWLIST)

Somente as chaves listadas em `SETTINGS_ALLOWLIST` (em `config.py`) podem ser
alteradas pelo dashboard. Exemplos de tunables editaveis: `SIGNAL_THRESHOLD`,
`LOOP_INTERVAL_SECONDS`, `SCREENER_ENABLED`, `GEM_TOP_N`, `POSITION_SIZE_PCT`.

### Chaves travadas (LOCKED)

As seguintes chaves sao **permanentemente bloqueadas** e nunca podem ser alteradas
via settings (nem via PUT na API do dashboard):

| Chave | Motivo |
|---|---|
| `CRYPTO_LIVE` | Toggle de execucao real — exige restart deliberado |
| `STOCKS_LIVE` | Toggle de execucao real — exige restart deliberado |
| `SENTIMENT_ENABLED` | Toggle de execucao real — exige restart deliberado |
| `MAX_DRAWDOWN_PCT` | Limite de risco hard — imutavel em runtime |
| `MAX_OPEN_POSITIONS` | Limite de risco hard — imutavel em runtime |

Qualquer tentativa de escrever uma dessas chaves via API retorna HTTP 422.

### API do dashboard

```
GET  /api/settings          # lista todas as variaveis com flag editavel/travado
PUT  /api/settings/:key     # atualiza um tunable (body JSON: {"value": "..."})
```

Valores sao validados contra tipo e range definidos em `SETTINGS_ALLOWLIST` antes
de serem persistidos.

---

## Desenvolvimento

```bash
# Instalar dependencias
pip install -e ".[dev]"

# Rodar todos os testes
pytest

# Qualidade
bun run lint && bun run typecheck
```
