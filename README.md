# soros

Bot de trading algoritmico pessoal e automatizado (cripto na Binance, acoes na Alpaca).
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

### Simbolos e universo

| Variavel | Default | Descricao |
|---|---|---|
| `CRYPTO_SYMBOLS` | `BTC/USDT,ETH/USDT,SOL/USDT` | Simbolos cripto pinned — sempre operados. |
| `STOCK_SYMBOLS` | `AAPL,MSFT,NVDA` | Simbolos de acoes pinned — sempre operados. |
| `CRYPTO_WATCHLIST` | _(vazio)_ | Candidatos cripto adicionais avaliados pelo screener. |
| `STOCK_WATCHLIST` | _(vazio)_ | Candidatos de acoes adicionais avaliados pelo screener. |

### Screener

| Variavel | Default | Descricao |
|---|---|---|
| `SCREENER_ENABLED` | `false` | Ativa o screener. Quando `false`, opera apenas os pinned. |
| `SCREENER_TOP_N` | `3` | Maximo de simbolos da watchlist selecionados pelo screener. |
| `SCREENER_MIN_VOLUME_USD` | `1000000` | Volume notional 24h minimo (USD) para qualificar. |

### Fontes de sentimento (chaves opcionais)

Sem chave, a fonte e ignorada e o score vira neutro — nunca quebra o ciclo.

| Variavel | Default | Descricao |
|---|---|---|
| `CRYPTOPANIC_API_KEY` | _(vazio)_ | Habilita scores de sentimento por moeda (cripto) via CryptoPanic. |
| `FINNHUB_API_KEY` | _(vazio)_ | Habilita scores de sentimento por ticker (acoes) via Finnhub. |

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

# Salvar resultados em JSON
python -m backtest --symbols BTC/USDT:crypto --start 2024-01-01 --end 2024-12-31 --output json --out resultado
```

O backtest replica o pipeline deterministico completo (momentum + volatility + funding).
Sentimento e excluido (sem replay de LLM). Quando `--screener` e usado, a lista de
simbolos vem de `engine.screener.screen()`, garantindo que o backtest reflita a mesma
selecao do loop ao vivo.

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
