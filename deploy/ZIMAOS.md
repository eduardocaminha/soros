# Runbook: soros no ZimaOS (x86)

Deploy 24/7 do bot + dashboard em Docker num ZimaOS Intel (x86/amd64), ligado no cabo.

---

## Pré-requisitos

- ZimaOS com Docker e Docker Compose disponíveis (já vêm pré-instalados)
- Git instalado no box
- Acesso SSH ou terminal direto ao box

---

## 1. Primeira instalação

```bash
# Clone o repositório
git clone <url-do-repo> soros
cd soros

# Copie e preencha o .env
cp .env.example .env
# Edite .env — leia as notas abaixo antes de mudar qualquer toggle
```

### Variáveis obrigatórias vs. opcionais

| Variável | Obrigatória? | Nota |
|---|---|---|
| `DB_PATH` | Não — o compose já força `/data/soros.db` | Não edite no container |
| `CRYPTO_LIVE` | Não — default `false` (paper) | Veja seção "Paper keyless" |
| `STOCKS_LIVE` | Não — default `false` | Idem |
| `BINANCE_API_KEY` / `SECRET` | Só se `CRYPTO_LIVE=true` | Deixe vazio pra paper |
| `ALPACA_API_KEY` / `SECRET` | Só se `STOCKS_LIVE=true` | Deixe vazio pra paper |
| `CLAUDE_CODE_OAUTH_TOKEN` | Não — best-effort | Veja seção "Claude" |
| `SENTIMENT_ENABLED` | Não — default `false` | Pode ligar sem token |

---

## 2. Build e subida inicial

**Buildar sempre no próprio box** — arch nativa amd64, sem cross-build:

```bash
docker compose up -d --build
```

Isso faz:
1. Builda a imagem do bot (`Dockerfile`) e do dashboard (`dashboard/Dockerfile`)
2. Cria o volume `soros_data` (SQLite compartilhado em `/data/soros.db`)
3. Sobe os dois serviços com `restart: unless-stopped`

Verificar que estão rodando:

```bash
docker compose ps
docker compose logs bot --tail=30
docker compose logs dashboard --tail=30
```

Dashboard acessível em: `http://<ip-do-box>:3000`

---

## 3. Paper trading keyless

Por default o bot roda em **paper mode** — sem chaves de exchange, sem capital real:

```
CRYPTO_LIVE=false   # default
STOCKS_LIVE=false   # default
```

Nesse modo:
- Ordens são simuladas internamente
- Nenhuma chave de API de exchange é necessária
- O universo de ativos vem do CoinGecko (market cap, keyless)

**Não mude `CRYPTO_LIVE` ou `STOCKS_LIVE` para `true` sem pelo menos 48h de paper validado.**  
Essas variáveis são permanentemente travadas — não editáveis via dashboard em runtime.  
Mudar exige editar `.env` e reiniciar o bot (`docker compose up -d --build` ou restart abaixo).

---

## 4. Claude: best-effort / degradação limpa

O sentimento pode usar a subscription do Claude para debates bull/bear, mas é **opcional**:

```bash
# Opcional: copie o token da subscription
# Localização no Mac: ~/.claude/.credentials.json → campo .oauth_token
CLAUDE_CODE_OAUTH_TOKEN=<token>
```

**Com token ausente ou expirado:** o sentimento cai automaticamente na base keyless  
(Fear & Greed Index + CoinGecko votes). O bot roda normalmente — nenhum erro, nenhuma  
interrupção. O Claude só entra em debates quando `SENTIMENT_ENABLED=true` **e** o token  
está presente e válido.

Para ligar o sentimento sem Claude (base keyless pura):

```bash
SENTIMENT_ENABLED=true
# CLAUDE_CODE_OAUTH_TOKEN= (vazio ou ausente)
```

---

## 5. Loop dev → deploy

Quando um novo commit chegar (warren fez push, você fez uma mudança):

```bash
git pull
docker compose up -d --build
```

O `--build` reconstrói as imagens com o código novo. Os containers são substituídos  
entre ciclos — sem downtime perceptível para o bot (o ciclo é de 1h por default).  
O volume `soros_data` **não é recriado** — o histórico de trades/SQLite é preservado.

Para reiniciar sem rebuild (só quando você mudou variáveis de `.env`):

```bash
docker compose up -d
```

---

## 6. Backup do volume

O SQLite fica no volume Docker `soros_data`, montado em `/data/soros.db` nos containers.

### Backup manual

```bash
# Copia o DB pra fora do volume (snapshot pontual)
docker run --rm \
  -v soros_data:/data \
  -v $(pwd)/backups:/backups \
  alpine \
  cp /data/soros.db /backups/soros_$(date +%Y%m%d_%H%M%S).db
```

### Backup automático com cron (opcional)

No ZimaOS ou no host, adicione um cron job:

```cron
0 3 * * * docker run --rm -v soros_data:/data -v /caminho/backups:/backups alpine cp /data/soros.db /backups/soros_$(date +\%Y\%m\%d).db
```

### Restaurar backup

```bash
# Para o bot antes de restaurar (evita corrupção WAL)
docker compose stop bot

docker run --rm \
  -v soros_data:/data \
  -v $(pwd)/backups:/backups \
  alpine \
  cp /backups/soros_<data>.db /data/soros.db

docker compose start bot
```

---

## 7. Operações comuns

```bash
# Ver logs em tempo real
docker compose logs -f

# Ver logs só do bot
docker compose logs -f bot

# Reiniciar um serviço específico
docker compose restart bot
docker compose restart dashboard

# Parar tudo (preserva volumes)
docker compose down

# Parar e remover volumes (DESTRÓI o SQLite — use só pra reset total)
docker compose down -v

# Ver uso de recursos
docker stats
```

---

## 8. Troubleshooting

**Dashboard não carrega:** verifique se o bot subiu primeiro (o dashboard depende do bot  
para que o volume seja inicializado com o DB).

```bash
docker compose logs bot --tail=50
```

**Erro de permissão em `/data`:** o container do bot cria o diretório/DB na primeira  
execução. Se o volume já existia com outro owner, force a recriação:

```bash
docker compose down -v   # ATENÇÃO: apaga o DB
docker compose up -d --build
```

**Token Claude expirado:** o bot continua rodando em modo keyless. Para renovar:

```bash
# No Mac onde o Claude Code está autenticado:
cat ~/.claude/.credentials.json | python3 -c "import json,sys; print(json.load(sys.stdin)['oauth_token'])"

# Atualize o .env no box e reinicie:
docker compose up -d
```
