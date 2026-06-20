FROM --platform=linux/amd64 oven/bun:1 AS deps
WORKDIR /app
COPY package.json bun.lock ./
RUN bun install --frozen-lockfile

FROM --platform=linux/amd64 oven/bun:1 AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN bun --bun run build

FROM --platform=linux/amd64 oven/bun:1
WORKDIR /app
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

ENV DB_PATH=/data/soros.db
EXPOSE 3000

CMD ["bun", "--bun", "run", "start"]
