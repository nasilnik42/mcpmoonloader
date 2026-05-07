# MoonLoader RAG MCP Server

Локальный MCP-сервер для Codex, Google Antigravity, Cursor, Claude Desktop и других MCP-клиентов.

Источник: локальная ChromaDB RAG-база BlastHack Wiki / MoonLoader Lua.

## Возможности

Tools:

- `moonloader_search` — semantic/hybrid поиск по MoonLoader Lua API.
- `moonloader_get_doc` — получить полную документацию по функции/событию/странице.
- `moonloader_list_api` — список всех индексированных API/страниц.
- `moonloader_context` — готовый большой context block для coding agent.
- `moonloader_stats` — статистика базы.

`moonloader_search(limit=0, max_chars_per_source=0)` может вернуть все найденные источники без программного лимита. Практический лимит всё равно зависит от MCP-клиента и контекстного окна модели.

## Установка зависимостей

```bash
pip install mcp chromadb requests
```

## Запуск вручную

```bash
cd /home/ubuntu/blast_moonloader_rag
python moonloader_rag_mcp.py
```

Это stdio MCP server. Обычно вручную он просто ждёт JSON-RPC от клиента.

## Codex config.toml

Codex использует `~/.codex/config.toml`.

Добавь:

```toml
[mcp_servers.moonloader_rag]
command = "python"
args = ["/home/ubuntu/blast_moonloader_rag/moonloader_rag_mcp.py"]
cwd = "/home/ubuntu/blast_moonloader_rag"
env = { GEMINI_API_KEY = "ТВОЙ GEMINI API KEY" }
```

После перезапуска Codex появятся MCP tools от `moonloader_rag`.

## Универсальный mcp.json / Antigravity-style config

Если клиент использует JSON-конфиг формата `mcpServers`, добавь:

```json
{
  "mcpServers": {
    "moonloader_rag": {
      "command": "python",
      "args": ["/home/ubuntu/blast_moonloader_rag/moonloader_rag_mcp.py"],
      "cwd": "/home/ubuntu/blast_moonloader_rag",
      "env": {
        "GEMINI_API_KEY": "ТВОЙ GEMINI API KEY"
      }
    }
  }
}
```

## Как заставить coding agent активно пользоваться MCP

Добавь в системный/проектный prompt агента:

```text
When working with GTA SA, SA:MP, SAMPFUNCS, MoonLoader Lua, or BlastHack Wiki API, always use the moonloader_rag MCP tools before writing or modifying code. Prefer moonloader_context for coding tasks and moonloader_get_doc for exact function names. Do not invent MoonLoader functions; verify signatures and return values through MCP.
```

Для репозитория можно положить в `AGENTS.md`:

```md
# MoonLoader Lua docs

Before writing MoonLoader Lua code, use MCP server `moonloader_rag`:
- `moonloader_context` for task-level context
- `moonloader_get_doc` for exact API/function/event docs
- `moonloader_search` for discovery

Use BlastHack Wiki snippets as primary source of truth. Do not hallucinate functions.
```

## Примеры tool-вызовов

- Найти команду чата:

```json
{"query":"как зарегистрировать команду чата sampRegisterChatCommand пример", "limit":10}
```

- Получить полную страницу:

```json
{"name_or_path":"onSendRpc"}
```

- Получить много источников для задачи:

```json
{"query":"написать moonloader lua скрипт с renderDrawBox и sampRegisterChatCommand", "max_sources":50, "max_total_chars":80000}
```

- Вернуть всё без лимита:

```json
{"query":"render", "limit":0, "max_chars_per_source":0}
```
