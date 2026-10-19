# langchain-vera

[LangChain](https://www.langchain.com) integration for [Vera](https://vera.waggle.ae), the
web-scraping and search API from Waggle: two tools for agents, a document loader for RAG, and a small
typed client.

**New to Vera?** [Sign up free](https://vera.waggle.ae/signup) and get 3,000 credits. No card needed.

| Class | What it does | Credits |
| --- | --- | --- |
| `VeraScrapeTool` | Reads a public page and returns its main content as **Markdown** | 1 (5 with `render=True`) |
| `VeraSearchTool` | Web, news or video search | 3 per search, whatever `max_results` is (1 to 100) |
| `VeraLoader` | Loads pages as `Document`s with `source` metadata | 1 per page (5 with `render=True`) |
| `VeraClient` | Typed sync and async client: scrape, search, contacts (1), products (1) | per call, as above |

A request Vera refuses or cannot complete is not charged.

## Install

```bash
pip install -U langchain-vera
# or
uv add langchain-vera
```

Python 3.10 or newer, `langchain-core` 1.x.

## Authentication

1. [Create a free Vera account](https://vera.waggle.ae/signup) and confirm your email.
2. In your Vera workspace open **API keys** and create a key (`vera_live_...`). Copy it when it
   appears; Vera keeps only a hash.
3. Export it:

   ```bash
   export VERA_API_KEY="vera_live_..."
   ```

Every class reads `VERA_API_KEY` by default. Pass `api_key="..."` to override it. Requests go to
`https://vera.waggle.ae/api/vera/v1` with `Authorization: Bearer <key>`. The key is a `SecretStr` and
never appears in a `repr`.

## Tools

```python
from langchain_vera import VeraScrapeTool, VeraSearchTool

search = VeraSearchTool(max_results=5)
scrape = VeraScrapeTool(max_characters=20_000)

print(search.invoke({"query": "open source web scrapers"}))
print(scrape.invoke({"url": "https://example.com"}))
```

`VeraScrapeTool` returns the page as Markdown. `VeraSearchTool` returns a numbered list of titles,
URLs and snippets. Both are plain `BaseTool`s with an `args_schema`, so any tool-calling model can use
them. The model chooses only the `url` or the `query`; everything that costs credits or context is a
setting you control:

| Tool | Settings |
| --- | --- |
| `VeraScrapeTool` | `render` (5 credits instead of 1, for JavaScript-built pages), `max_characters` (cut long pages to protect the context window) |
| `VeraSearchTool` | `search_type` (`"web"`, `"news"`, `"videos"`), `max_results`, `country` (`"ae"`), `language` (`"ar"`) |
| both | `api_key`, `base_url`, `timeout` (seconds, default 60), `client` (your own `VeraClient`) |

### With an agent

```python
from langchain.agents import create_agent  # pip install langchain

agent = create_agent(
    llm, tools=[VeraSearchTool(max_results=5), VeraScrapeTool(max_characters=20_000)]
)
agent.invoke({"messages": [("user", "Find the Vera API docs and summarise how credits work.")]})
```

`llm` is any tool-calling chat model.

### Credits and metadata as an artifact

Both tools use `response_format="content_and_artifact"`. Invoked with a tool call (which is what an
agent does), the `ToolMessage` carries the credits used and more in its `artifact`:

```python
message = scrape.invoke(
    {"type": "tool_call", "name": "vera_scrape", "id": "1", "args": {"url": "https://example.com"}}
)
message.content  # the Markdown
message.artifact  # {"url": ..., "title": ..., "status_code": 200, "credits_used": 1, "request_id": ..., "elapsed_ms": ...}
```

For search, the artifact is the full response data (`results`, `ai_overview`, `knowledge_graph` and
other extras when present) plus `credits_used` and `request_id`.

### Errors inside an agent

A page Vera refuses (for example robots.txt disallows it, or the URL is not public) comes back to the
model as an error message, so it can try something else: the tools raise a `ToolException` with
`handle_tool_error=True`. Set `handle_tool_error=False` to raise instead. Errors the model cannot fix
(bad key, no credits, rate limit, server trouble) always raise a typed error, see
[Errors](#errors).

## Document loader

```python
from langchain_vera import VeraLoader

loader = VeraLoader(
    ["https://example.com", "https://example.org"],
    continue_on_failure=True,  # skip a page Vera cannot read, with a logged warning
)
docs = loader.load()  # also: loader.lazy_load(), await loader.aload()
docs[0].page_content  # Markdown
docs[0].metadata
# {"source": "https://example.com", "title": "Example Domain", "description": ..., "language": "en",
#  "canonical_url": ..., "status_code": 200, "request_id": ..., "credits_used": 1}
```

`metadata["source"]` is the URL you asked for. Other keys appear only when Vera returned them (and
`final_url` when the page redirected). `continue_on_failure` skips pages that are refused, unreachable
or hit a server error; a bad key, no credits or a rate limit always raise, since every page would fail
the same way. Pass `render=True` for JavaScript-built pages (5 credits each).

For RAG, split with [`langchain-text-splitters`](https://pypi.org/project/langchain-text-splitters/):

```python
from langchain_text_splitters import MarkdownTextSplitter

chunks = MarkdownTextSplitter(chunk_size=1000).split_documents(docs)
```

## Client

```python
from langchain_vera import VeraClient

client = VeraClient()  # reads VERA_API_KEY

page = client.scrape("https://example.com")  # 1 credit
page.data["markdown"], page.credits_used, page.request_id

html = client.scrape("https://example.com", format="html")  # data["html"]
spa = client.scrape("https://spa.example.com", render=True, wait_for=".results")  # 5 credits
prices = client.scrape(
    "https://shop.example.com/item",
    fields={
        "title": "h1",
        "price": ".price",
        "image": {"selector": "img.hero", "attribute": "src"},
    },
)  # prices.data["extracted"]

client.search("vera scraping", kind="news", limit=20, country="ae", language="en")  # 3 credits
client.contacts("https://example.com/contact")  # 1 credit
client.products("https://shop.example.com/item")  # 1 credit
```

Every method has an async twin (`ascrape`, `asearch`, `acontacts`, `aproducts`). Pass
`idempotency_key="..."` (16 to 128 letters, digits, `_ . : -`) to make a retried call free: the same
key and request within 24 hours replays the first answer, with `credits_used == 0` and
`idempotency_replayed == True`.

How the arguments map to the API:

| Argument | API field |
| --- | --- |
| `format="markdown"` | `response_format: "markdown"` |
| `format="html"` | `return_html: true` |
| `render=True` | `render: true` |
| `fields={name: selector or {selector, attribute, multiple}}` | `extract` |
| `min_quality_score`, `wait_ms`, `wait_for` | the same names |

`fields` are CSS rules read from the static page: up to 10, one tag, `#id`, `.class` or `[attribute]`
selector each, and not with `render=True` or `format="html"`. Invalid arguments raise `ValueError`
before anything is sent.

There are no automatic retries: a retried scrape could be charged twice. Retry yourself, with an
idempotency key. Timeouts are explicit: 60 seconds by default with a 10 second connect timeout; pass
`timeout=` (seconds or an `httpx.Timeout`).

## Errors

All errors derive from `langchain_vera.VeraError`.

| Error | When | Notes |
| --- | --- | --- |
| `VeraConfigurationError` | No API key | Also a `ValueError` |
| `VeraAuthenticationError` | 401 | Bad, expired or revoked key |
| `VeraInsufficientCreditsError` | 402 | Top up at <https://vera.waggle.ae/pricing> |
| `VeraPermissionError` | 403 | For example, email not confirmed yet |
| `VeraRequestError` | 422 and other 4xx | Refused request or unreadable page, `.code` names failed fetches (`robots_disallowed`...). Not charged |
| `VeraConflictError` | 409, 410 | Idempotency key reused or still running |
| `VeraRateLimitError` | 429 | `.retry_after` seconds, also the free-account daily limit |
| `VeraServerError` | 5xx | Try again shortly |
| `VeraTimeoutError`, `VeraConnectionError` | No answer | `VeraTimeoutError` is a `VeraConnectionError` |
| `VeraResponseError` | 2xx with an unexpected body | |

`VeraAPIError` is the parent of the HTTP ones. Each carries `.status_code`, `.code`, `.request_id`,
`.credits_used` (0 means not charged) and `.retry_after`.

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy langchain_vera tests
uv run pytest                      # unit tests, HTTP mocked, no network
```

The unit tests include LangChain's standard `ToolsUnitTests` for both tools and a contract test that
checks every request body against a snapshot of Vera's published OpenAPI request schemas
(`tests/fixtures`, from <https://vera.waggle.ae/api/vera/v1/openapi.json>).

One live test is skipped by default because it spends a credit and needs a key:

```bash
VERA_API_KEY=vera_live_... uv run pytest -m live
```

## Publishing (maintainers)

Replace `OWNER` in the `[project.urls]` of `pyproject.toml` with the GitHub account that will host the
repository, then:

```bash
uv sync
uv run ruff check . && uv run mypy langchain_vera tests && uv run pytest   # all green first
rm -rf dist && uv build                      # dist/langchain_vera-<version>.tar.gz and .whl
uvx twine check dist/*                        # validates the package metadata and README
```

Optional dry run on TestPyPI (create a token at <https://test.pypi.org/manage/account/token/>):

```bash
uv publish --publish-url https://test.pypi.org/legacy/ --token "$TEST_PYPI_TOKEN"
```

Publish to PyPI (create an API token at <https://pypi.org/manage/account/token/>; scope it to the
project once it exists):

```bash
uv publish --token "$PYPI_API_TOKEN"
# or, with twine:
TWINE_USERNAME=__token__ TWINE_PASSWORD="$PYPI_API_TOKEN" uvx twine upload dist/*
```

Then tag the release:

```bash
git tag v0.1.0 && git push origin main --tags
```

To release from GitHub Actions without a token instead, add a PyPI trusted publisher (repository,
workflow `publish.yml`, environment `pypi`) and publish a GitHub release: `.github/workflows/publish.yml`
builds and runs `uv publish --trusted-publishing always`.

To release a new version, bump `version` in `pyproject.toml`, add it to `CHANGELOG.md`, and repeat.

## Licence

[MIT](LICENSE)
