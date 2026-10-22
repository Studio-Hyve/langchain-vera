# langchain-vera

[![CI](https://github.com/Studio-Hyve/langchain-vera/actions/workflows/ci.yml/badge.svg)](https://github.com/Studio-Hyve/langchain-vera/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/langchain-vera)](https://pypi.org/project/langchain-vera/)
[![Python](https://img.shields.io/pypi/pyversions/langchain-vera)](https://pypi.org/project/langchain-vera/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

[LangChain](https://www.langchain.com) integration for [Vera](https://vera.waggle.ae), the web data
and search API from Waggle: seven tools for agents (read a page, search the web, read a full results
page on five engines, keyword ideas, AI answers, a GEO query radar and a company footprint), a document
loader for RAG, and a small typed client.

**New to Vera?** [Sign up](https://vera.waggle.ae/signup) for free credits. No card needed.

| Class | What it does | Credits |
| --- | --- | --- |
| `VeraScrapeTool` | Reads a public page and returns its main content as **Markdown** | 1 (5 with `render=True`) |
| `VeraSearchTool` | Web, news or video search | 3 per search, whatever `max_results` is (1 to 100) |
| `VeraSerpTool` | A full results page on Google, Bing, DuckDuckGo, Brave or Yahoo: organic results, AI Overview, People Also Ask, optional map listings and shopping | by result |
| `VeraKeywordIdeasTool` | Keyword ideas for a seed, grouped by intent | by result |
| `VeraAiAnswersTool` | Asks ChatGPT, Claude or Gemini a question and reports whether a brand is named: mentions, citations, share of voice | by result, the most expensive |
| `VeraGeoRadarTool` | How likely AI answers are to cite a brand for one query, with forum threads, list articles and question gaps | by result |
| `VeraCompanyFootprintTool` | How a company shows up in search: own properties, profiles, map listings, third-party mentions | by result |
| `VeraLoader` | Loads pages as `Document`s with `source` metadata | 1 per page (5 with `render=True`) |
| `VeraClient` | Typed sync and async client for every route above, plus contacts (1) and products (1) | per call, as above |

A request Vera refuses or cannot complete is not charged.

**"By result" means priced by what comes back.** Vera works out the most a call can cost and holds that
amount first, you are charged for what was actually served, and the rest is given back. `credits_used`
(in the tool artifact and on every `VeraResponse`) is the charge. A results page of up to ten results is
typically 3 credits live and 1 when served from Vera's cache; every further ten results is another page,
and the map listings and shopping add-ons are priced on their own. AI answers are priced per call to each
assistant and cost far more than a search, Claude the most, so the AI answers tool asks one assistant
once unless you say otherwise. On free credits a daily pace also applies to the amount held, so one
expensive call can use up a whole day: a second one the same day raises `VeraRateLimitError`.

## Install

```bash
pip install -U langchain-vera
# or
uv add langchain-vera
```

Python 3.10 or newer, `langchain-core` 1.x.

## Authentication

1. [Create a Vera account](https://vera.waggle.ae/signup) (free credits on sign-up) and confirm your
   email.
2. In your Vera workspace open **API keys** and create a key (`vera_live_...`). Copy it when it
   appears; Vera keeps only a hash.
3. Export it:

   ```bash
   export VERA_API_KEY="vera_live_..."
   ```

Every class reads `VERA_API_KEY` by default. Pass `api_key="..."` to override it. Requests go to the
direct Vera API at `https://vera.waggle.ae/v1` with `Authorization: Bearer <key>`; pass `base_url=` to
point a client, tool or loader somewhere else (for example a staging Vera). The website's `/api/vera`
proxy is not used: it does not carry the newer routes and gives up on a long request after 40 seconds.
The key is a `SecretStr` and never appears in a `repr`.

## Tools

```python
from langchain_vera import VeraScrapeTool, VeraSearchTool

search = VeraSearchTool(max_results=5)
scrape = VeraScrapeTool(max_characters=20_000)

print(search.invoke({"query": "open source web scrapers"}))
print(scrape.invoke({"url": "https://example.com"}))
```

`VeraScrapeTool` returns the page as Markdown. `VeraSearchTool` returns a numbered list of titles,
URLs and snippets. All the tools are plain `BaseTool`s with an `args_schema`, so any tool-calling model
can use them. The model chooses what it asks about (a `url`, a `query`, a `seed`...); everything that
costs credits or context is a setting you control:

| Tool | Settings |
| --- | --- |
| `VeraScrapeTool` | `render` (5 credits instead of 1, for JavaScript-built pages), `max_characters` (cut long pages to protect the context window) |
| `VeraSearchTool` | `search_type` (`"web"`, `"news"`, `"videos"`), `max_results`, `country` (`"ae"`), `language` (`"ar"`) |
| `VeraSerpTool` | `engine` (`"google"`, `"bing"`, `"duckduckgo"`, `"brave"`, `"yahoo"`), `depth` (1 to 100, default 10), `gl`, `hl`, `location`, `local_pack`, `shopping` (Google add-ons), `fresh`, `max_characters` |
| `VeraKeywordIdeasTool` | `limit` (1 to 200, default 50), `expand` (`"none"`, `"questions"`, `"alphabet"`, `"all"`), `gl`, `hl`, `location`, `fresh`, `max_characters` |
| `VeraAiAnswersTool` | `engines` (default `["chatgpt"]`), `samples` (1 to 5, default 1), `brand`, `brand_domain`, `competitors`, `country`, `language`, `city` |
| `VeraGeoRadarTool` | `brand`, `domain`, `depth` (1 to 100, default 30), `location`, `gl`, `hl`, `fresh`, `max_characters` |
| `VeraCompanyFootprintTool` | `depth` (1 to 100, default 20), `location`, `gl`, `hl`, `fresh`, `max_characters` |
| all | `api_key`, `base_url`, `timeout` (seconds), `client` (your own `VeraClient`) |

Settings named `gl` and `hl` are Google's country and language codes, as the API names them, for example
`gl="ae"`, `hl="ar"`. A tool's `timeout` already allows for how long it can take: 60 seconds for the page
and search tools, 50 for a results page, 80 for keyword ideas and the radar, 65 for a company footprint
and 175 for AI answers (the API takes up to about 150).

### Results pages

```python
from langchain_vera import VeraSerpTool

tool = VeraSerpTool(gl="ae", hl="en", depth=10)
print(tool.invoke({"query": "best coffee in dubai"}))  # a Markdown brief for the model

local = VeraSerpTool(location="Dubai", gl="ae", local_pack=True)  # adds the map listings
shops = VeraSerpTool(shopping=True)  # adds Google Shopping
```

The brief covers the organic results (rank, title, URL, snippet, tagged forum, video or shop) and, on
Google, the AI Overview with its citations and People Also Ask. With a tool call the artifact is the
whole results page (`organic`, `serp_features`, `provenance`...).

### Keyword ideas

```python
from langchain_vera import VeraKeywordIdeasTool

tool = VeraKeywordIdeasTool(gl="ae", hl="en", expand="questions")
print(tool.invoke({"seed": "specialty coffee"}))
# Keyword ideas for 'specialty coffee' (42 ideas)
#
# question:
# - how to brew specialty coffee
# ...
```

Ideas come from Google autocomplete, People Also Ask and related searches, each labelled `question`,
`comparison`, `commercial`, `local` or `informational`. Ideas only: no search volumes. `expand` sets how
many autocomplete calls are made: `"none"` is the cheapest, `"all"` the most.

### AI answers

```python
from langchain_vera import VeraAiAnswersTool

tool = VeraAiAnswersTool(
    brand="Roasters",
    brand_domain="roasters.ae",
    competitors=["Other Roasters"],
    engines=["chatgpt"],  # "claude" and "gemini" are separate paid calls
    samples=1,
    country="ae",
)
print(tool.invoke({"prompt": "What are the best specialty coffee roasters in Dubai?"}))
# AI answers for 'What are the best ...?': is Roasters (roasters.ae) named?
# - ChatGPT: Roasters named in 1 of 1 answer (share of voice 40%, average first rank 2, ...)
#     Competitors: Other Roasters: named in 1 of 1, average rank 1
#     Most cited domains: example.com 50%, roasters.ae 25%
```

The model can also pass `brand`, `brand_domain` and `competitors`, which override the tool's. Use
`samples` above 1 for a mention rate. `engines` is never empty: the API asks every assistant when the
field is left out, which is the most expensive way to run this, so the tool always names them. An
assistant that is not set up answers `not_configured` and costs nothing. `"perplexity"` is also accepted
when Vera has it set up. The artifact has the full metrics per assistant (`engines[...]["aggregate"]`).

### GEO radar and company footprint

```python
from langchain_vera import VeraCompanyFootprintTool, VeraGeoRadarTool

radar = VeraGeoRadarTool(brand="Roasters", domain="roasters.ae", location="Dubai", gl="ae")
print(radar.invoke({"query": "best specialty coffee roasters in dubai"}))

footprint = VeraCompanyFootprintTool(location="Dubai", gl="ae")
print(footprint.invoke({"company": "Roasters", "domain": "roasters.ae"}))
```

The radar returns a citation probability score (0 to 100), the forum threads worth joining, the list
articles that name your competitors, the questions your site does not answer yet and a brief for AI
writers. The footprint separates the company's own web properties, its social and reference profiles
(Wikipedia, LinkedIn, Instagram, Crunchbase), its map listings, what third parties say and the questions
people ask about the brand.

> **Defaults.** Leave `location` and `gl` out of the radar or the footprint and the API currently uses
> Dubai and `ae`. Set them for any other market. AI answers currently use `ae` and `en` for `country`
> and `language` when left out.

### With an agent

```python
from langchain.agents import create_agent  # pip install langchain

agent = create_agent(
    llm,
    tools=[
        VeraSerpTool(depth=10, max_characters=20_000),
        VeraScrapeTool(max_characters=20_000),
        VeraKeywordIdeasTool(expand="none"),
    ],
)
agent.invoke(
    {"messages": [("user", "Who ranks for 'specialty coffee dubai' and what do they say?")]}
)
```

`llm` is any tool-calling chat model. Add `VeraAiAnswersTool` only when you want the agent to be able to
spend credits on assistant answers; its description says it is expensive.

### Credits and metadata as an artifact

Every tool uses `response_format="content_and_artifact"`. Invoked with a tool call (which is what an
agent does), the `ToolMessage` carries the credits used and more in its `artifact`:

```python
message = scrape.invoke(
    {"type": "tool_call", "name": "vera_scrape", "id": "1", "args": {"url": "https://example.com"}}
)
message.content  # the Markdown
message.artifact  # {"url": ..., "title": ..., "status_code": 200, "credits_used": 1, "request_id": ..., "elapsed_ms": ...}
```

For search, results page, keyword ideas, AI answers, radar and footprint, the artifact is the full
response data (for example `results`, `organic`, `ideas`, `engines`...) plus `credits_used` and
`request_id`.

### Errors inside an agent

A request Vera refuses (for example robots.txt disallows a page, the URL is not public, or a field the
API rejects) comes back to the model as an error message, so it can try something else: the tools raise
a `ToolException` with `handle_tool_error=True`. Set `handle_tool_error=False` to raise instead. Errors
the model cannot fix (bad key, no credits, rate limit, server trouble or a timeout) always raise a typed
error, see [Errors](#errors). An invalid argument the model chose raises `ValueError` before anything
is sent.

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

# Search tools: priced by what comes back
serp = client.serp("best coffee in dubai", engine="google", gl="ae", hl="en", depth=10)
serp.data["organic"], serp.data["serp_features"], serp.credits_used
client.serp(
    "coffee shop",
    location="Dubai",
    gl="ae",
    features=["organic", "paa", "ai_overview", "local_pack"],
)  # map listings
ideas = client.keyword_ideas("specialty coffee", gl="ae", expand="questions", limit=50)
ideas.data["ideas"]  # [{"keyword": ..., "sources": [...], "intent": ...}]
answers = client.ai_answers(
    "What are the best specialty coffee roasters in Dubai?",
    "Roasters",
    brand_domain="roasters.ae",
    competitors=["Other Roasters"],
    engines=["chatgpt"],
    samples=1,
)  # answers.data["engines"][0]["aggregate"]
client.geo_radar_query(
    "best specialty coffee roasters in dubai",
    brand="Roasters",
    domain="roasters.ae",
    location="Dubai",
    gl="ae",
)
client.company_footprint("Roasters", domain="roasters.ae", location="Dubai", gl="ae")
```

Every method has an async twin (`ascrape`, `asearch`, `acontacts`, `aproducts`, `aserp`,
`akeyword_ideas`, `aai_answers`, `ageo_radar_query`, `acompany_footprint`). Pass
`idempotency_key="..."` (16 to 128 letters, digits, `_ . : -`) to make a retried call free: the same
key and request within 24 hours replays the first answer, with `credits_used == 0` and
`idempotency_replayed == True`. A failed call is replayed too, so retry a failed call with a new key.

How the arguments map to the API:

| Argument | API field |
| --- | --- |
| `format="markdown"` | `response_format: "markdown"` |
| `format="html"` | `return_html: true` |
| `render=True` | `render: true` |
| `fields={name: selector or {selector, attribute, multiple}}` | `extract` |
| `min_quality_score`, `wait_ms`, `wait_for` | the same names |
| `serp(q, engine=...)` | `POST /serp/{engine}` with `q`, `gl`, `hl`, `location`, `depth`, `features`, `fresh` |
| `keyword_ideas(seed, ...)` | `POST /keywords/ideas` with `seed`, `gl`, `hl`, `location`, `limit`, `expand`, `fresh` |
| `ai_answers(prompt, brand, ...)` | `POST /ai/answers` with `prompt`, `brand`, `brand_domain`, `competitors`, `engines`, `samples`, `country`, `language`, `city` |
| `geo_radar_query(query, ...)` | `POST /geo/radar/query` with `query`, `brand`, `domain`, `location`, `gl`, `hl`, `depth`, `fresh` |
| `company_footprint(company, ...)` | `POST /company/footprint` with `company`, `domain`, `location`, `gl`, `hl`, `depth`, `fresh` |

`fields` are CSS rules read from the static page: up to 10, one tag, `#id`, `.class` or `[attribute]`
selector each, and not with `render=True` or `format="html"`. Invalid arguments raise `ValueError`
before anything is sent. `serp` features beyond the default results page are `"local_pack"` (the
Google Maps listings) and `"shopping"`, on Google only; `ai_answers` never sends an empty `engines`.

There are no automatic retries: a retried call could be charged twice. Retry yourself, with an
idempotency key. Timeouts are explicit: 60 seconds by default with a 10 second connect timeout, and the
long search tools wait as long as they need (see [Tools](#tools)); pass `timeout=` (seconds or an
`httpx.Timeout`) to set one for every call.

## Errors

All errors derive from `langchain_vera.VeraError`.

| Error | When | Notes |
| --- | --- | --- |
| `VeraConfigurationError` | No API key | Also a `ValueError` |
| `VeraAuthenticationError` | 401 | Bad, expired or revoked key |
| `VeraInsufficientCreditsError` | 402 | Out of credits: top up at <https://vera.waggle.ae/pricing> |
| `VeraPermissionError` | 403 | For example, email not confirmed yet |
| `VeraRequestError` | 400, 422 and other 4xx | Invalid input, with the API's message (or a page that cannot be read), `.code` names failed fetches (`robots_disallowed`...). Not charged |
| `VeraConflictError` | 409, 410 | Idempotency key reused or still running |
| `VeraRateLimitError` | 429 | `.retry_after` seconds, also the free-account daily pace |
| `VeraServerError` | 5xx | 502 and 503: temporarily unavailable, not charged, `.retry_after` says when to retry. 504: Vera timed out waiting for the service behind the tool, not charged: try again, or ask for less (a smaller `depth`, fewer `samples`, one assistant) |
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

The unit tests include LangChain's standard `ToolsUnitTests` for every tool and a contract test that
checks every request body against a snapshot of Vera's published OpenAPI request schemas and example
bodies (`tests/fixtures`, from <https://vera.waggle.ae/v1/openapi.json>).

The live tests are skipped by default because they spend credits and need a key:

```bash
VERA_API_KEY=vera_live_... uv run pytest -m live
# the AI answers test costs far more than the others and needs a second opt-in:
VERA_API_KEY=vera_live_... VERA_LIVE_EXPENSIVE=1 uv run pytest -m live
```

## Publishing (maintainers)

The repository is <https://github.com/Studio-Hyve/langchain-vera>; `[project.urls]` in `pyproject.toml`
already points there. Create it empty and public in the Studio-Hyve organisation, then push this one:

```bash
git remote add origin git@github.com:Studio-Hyve/langchain-vera.git
git push -u origin main
```

Check the package before releasing:

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

Release from GitHub Actions without a token, using PyPI trusted publishing:

1. On pypi.org open **Your account > Publishing** and add a *pending publisher* (the project does not
   exist yet): project name `langchain-vera`, owner `Studio-Hyve`, repository `langchain-vera`,
   workflow `publish.yml`, environment `pypi`.
2. In the GitHub repository open **Settings > Environments** and create an environment named `pypi`
   (add required reviewers if you want a human to approve each release).
3. Publish a GitHub release (tag `v0.1.0`): `.github/workflows/publish.yml` builds, runs
   `twine check` and `uv publish --trusted-publishing always`.

Or publish once with a token (create an API token at <https://pypi.org/manage/account/token/>; scope it
to the project once it exists):

```bash
uv publish --token "$PYPI_API_TOKEN"
# or, with twine:
TWINE_USERNAME=__token__ TWINE_PASSWORD="$PYPI_API_TOKEN" uvx twine upload dist/*
```

Then tag the release (skip this if you released from a GitHub release, which created the tag):

```bash
git tag v0.1.0 && git push origin main --tags
```

To release a new version, bump `version` in `pyproject.toml`, add it to `CHANGELOG.md`, and repeat.

## Licence

[MIT](LICENSE)
