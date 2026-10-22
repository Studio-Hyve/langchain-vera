# Changelog

## 0.1.0

Initial release.

- **Base URL**: the direct Vera API, `https://vera.waggle.ae/v1` (the website's `/api/vera` proxy lacks
  the newer routes and cuts a request at 40 s). Configurable with `base_url=` on the client, the tools
  and the loader.
- `VeraScrapeTool`: a page as Markdown, 1 credit (5 with `render=True`).
- `VeraSearchTool`: web, news or video search, 3 credits.
- `VeraSerpTool`: a full results page on Google, Bing, DuckDuckGo, Brave or Yahoo
  (`POST /v1/serp/{engine}`), with depth, `gl`, `hl`, location, cache control and the Google
  `local_pack` and `shopping` add-ons. Priced by what comes back.
- `VeraKeywordIdeasTool`: keyword ideas for a seed grouped by intent (`POST /v1/keywords/ideas`), with
  limit and expansion.
- `VeraAiAnswersTool`: what ChatGPT, Claude or Gemini answer for a question and whether a brand is in it
  (`POST /v1/ai/answers`): mentions, citations and share of voice. One assistant and one sample by
  default; an empty engine list is refused so it can never ask them all.
- `VeraGeoRadarTool` (`POST /v1/geo/radar/query`) and `VeraCompanyFootprintTool`
  (`POST /v1/company/footprint`).
- `VeraLoader`: a document loader that turns each scrape into a `Document` with `source` metadata.
- `VeraClient`: a typed httpx client (sync and async) for scrape, search, contacts, products and the
  search tools above, with explicit timeouts (the long calls wait as long as the API needs: 175 s for
  AI answers) and typed errors.
- Errors: 402 out of credits with a link to pricing, 400 and 422 invalid input with the API's message,
  503 temporarily unavailable and 504 timed out (both not charged).
- `credits_used` is on every response and in every tool artifact (the `X-Waggle-Credits-Used` header is
  the authority).
