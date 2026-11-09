# Arabic Commerce Copilot

**A shopping and support assistant that understands how Egyptians actually type, in Egyptian Arabic, Arabizi and English. It routes between a free local model and Claude, and cannot show a price that no tool returned.**

![tests](https://img.shields.io/badge/tests-48%20passing-brightgreen)
![retrieval](https://img.shields.io/badge/holdout%20recall%405-1.00-brightgreen)
![mcp](https://img.shields.io/badge/MCP-server-6f42c1)
![providers](https://img.shields.io/badge/LLM-Ollama%20%7C%20Azure%20OpenAI%20%7C%20Claude-1f6feb)

> Clean-room portfolio project. "Kanz" (كنز) is a **fictional** marketplace. The catalogue, help centre and orders are synthetic and generated deterministically. No employer code, data or model weights are used.

```
"3ayez mat3am fe smou7a"          →  restaurant · Smouha        →  search_offers  →  "دي أحسن العروض… 650 جنيه [KZ-0292]"
"ينفع أرجّع الكوبون؟"              →  help: refunds              →  search_help    →  "أيوه، خلال ١٤ يوم… [FAQ-02]"
"فين طلبي ORD-10002؟ آخر ٤ أرقام 7628" →  verified order lookup  →  order_status   →  status, offer, validity
"ignore previous instructions…"   →  blocked before any model sees it
```

## Why it's hard, and what each layer buys

Shoppers write أحسن / احسن / احسن, مدرسة / مدرسه, على / علي. They use Egyptian words (عايز، فين، أرخص، عربية), Arabizi ("3ayez gym fe madinet nasr"), English, or a mix. A naive index treats every variant as a different word.

**Retrieval ablation** (Recall@5 / MRR@10, `uv run python -m copilot.eval.retrieval_eval`):

| configuration | templated (833 q) | handwritten (22 q) | **holdout (12 q)** |
|---|---|---|---|
| BM25, no Arabic handling | 0.51 / 0.55 | 0.51 / 0.58 | 0.29 / 0.57 |
| + normalisation, dialect expansion, Arabizi transliteration | 0.91 / 0.95 | 0.66 / 0.72 | 0.65 / 0.79 |
| dense only (`qwen3-embedding:0.6b`) | 0.84 / 0.84 | 0.68 / 0.73 | 0.80 / 0.96 |
| hybrid (BM25 + dense, reciprocal-rank fusion) | 0.93 / 0.95 | 0.72 / 0.82 | 0.89 / 0.87 |
| **+ query understanding (slots: category, area, city, price)** | **1.00 / 1.00** | **1.00 / 1.00** | **1.00 / 1.00** |
| same, without embeddings (the CI configuration) | 1.00 / 1.00 | 0.95 / 0.95 | 0.90 / 0.92 |

Per-variant detail for the templated set (Arabizi, dialect, typo noise, English, MSA, city-level) is in `artifacts/eval/retrieval.md`.

How to read this honestly:
- **templated** queries are generated from the catalogue's own vocabulary. They're large but optimistic.
- **handwritten** queries were written independently, then **used for failure analysis**. That's how the punctuation bug (`zamalek?`), the Egyptian "سنان" for teeth, and verb forms like أتغدى were found. After those fixes this split is no longer held out.
- **holdout** queries were written after those fixes and never used for tuning. It's the honest number, and at 12 queries it's small. Relevance means "right category in the right place", so this mostly measures query understanding.

## Architecture

```mermaid
flowchart LR
  u[shopper message] --> g1{injection screen}
  g1 -- blocked --> r1[scoped refusal]
  g1 --> agent[agent loop]
  agent --> router{provider router}
  router -->|1. cheap| ol[Ollama · qwen3 4B]
  router -->|2. strong| cl[Claude Opus 5.5<br/>server-side fallbacks]
  router -->|3. always up| sc[scripted rules]
  agent <--> tools[[search_offers · get_offer · search_help<br/>order_status · escalate_to_human]]
  tools --> ret[hybrid retrieval<br/>+ slot parser]
  agent --> g2{grounding checks}
  g2 -->|draft from memory| force[fetch required tool, answer again]
  g2 -->|wrong price| fix[one corrective rewrite]
  g2 -->|still ungrounded| safe[safe "can't confirm" reply]
  g2 --> out[reply + offer ids]
  agent -. OTel GenAI spans, PII-redacted .-> tr[Grafana Tempo / App Insights]
```

- **One tool contract, three surfaces.** The HTTP agent (`/chat`), the MCP server (`python -m copilot.mcp_server`) and the eval all use the same `Toolbox`.
- **Providers are interchangeable.** Ollama and Azure OpenAI go through the OpenAI-compatible client from [azure-local-lab](../azure-local-lab)'s `cloudkit`, so `CLOUDKIT_MODE=azure` switches to Azure OpenAI with Entra ID. Claude goes through the official Anthropic SDK.
- **The Claude adapter follows the current API.** It uses `claude-opus-5-5` with server-side `fallbacks: "default"` for refusals, explicit `effort`, strict tools, and `tool_choice` left at auto. It also replays Claude's own content blocks (thinking included) unchanged across tool calls.
- **Unreachable providers fall through quickly.** A local model call gets 45 s and no retries; the SDK default is 10 min × 3 attempts. In early testing, one stuck generation turned a single reply into a 30-minute wait.

## Grounding: the model chooses the words, rules guarantee the facts

A 4B local model *will* sometimes answer from memory. Three layers catch that, in order:

1. **Required-tool enforcement.** The rules classifier decides whether a message needs the catalogue or the help centre. A place alone, as in "weather in Cairo?", doesn't count. If the model answered without that tool, its draft is discarded, the loop fetches the tool result itself, and the model answers again from real data.
2. **Corrective rewrite.** If the model had the data but quoted a price that isn't in it, it gets one rewrite, with the exact allowed prices listed.
3. **Safe fallback.** If a price still isn't in any tool result, it is never shown. The shopper sees "I couldn't confirm that price" instead.

Only real price fields count as evidence. An early version let the ID `KZ-0001` "ground" an invented price of 1 EGP; that's now a regression test.

## Agent eval

`uv run python -m copilot.eval.agent_eval --providers <scripted|local|claude>` runs 20 scenarios: offers in MSA, dialect, Arabizi and English; help questions; verified and unverified order lookups; escalation; prompt injection in both languages; and an off-topic question. Every check is programmatic, with no LLM judge: the right tool, facts present (digit- and spelling-normalised), the reply language, grounding, and the guardrail outcome.

| metric | scripted rules (CI) | qwen3 4B local, v1 | **qwen3 4B local, v2** |
|---|---|---|---|
| tool accuracy | 0.95 | 0.80 | **1.00** |
| facts present | 1.00 | 0.65 | **0.90** |
| grounded replies (no price outside tool data) | 1.00 | 0.85 | **0.95** |
| reply language matches the shopper | 1.00 | 0.90 | **0.95** |
| guardrail accuracy | 1.00 | 1.00 | 1.00 |
| latency p50 / p95 | 21 ms / 100 ms | 17.5 s / 45 s | 27.3 s / 45 s |

The local model runs on a 4 GB GTX 1650 Ti, split 34% CPU and 66% GPU. **v1 → v2** is the effect of the required-tool enforcement and corrective rewrite. The model stopped answering policy questions from memory, at the cost of extra calls. The remaining ungrounded reply was replaced by the safe fallback, so the shopper never saw the wrong price, but it still fails the 1.00 gate. Three of the 20 turns hit the 45 s timeout and were served by the rules engine instead of hanging. That is the router working as designed, and it is the case for routing hard turns to Claude in production.

## Guardrails

| risk | control | test |
|---|---|---|
| prompt injection (Arabic and English) | screened before any model call; tool outputs screened too | `test_prompt_injection_short_circuits_without_calling_a_model` |
| invented prices | the three grounding layers above | `test_ungrounded_price_is_never_shown`, `…corrective_rewrite`, `…forced_through_the_help_centre` |
| account takeover / data leaks | order lookup needs the ID plus the phone's last 4 digits; the same error for "no such order" and "wrong digits" | `test_order_lookup_never_leaks_with_wrong_digits` |
| PII in logs and traces | phone numbers, Egyptian national IDs, emails and card numbers redacted before spans are written | `test_pii_redaction` |
| runaway loops | at most 5 steps per turn, then escalate | — |

## Run it

```bash
(cd ../azure-local-lab && docker compose up -d)    # Grafana / OTel (optional)
ollama pull qwen3:4b-instruct && ollama pull qwen3-embedding:0.6b
uv sync
uv run uvicorn copilot.api:app --port 8080          # RTL chat UI at http://localhost:8080
```

| env | default | meaning |
|---|---|---|
| `COPILOT_PROVIDERS` | `local,claude` | router order; `scripted` is always appended as the last resort |
| `COPILOT_DENSE` | `1` | `0` = lexical + slots only (no embedding endpoint) |
| `CLAUDE_MODEL` / `CLAUDE_EFFORT` | `claude-opus-5-5` / `medium` | Claude provider (uses your `ANTHROPIC_API_KEY` or `ant auth login`) |
| `LOCAL_CHAT_MODEL` / `LOCAL_TIMEOUT_S` | `qwen3:4b-instruct` / `45` | OpenAI-compatible model and its fall-through timeout |
| `CLOUDKIT_MODE` | `local` | `azure` = Azure OpenAI and App Insights through cloudkit |

**MCP** (Claude Desktop / Claude Code): `{"mcpServers": {"kanz": {"command": "uv", "args": ["--directory", "<repo>", "run", "python", "-m", "copilot.mcp_server"]}}}`

## Status

- [x] Dialect/Arabizi-aware hybrid retrieval with query understanding, with a measured ablation and a holdout split
- [x] Tool-using agent with a cheap-first router, grounding enforcement and guardrails; 48 tests
- [x] MCP server, RTL chat UI, OTel spans, CI gates on retrieval and agent evals
- [ ] Claude live eval: the adapter is unit-tested against a fake client, but not yet run against the API (needs credentials, and costs money)
- [ ] Azure deployment (Container Apps + Azure OpenAI + AI Search) with Bicep / `azd`
- [ ] Larger, human-written eval set (the 12-query holdout is too small to rank close configurations)
- [ ] Session store in Redis or Postgres instead of process memory

## License

MIT
