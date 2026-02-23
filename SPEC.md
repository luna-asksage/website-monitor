# Spec: AI Model Availability Monitor v2

**Author:** Durin (drafted for Luna)  
**Date:** 2026-02-23  
**Repo:** github.com/luna-asksage/website-monitor  
**Status:** DRAFT — awaiting review

---

## 1. Problem Statement

The current website-monitor detects new AI models by scraping news feeds and using LLM classification. This produces:
- **False positives:** Existing models caught as "new," unrelated product news flagged
- **False negatives:** New models missed when announcement wording doesn't match LLM expectations
- **No platform availability tracking:** No visibility into whether models are on GCP, AWS GovCloud/SIPR, Azure, Azure Gov, or US-only regions

## 2. Requirements

### 2.1 Functional Requirements

| ID | Requirement |
|----|-------------|
| FR-1 | Detect new Claude models and track availability on: **direct API**, **GCP Vertex AI**, **AWS Bedrock**, **AWS GovCloud/SIPR** |
| FR-2 | Detect new OpenAI models and track availability on: **direct API**, **Azure OpenAI**, **Azure Government** |
| FR-3 | Detect new Gemini models and track availability in: **US regions only** (not global-only models) |
| FR-4 | All detected models must be **API-available** (not just announced, not consumer-only, not research previews) |
| FR-5 | Create GitHub Issue when a **new model** is detected on **any platform** |
| FR-6 | Update existing GitHub Issue when a known model becomes available on a **new platform** |
| FR-7 | Issue body must include an **availability matrix** showing all tracked platforms |
| FR-8 | Retain existing functionality: watchlist, mistake tracking, auto-close |

### 2.2 Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR-1 | Zero false positives for models already in `known_models.json` |
| NFR-2 | No API keys required for cloud platforms (public docs only) |
| NFR-3 | Runs on GitHub Actions free tier (existing infra) |
| NFR-4 | Graceful degradation if any single doc page is unreachable |
| NFR-5 | Detection latency ≤ 6 hours (matches current cron schedule) |

## 3. Architecture

### 3.1 Current Flow (news-based)

```
RSS/HTML feeds → Extract items → LLM: "Is this new?" → Create issue
                                  ↑ fuzzy, error-prone
```

### 3.2 Proposed Flow (docs-based + news enrichment)

```
┌─────────────────────────────────────────────────────────┐
│ PHASE 1: MODEL CATALOG SCRAPE (deterministic)          │
│                                                         │
│  For each provider:                                     │
│    1. Fetch public docs pages (model listings)          │
│    2. Extract model IDs via regex/HTML parsing          │
│    3. Tag each model with platform availability         │
│    4. Diff against known_models.json                    │
│    5. New model? → flag for issue creation              │
│    6. Known model + new platform? → flag for update     │
├─────────────────────────────────────────────────────────┤
│ PHASE 2: NEWS ENRICHMENT (optional, LLM-assisted)      │
│                                                         │
│  For each newly detected model:                         │
│    1. Search existing news feed items for context       │
│    2. LLM: Generate issue body with capabilities,       │
│       pricing, context window, release notes link       │
│    3. If no news match → create issue with model ID     │
│       and availability matrix only                      │
├─────────────────────────────────────────────────────────┤
│ PHASE 3: ISSUE MANAGEMENT (existing + enhanced)        │
│                                                         │
│  - Create issue for new models                          │
│  - Comment on existing issue for new platform avail.    │
│  - Watchlist resolution (existing)                      │
│  - Auto-close (existing)                                │
│  - Mistake tracking (existing)                          │
└─────────────────────────────────────────────────────────┘
```

### 3.3 Key Design Decision: Deterministic Detection

The **core detection** (Phase 1) uses **no LLM**. It's pure scrape + regex + diff. This eliminates the entire class of LLM classification errors for the "is this a new model?" question.

The LLM is only used for **enrichment** — generating human-readable issue descriptions from the raw data. If the LLM fails, the issue is still created with the raw model ID and availability data.

## 4. Data Sources

### 4.1 Claude (Anthropic)

| Platform | URL | Parse Strategy |
|----------|-----|----------------|
| Direct API | `https://docs.anthropic.com/en/docs/about-claude/models` | Regex: `claude-[\w\d.-]+` from model ID tables/code blocks |
| GCP Vertex AI | `https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude` | Extract model IDs from "Available models" or similar sections |
| AWS Bedrock | `https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html` | Extract Anthropic rows from the supported models table |
| AWS GovCloud | `https://docs.aws.amazon.com/govcloud-us/latest/UserGuide/govcloud-bedrock.html` | Extract Anthropic rows from GovCloud models table |

**Notes:**
- AWS SIPR availability may not be on public docs. Flag in spec: if SIPR info is classified/non-public, we can only track GovCloud (which is public). Luna to confirm if there's a public source for SIPR model listings.
- Vertex AI page structure may list models in prose, code examples, or tables — parser needs to handle all three.

### 4.2 OpenAI

| Platform | URL | Parse Strategy |
|----------|-----|----------------|
| Direct API | `https://platform.openai.com/docs/models` | Regex: `gpt-[\w\d.-]+`, `o[1-9][\w-]*`, `codex-[\w\d-]+` |
| Azure OpenAI | `https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models` | Extract model names from availability tables |
| Azure Government | `https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models` (Gov section) OR `https://learn.microsoft.com/en-us/azure/azure-government/documentation-government-cognitive-services` | Extract from Gov-specific tables |

**Notes:**
- Azure docs page has region-specific model tables. Gov models are sometimes on the same page under a "Government regions" section, sometimes on a separate page. Parser should check both.
- OpenAI direct models page is heavily JS-rendered. May need to fall back to the API reference page or `platform.openai.com/docs/api-reference` if scraping fails.

### 4.3 Gemini (Google)

| Platform | URL | Parse Strategy |
|----------|-----|----------------|
| Direct API | `https://ai.google.dev/gemini-api/docs/models` | Regex: `gemini-[\w\d.-]+` |
| US Region Availability | `https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations` | Cross-reference model list with US region columns |

**Notes:**
- "US regions only" means the model must be available in at least one US region (`us-central1`, `us-east1`, `us-east4`, `us-west1`, etc.). Models that are only listed under non-US regions are excluded.
- The locations page is a large table; parse carefully for Gemini rows × US columns.

## 5. Data Model

### 5.1 Enhanced `known_models.json`

```json
{
  "models": {
    "anthropic": {
      "claude-opus-4-6": {
        "first_seen": "2026-01-15T00:00:00Z",
        "platforms": {
          "direct_api": { "available": true, "first_seen": "2026-01-15T00:00:00Z" },
          "gcp_vertex": { "available": true, "first_seen": "2026-01-20T00:00:00Z" },
          "aws_bedrock": { "available": true, "first_seen": "2026-01-18T00:00:00Z" },
          "aws_govcloud": { "available": false, "first_seen": null }
        },
        "issue_number": 42
      }
    },
    "openai": {
      "gpt-5.2": {
        "first_seen": "2026-02-01T00:00:00Z",
        "platforms": {
          "direct_api": { "available": true, "first_seen": "2026-02-01T00:00:00Z" },
          "azure": { "available": true, "first_seen": "2026-02-10T00:00:00Z" },
          "azure_gov": { "available": false, "first_seen": null }
        },
        "issue_number": 55
      }
    },
    "google": {
      "gemini-2.5-pro": {
        "first_seen": "2026-01-28T00:00:00Z",
        "platforms": {
          "direct_api": { "available": true, "first_seen": "2026-01-28T00:00:00Z" },
          "vertex_us": { "available": true, "first_seen": "2026-01-28T00:00:00Z", "regions": ["us-central1", "us-east4"] }
        },
        "issue_number": 48
      }
    }
  },
  "updated": "2026-02-23T23:00:00Z",
  "schema_version": 2
}
```

### 5.2 Platform Registry

```python
PLATFORMS = {
    "anthropic": {
        "direct_api": {
            "name": "Anthropic API",
            "url": "https://docs.anthropic.com/en/docs/about-claude/models",
            "parser": "anthropic_docs",
        },
        "gcp_vertex": {
            "name": "GCP Vertex AI",
            "url": "https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude",
            "parser": "gcp_vertex_claude",
        },
        "aws_bedrock": {
            "name": "AWS Bedrock",
            "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html",
            "parser": "aws_bedrock_anthropic",
        },
        "aws_govcloud": {
            "name": "AWS GovCloud",
            "url": "https://docs.aws.amazon.com/govcloud-us/latest/UserGuide/govcloud-bedrock.html",
            "parser": "aws_govcloud_anthropic",
        },
    },
    "openai": {
        "direct_api": {
            "name": "OpenAI API",
            "url": "https://platform.openai.com/docs/models",
            "parser": "openai_docs",
        },
        "azure": {
            "name": "Azure OpenAI",
            "url": "https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models",
            "parser": "azure_openai",
        },
        "azure_gov": {
            "name": "Azure Government",
            "url": "https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models",
            "parser": "azure_gov_openai",
        },
    },
    "google": {
        "direct_api": {
            "name": "Google AI Studio",
            "url": "https://ai.google.dev/gemini-api/docs/models",
            "parser": "gemini_docs",
        },
        "vertex_us": {
            "name": "Vertex AI (US)",
            "url": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations",
            "parser": "vertex_us_gemini",
        },
    },
}
```

## 6. Parser Design

### 6.1 Parser Interface

Each parser implements:

```python
class PlatformParser:
    def parse(self, html: str) -> set[str]:
        """
        Extract model IDs from HTML content.
        
        Returns:
            Set of normalized model ID strings.
            Empty set on parse failure (never raises).
        """
```

### 6.2 Model ID Normalization

Model IDs must be normalized before comparison:
- Lowercase
- Strip date suffixes for comparison but preserve in storage (e.g., `claude-3.5-sonnet-20241022` normalizes to `claude-3.5-sonnet` for dedup, but stores the full ID)
- Map known aliases (e.g., `claude-3-5-sonnet` → `claude-3.5-sonnet`)

**Open question for Luna:** How granular do you want tracking? Options:
1. **Model family only** (`claude-sonnet-4`) — fewest issues, might miss point releases
2. **Model version** (`claude-sonnet-4-5`) — good balance
3. **Full model ID with date** (`claude-sonnet-4-5-20260115`) — most granular, noisiest

### 6.3 Parser Resilience

Each parser must:
- Return empty set on failure (not crash)
- Log warnings when page structure doesn't match expected patterns
- Track "confidence" — if regex matches 0 models on a page that previously had 10+, that's likely a page structure change, not all models being removed. Log error, return cached data.

```python
def _validate_parse_result(self, provider: str, platform: str, 
                            found: set[str], cached: set[str]) -> set[str]:
    """Guard against page structure changes wiping the model list."""
    if cached and len(found) == 0:
        logger.error(f"{provider}/{platform}: Found 0 models but cache has {len(cached)}. "
                     f"Possible page structure change. Using cached data.")
        return cached
    if cached and len(found) < len(cached) * 0.5:
        logger.warning(f"{provider}/{platform}: Found {len(found)} models but cache has "
                       f"{len(cached)}. Significant drop — review parser.")
    return found
```

## 7. Issue Format

### 7.1 New Model Issue

```markdown
## 🟠 New Claude Model: claude-opus-4-7

**Provider:** Anthropic  
**First Detected:** 2026-02-23  
**Type:** API Model

### Availability Matrix

| Platform | Available | Since |
|----------|-----------|-------|
| Anthropic API | ✅ | 2026-02-23 |
| GCP Vertex AI | ❌ | — |
| AWS Bedrock | ❌ | — |
| AWS GovCloud | ❌ | — |

### Details
<!-- LLM-enriched content here, or "No release notes found yet." -->

### Source Links
- [Anthropic Models Docs](https://docs.anthropic.com/en/docs/about-claude/models)

---
*Detected by AI Model Availability Monitor v2*
```

### 7.2 Platform Availability Update (Comment on Existing Issue)

```markdown
## 🆕 Platform Update

**claude-opus-4-7** is now available on **AWS Bedrock**!

### Updated Availability Matrix

| Platform | Available | Since |
|----------|-----------|-------|
| Anthropic API | ✅ | 2026-02-23 |
| GCP Vertex AI | ✅ | 2026-02-25 |
| AWS Bedrock | ✅ | 2026-03-01 |
| AWS GovCloud | ❌ | — |
```

## 8. Migration Path

### 8.1 Backward Compatibility

- Existing `known_models.json` (schema v1: flat lists) must be migrated to schema v2 (platform-aware objects)
- Migration runs automatically on first v2 execution
- Existing models get `direct_api: available` set, all other platforms set to `unknown` (not `false`) until first full scrape

### 8.2 Existing News Monitor

- **Keep it.** The RSS/HTML news scraping continues to run as Phase 2 enrichment
- News items that don't correspond to a Phase 1 model detection are logged but **do not create issues** (eliminates false positives)
- Non-model announcements (pricing, deprecations, API changes) are **out of scope** for v2

## 9. Configuration Changes

### 9.1 Updated `config.json`

```json
{
  "asksage_api": {
    "url": "https://api.asksage.ai/server/query",
    "model": "google-claude-45-opus",
    "temperature": 0.1
  },
  "storage_dir": "storage",
  "detection": {
    "model_granularity": "version",
    "news_enrichment": true,
    "standalone_news_issues": false,
    "platforms": {
      "anthropic": ["direct_api", "gcp_vertex", "aws_bedrock", "aws_govcloud"],
      "openai": ["direct_api", "azure", "azure_gov"],
      "google": ["direct_api", "vertex_us"]
    }
  },
  "gemini_us_regions": [
    "us-central1", "us-east1", "us-east4", "us-south1", "us-west1", "us-west4"
  ]
}
```

## 10. Edge Cases

| Edge Case | Handling |
|-----------|----------|
| Doc page returns 403/5xx | Use cached models, log warning, don't create false "model removed" events |
| Doc page structure changes | Confidence check (§6.3) catches this, uses cache, creates `type:bug` issue |
| Model renamed (e.g., version suffix change) | Normalization layer handles; configurable granularity |
| Model deprecated/removed from docs | **Do not auto-close issue.** Log observation. Deprecation tracking is a future feature |
| Same model on direct API + cloud with different IDs | Alias mapping in config (e.g., Bedrock uses `anthropic.claude-3-sonnet-20240229-v1:0`) |
| JS-rendered pages (OpenAI) | Primary: use requests + BeautifulSoup. Fallback: hardcode known model patterns as safety net |
| Rate limiting on doc pages | Respectful fetch intervals (1-2s between requests), exponential backoff on 429 |
| GovCloud info not publicly available | Track GovCloud only (public). No SIPR tracking. |

### 10.1 LLM Training Data Staleness (Critical)

**Problem:** The LLM used for enrichment may have stale training data. It might
"think" a model is new because it wasn't in training data, even though the model
has existed for months. Conversely, it might not recognize a genuinely new model
because it has no training data about it.

**Example:** Gemini 3.1 releases in Feb 2026. The LLM's training data might not
include Gemini 3.0 (released earlier), causing it to flag Gemini 3.0 as "new" too.

**Mitigation — the LLM NEVER decides what is "new":**

1. **"New" is defined exclusively by diffing against `known_models.json`.**
   A model is new if and only if it appears on a doc page AND is not in the
   known models store. The LLM has zero say in this determination.

2. **First-run seeding:** On first v2 execution, ALL models found across all
   doc pages are written to `known_models.json` as "known." No issues are
   created on the first run. This establishes the baseline from reality, not
   from any LLM's understanding.

3. **LLM enrichment prompts must NOT ask "is this new?"** The enrichment
   prompt receives a model ID that has already been determined to be new by
   the diff engine. The LLM is only asked: "Given this model ID and any
   available release notes, generate a description of its capabilities."

4. **No LLM-based model list generation.** The known models list is built
   entirely from doc page scraping, never from asking an LLM to list models
   it knows about. The fallback model lists in the current code (hardcoded
   lists of model names) should be removed in v2 — they're a staleness vector.

## 11. Decisions (Resolved)

1. **SIPR:** No public source. **Skip AWS GovCloud entirely** (not needed).
2. **Model granularity:** **Version-level** (e.g., `claude-sonnet-4-5`, not date-stamped).
3. **Bedrock model IDs:** **Normalize** to provider's canonical naming.
4. **Notifications:** GitHub Issues only (emails Luna automatically).
5. **Non-model announcements:** Models-only for now. Pricing/deprecation/API changes are out of scope.
6. **File structure:** **Refactor `monitor.py` in-place** — better long-term.
7. **OpenAI direct API:** **Skip** — JS-rendered page returns 403. Azure covers all OpenAI models.
8. **Azure Gov source:** Raw markdown from `MicrosoftDocs/azure-ai-docs` GitHub repo.

## 12. Implementation Plan

| Phase | Work | Estimate |
|-------|------|----------|
| 1 | Schema v2 migration + platform registry | Small |
| 2 | Parsers for all 9 doc pages | Medium (most of the work) |
| 3 | Diff engine + issue creation/update | Small |
| 4 | News enrichment integration | Small |
| 5 | Config updates + testing | Small |
| 6 | GitHub Actions workflow updates | Trivial |

**Total estimate:** A solid afternoon of focused work for the coding agent.

## 13. Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Doc pages change structure frequently | Medium | Confidence checks, cached fallback, `type:bug` auto-issue |
| JS-rendered pages block scraping | Medium | Fallback patterns, consider using a headless browser in GH Actions if needed |
| Bedrock/Azure ID format divergence | High | Alias mapping config, normalize on ingest |
| GovCloud docs are sparse/lagging | Medium | Accept gaps, manual override field |

---

*End of spec. Awaiting Luna's review and answers to open questions (§11).*
