# 🤖 AI Release Monitor

Autonomous monitoring system that checks AI company websites and creates GitHub Issues when new model releases, API updates, or developer-relevant changes are detected.

## Features

- 🔍 **Smart Detection** - Multi-step LLM analysis to identify relevant releases
- 🚀 **Parallel Processing** - Fetches and analyzes content concurrently
- 📋 **GitHub Issues** - Automatically creates detailed issues for each release
- 🔄 **Auto-Cleanup** - Issues auto-close after 7 days
- 🎯 **No Duplicates** - Tracks handled releases to prevent repeat notifications
- ⏰ **Daily Monitoring** - Runs automatically via GitHub Actions
- 💰 **Free Forever** - Uses GitHub's free infrastructure

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  1. EXTRACT                                                 │
│     - Fetch RSS feeds and web pages                         │
│     - Filter to last 7 days OR 10 items (whichever first)   │
│     - Skip already-handled releases                         │
├─────────────────────────────────────────────────────────────┤
│  2. FETCH (Parallel)                                        │
│     - Retrieve full article content for each item           │
│     - Handle 403s gracefully (use RSS description instead)  │
├─────────────────────────────────────────────────────────────┤
│  3. ANALYZE (Parallel, Multi-Step LLM)                      │
│     - Query 1: "What is this article about?"                │
│     - Query 2: "Is this a product/API release?" (YES/NO)    │
│     - Query 3: "Generate GitHub issue body" (if relevant)   │
├─────────────────────────────────────────────────────────────┤
│  4. CREATE ISSUES                                           │
│     - Create GitHub Issue for each relevant release         │
│     - Add to handled releases database                      │
│     - Auto-close issues older than 7 days                   │
└─────────────────────────────────────────────────────────────┘
```

## Monitored Sites

| Source | Type | What's Detected |
|--------|------|-----------------|
| 🔵 **OpenAI** | RSS Feed | New models, API updates, product launches |
| 🟠 **Anthropic** | News Page | Claude releases, API changes |
| 🟡 **Google Gemini** | Changelog | API updates, new capabilities |

## What Gets Flagged

✅ **Relevant (Creates Issue):**
- New AI models (GPT-5, Claude 4, Gemini 2.5, etc.)
- Model version updates
- New API features/endpoints
- Pricing changes
- SDK releases
- Capability improvements
- Deprecations

❌ **Not Relevant (Ignored):**
- Company news (hiring, funding)
- Research papers without products
- Policy/safety announcements
- Blog posts about AI concepts

## Setup

### 1. Fork this repository

### 2. Add secrets

Go to **Settings → Secrets and variables → Actions** and add:

| Secret | Description |
|--------|-------------|
| `ASKSAGE_API_TOKEN` | Your AskSage API token |

> `GITHUB_TOKEN` is provided automatically by GitHub Actions

### 3. Enable Actions

Go to **Actions** tab and enable workflows if prompted.

### 4. (Optional) Trigger manually

Go to **Actions → AI Website Monitor → Run workflow** to test immediately.

## Configuration

### config.json

```json
{
  "asksage_api": {
    "url": "https://api.asksage.ai/server/query",
    "model": "google-claude-45-opus",
    "temperature": 0.2
  },
  "storage_dir": "storage"
}
```

### Change Schedule

Edit `.github/workflows/monitor.yml`:

```yaml
schedule:
  # Every 6 hours
  - cron: '0 */6 * * *'
  
  # Or daily at 8 AM UTC
  - cron: '0 8 * * *'
```

### Add More Sources

Edit `monitor.py` and add to the `self.websites` dictionary:

```python
"new_source": {
    "url": "https://example.com/news/rss.xml",
    "name": "Example AI",
    "type": "rss",  # or "html_news", "html_changelog"
    "emoji": "🟣",
    "label": "example",
    "base_url": "https://example.com"
}
```

## File Structure

```
├── monitor.py              # Main monitoring script
├── config.json             # Configuration
├── requirements.txt        # Python dependencies
├── storage/
│   └── handled_releases.json   # Tracks processed releases
├── .github/
│   └── workflows/
│       └── monitor.yml     # GitHub Actions workflow
└── monitor.log             # Local debug log (not committed)
```

## Labels

Issues are created with these labels:

| Label | Description |
|-------|-------------|
| `ai-release` | All detected releases |
| `openai` / `anthropic` / `gemini` | Source-specific |
| `auto-close-7d` | Will be auto-closed after 7 days |

## Troubleshooting

### No issues being created

1. Check **Actions** tab for workflow runs
2. Download `monitor.log` artifact for details
3. Verify `ASKSAGE_API_TOKEN` secret is set

### Too many/few detections

Adjust the LLM prompts in `_analyze_item()` method in `monitor.py`

### Rate limiting

Reduce parallel workers in `monitor.py`:
```python
self.max_fetch_workers = 3  # Default: 5
self.max_llm_workers = 2    # Default: 3
```

## Development

### Run locally

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export ASKSAGE_API_TOKEN="your-token"
export GITHUB_TOKEN="your-github-pat"
export GITHUB_REPOSITORY="owner/repo"

# Run
python monitor.py
```

### Clean restart

```bash
rm -rf storage/ monitor.log
```

## License

MIT
