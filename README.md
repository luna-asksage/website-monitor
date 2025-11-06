# 🤖 Website Monitor

Autonomous monitoring system that checks AI product/API websites daily and creates GitHub Issues when relevant updates are detected.

## Features

- ✅ Monitors OpenAI, Google Gemini, and Anthropic websites
- ✅ Uses LLM to intelligently filter for product/API changes only
- ✅ Creates GitHub Issues with detailed change summaries
- ✅ Fully autonomous via GitHub Actions
- ✅ Searchable history of all detected changes
- ✅ Uses AskSage api (only credential/token needed)
- ✅ Free forever on GitHub's infrastructure

## How It Works

1. GitHub Actions runs daily at 2 PM UTC
2. Fetches current content from each website
3. Compares with previous snapshot
4. If changed, LLM analyzes for product/API relevance
5. Creates GitHub Issue if relevant changes detected
6. Updates storage with new snapshot

## Monitored Sites

- 🔵 **OpenAI** - Product releases and updates
- 🟡 **Google Gemini** - API changelog
- 🟠 **Anthropic** - Product and API news

## Setup

See [SETUP.md](SETUP.md) for complete setup instructions.

## Customization

### Change Schedule

Edit `.github/workflows/monitor.yml`:

```yaml
schedule:
  - cron: '0 14 * * *'  # 2 PM UTC
