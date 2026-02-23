#!/usr/bin/env python3
"""
AI Model Availability Monitor v2
Deterministic model detection via doc-page scraping + diff.
LLM used only for enrichment, never for classification.
Creates GitHub Issues when new models or platform availability changes are detected.
"""

import argparse
import os
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
import requests
from bs4 import BeautifulSoup


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

SCHEMA_VERSION = 2

US_REGIONS = [
    "us-central1", "us-east1", "us-east4", "us-east5",
    "us-south1", "us-west1", "us-west4",
]

# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PlatformStatus:
    """Availability status for a model on a specific platform."""
    available: bool
    first_seen: Optional[str] = None
    regions: Optional[List[str]] = None

    def to_dict(self) -> dict:
        d = {"available": self.available, "first_seen": self.first_seen}
        if self.regions is not None:
            d["regions"] = self.regions
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'PlatformStatus':
        return cls(
            available=data.get("available", False),
            first_seen=data.get("first_seen"),
            regions=data.get("regions"),
        )


@dataclass
class KnownModel:
    """A model tracked across platforms."""
    first_seen: str
    platforms: Dict[str, PlatformStatus] = field(default_factory=dict)
    issue_number: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "first_seen": self.first_seen,
            "platforms": {k: v.to_dict() for k, v in self.platforms.items()},
            "issue_number": self.issue_number,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'KnownModel':
        platforms = {}
        for k, v in data.get("platforms", {}).items():
            platforms[k] = PlatformStatus.from_dict(v)
        return cls(
            first_seen=data["first_seen"],
            platforms=platforms,
            issue_number=data.get("issue_number"),
        )


@dataclass
class ModelDelta:
    """A detected change: new model or new platform availability."""
    provider: str
    model_id: str
    is_new_model: bool
    new_platforms: List[str]
    all_platforms: Dict[str, PlatformStatus]


@dataclass
class WatchlistItem:
    """An item being watched for future availability."""
    issue_number: int
    title: str
    watch_for: str
    why: str
    sources: List[str]
    created_at: str
    expires_at: str

    def to_dict(self) -> dict:
        return asdict(self)


# =============================================================================
# Required Labels
# =============================================================================

REQUIRED_LABELS = [
    {"name": "type:release", "color": "0E8A16", "description": "AI model release/availability detected"},
    {"name": "type:platform-update", "color": "0075CA", "description": "Existing model available on new platform"},
    {"name": "type:watchlist", "color": "1D76DB", "description": "Watching for future availability"},
    {"name": "type:bug", "color": "D73A4A", "description": "System error"},
    {"name": "openai", "color": "412991", "description": "OpenAI source"},
    {"name": "anthropic", "color": "D97706", "description": "Anthropic source"},
    {"name": "gemini", "color": "FBBC04", "description": "Google Gemini source"},
    {"name": "not-relevant", "color": "E99695", "description": "False positive"},
    {"name": "auto-close-7d", "color": "BFD4F2", "description": "Auto-closes after 7 days"},
]


# =============================================================================
# Platform Registry
# =============================================================================

PLATFORMS = {
    "anthropic": {
        "direct_api": {
            "name": "Anthropic API",
            "url": "https://docs.anthropic.com/en/docs/about-claude/models",
            "parser": "parse_anthropic_docs",
        },
        "gcp_vertex": {
            "name": "GCP Vertex AI",
            "url": "https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude",
            "parser": "parse_gcp_vertex_claude",
        },
        "aws_bedrock": {
            "name": "AWS Bedrock",
            "url": "https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html",
            "parser": "parse_aws_bedrock_anthropic",
        },
        "aws_govcloud": {
            "name": "AWS GovCloud",
            "url": "https://docs.aws.amazon.com/govcloud-us/latest/UserGuide/govcloud-bedrock.html",
            "parser": "parse_aws_govcloud_anthropic",
        },
    },
    "openai": {
        "direct_api": {
            "name": "OpenAI API",
            "url": "https://platform.openai.com/docs/models",
            "parser": "parse_openai_docs",
        },
        "azure": {
            "name": "Azure OpenAI",
            "url": "https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models",
            "parser": "parse_azure_openai",
        },
        "azure_gov": {
            "name": "Azure Government",
            "url": "https://raw.githubusercontent.com/MicrosoftDocs/azure-ai-docs/main/articles/ai-foundry/openai/azure-government.md",
            "parser": "parse_azure_gov_openai",
        },
    },
    "google": {
        "direct_api": {
            "name": "Google AI Studio",
            "url": "https://ai.google.dev/gemini-api/docs/models",
            "parser": "parse_gemini_docs",
        },
        "vertex_us": {
            "name": "Vertex AI (US)",
            "url": "https://cloud.google.com/vertex-ai/generative-ai/docs/learn/locations",
            "parser": "parse_vertex_us_gemini",
        },
    },
}


# =============================================================================
# Model ID Normalization
# =============================================================================

# Date suffix pattern: strip YYYYMMDD, YYYY-MM-DD, or MM-YYYY suffixes
DATE_SUFFIX_RE = re.compile(
    r'[-_](?:'
    r'2[0-9]{3}[01][0-9][0-3][0-9]'      # YYYYMMDD
    r'|2[0-9]{3}-[01][0-9]-[0-3][0-9]'    # YYYY-MM-DD
    r'|[01][0-9]-2[0-9]{3}'               # MM-YYYY
    r')(?:[-_]v\d+(?::\d+)?)?$'
)

# Bedrock model ID patterns
BEDROCK_PREFIX_RE = re.compile(r'^(?:anthropic|amazon|meta|cohere|stability|ai21|mistral)\.')
BEDROCK_SUFFIX_RE = re.compile(r'[-_]v\d+(?::\d+)?$')

# Version dash normalization: "3-5" -> "3.5" when followed by a dash (part of compound ID)
# Matches: claude-3-5-sonnet, gemini-2-5-flash, but NOT claude-3-haiku
VERSION_DASH_RE = re.compile(r'(\d+)-(\d+)(?=-)')

# Junk patterns: strings that look like model IDs but aren't
JUNK_PATTERNS_RAW = [
    # Anthropic non-model strings (page content, links, image filenames)
    r'claude-(?:docs|in-|on-|is-|prompting|code-analytics|best-practice|models|for-|with-|and-)',
    r'claude-.*\.(?:png|jpg|svg|gif|webp|pdf)$',
    # Gemini CSS classes, UI components, page artifacts (NOT real model IDs)
    r'gemini-api(?:-|$)',                    # gemini-api, gemini-api-card, etc.
    r'gemini-card',                          # gemini-card-centered, etc.
    r'gemini-centered',                      # CSS layout classes
    r'gemini-icon',                          # icon components
    r'gemini-model-(?:desc|details|grid|name|row|font|table|button)',  # CSS classes
    r'gemini-.*\.(?:png|jpg|svg|gif|webp|pdf|svg)$',  # image files
    r'gemini-.*(?:deprecated|_\d+$)',        # deprecated markers, underscore suffixes
    r'gemini-flash-latest$',                 # generic alias, not a real model
    # OpenAI trailing punctuation
    r'gpt-\d+[a-z]*\.$',
    # Too generic / too short
    r'^(?:claude|gemini|gpt)-?$',
]
JUNK_PATTERNS = [re.compile(p) for p in JUNK_PATTERNS_RAW]


def _is_junk_model_id(model_id: str) -> bool:
    """Return True if the model ID is a false positive from regex scraping."""
    for pattern in JUNK_PATTERNS:
        if pattern.search(model_id):
            return True
    return False


def normalize_model_id(raw_id: str, provider: str = "") -> str:
    """
    Normalize a model ID to version level.
    Strips date suffixes, provider prefixes, normalizes version dashes to dots.
    """
    model_id = raw_id.strip().lower()

    # Strip Bedrock-style provider prefix
    model_id = BEDROCK_PREFIX_RE.sub('', model_id)

    # Strip date suffixes
    model_id = DATE_SUFFIX_RE.sub('', model_id)

    # Strip Bedrock version suffixes like -v1:0
    model_id = BEDROCK_SUFFIX_RE.sub('', model_id)

    # Normalize version dashes to dots: claude-3-5-sonnet -> claude-3.5-sonnet
    model_id = VERSION_DASH_RE.sub(r'\1.\2', model_id)

    # Strip trailing dots/dashes
    model_id = model_id.rstrip('.-')

    return model_id


def _normalize_and_filter(raw_ids: Set[str], provider: str,
                           min_len: int = 7) -> Set[str]:
    """Normalize model IDs, filter junk, deduplicate."""
    normalized = set()
    for raw_id in raw_ids:
        if _is_junk_model_id(raw_id):
            continue
        nid = normalize_model_id(raw_id, provider)
        if _is_junk_model_id(nid):
            continue
        if len(nid) >= min_len and not nid.endswith('-'):
            normalized.add(nid)
    return normalized


# =============================================================================
# LLM Client (Enrichment Only)
# =============================================================================

class LLMClient:
    """Handles LLM API interactions for enrichment (never classification)."""

    def __init__(self, config: dict):
        self.api_url = config.get('url', '')
        self.api_token = config.get('token', '')
        self.model = config.get('model', 'google-gemini-2.5-pro')
        self.temperature = config.get('temperature', 0.1)

    def query(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        """Send a prompt to the LLM. Returns (response_text, error_message)."""
        if not self.api_token:
            return None, "No API token configured"

        try:
            files = {
                'model': (None, self.model),
                'temperature': (None, str(self.temperature)),
                'message': (None, json.dumps([{"user": "me", "message": prompt}])),
                'system_prompt': (None, "You are a helpful assistant. Follow the instructions precisely."),
                'dataset': (None, 'none'),
                'usage': (None, 'True')
            }

            response = requests.post(
                self.api_url,
                headers={'x-access-tokens': self.api_token},
                files=files,
                timeout=90
            )
            response.raise_for_status()
            result = response.json()

            response_text = None
            if isinstance(result, dict):
                response_text = result.get('message')
                if not response_text or not response_text.strip():
                    response_text = result.get('text')
                if not response_text or not response_text.strip():
                    resp_field = result.get('response', '')
                    if resp_field and resp_field.upper() not in ['OK', 'SUCCESS', 'ERROR']:
                        response_text = resp_field
            elif isinstance(result, str):
                response_text = result
            else:
                response_text = str(result)

            response_text = response_text.strip() if response_text else ""
            if not response_text:
                return None, "Empty response from API"

            return response_text, None

        except requests.exceptions.Timeout:
            return None, "Request timed out"
        except requests.exceptions.RequestException as e:
            return None, f"Request failed: {e}"
        except Exception as e:
            return None, f"Unexpected error: {e}"


# =============================================================================
# Platform Parsers
# =============================================================================

class PlatformParsers:
    """Extract model IDs from provider documentation pages."""

    def __init__(self, session: requests.Session):
        self.session = session

    def _fetch(self, url: str, timeout: int = 30) -> Optional[str]:
        """Fetch a URL and return its HTML content."""
        try:
            resp = self.session.get(url, timeout=timeout)
            if resp.ok:
                return resp.text
            logger.warning(f"Fetch failed for {url}: HTTP {resp.status_code}")
            return None
        except requests.exceptions.Timeout:
            logger.warning(f"Fetch timed out for {url}")
            return None
        except Exception as e:
            logger.warning(f"Fetch error for {url}: {e}")
            return None

    # ---- Anthropic ----

    def _extract_claude_ids(self, html: str) -> Set[str]:
        """Extract and normalize Claude model IDs from HTML."""
        text = html.lower()
        bedrock_ids = set(re.findall(r'anthropic\.claude-[\w\d.-]+(?::[\d]+)?', text))
        plain_ids = set(re.findall(r'claude-[\w\d][\w\d.-]*', text))
        return _normalize_and_filter(bedrock_ids | plain_ids, "anthropic")

    def parse_anthropic_docs(self, url: str) -> Set[str]:
        html = self._fetch(url)
        return self._extract_claude_ids(html) if html else set()

    def parse_gcp_vertex_claude(self, url: str) -> Set[str]:
        html = self._fetch(url)
        return self._extract_claude_ids(html) if html else set()

    def parse_aws_bedrock_anthropic(self, url: str) -> Set[str]:
        html = self._fetch(url)
        return self._extract_claude_ids(html) if html else set()

    def parse_aws_govcloud_anthropic(self, url: str) -> Set[str]:
        html = self._fetch(url)
        return self._extract_claude_ids(html) if html else set()

    # ---- OpenAI ----

    def _extract_openai_ids(self, text: str) -> Set[str]:
        """Extract and normalize OpenAI model IDs from text."""
        text_lower = text.lower()
        patterns = [
            r'gpt-[\w\d][\w\d.-]*',
            r'o[1-9]-[\w-]+',
            r'(?<![/\w])o[1-9](?![\w])',
        ]
        raw_ids = set()
        for pattern in patterns:
            raw_ids.update(re.findall(pattern, text_lower))
        return _normalize_and_filter(raw_ids, "openai", min_len=2)

    def parse_openai_docs(self, url: str) -> Set[str]:
        html = self._fetch(url)
        return self._extract_openai_ids(html) if html else set()

    def parse_azure_openai(self, url: str) -> Set[str]:
        """Extract OpenAI model IDs available on Azure OpenAI Service."""
        html = self._fetch(url)
        if not html:
            return set()

        soup = BeautifulSoup(html, 'html.parser')

        # Prefer models found in tables (more precise than full-page scan)
        tables = soup.find_all('table')
        table_models = set()
        for table in tables:
            table_text = table.get_text()
            table_models.update(self._extract_openai_ids(table_text))

        if table_models:
            return table_models

        # Fallback: full page
        return self._extract_openai_ids(soup.get_text())

    def parse_azure_gov_openai(self, url: str) -> Set[str]:
        """Extract OpenAI model IDs from Azure Government docs (raw markdown)."""
        content = self._fetch(url)
        if not content:
            return set()

        gov_models = set()

        # Parse markdown tables: look for rows with ✅ indicating availability
        # Table format: | Region | model1 | model2 | ... |
        # Row format:   | usgovarizona | ✅ | - | ✅ | ... |
        lines = content.split('\n')
        header_line = None
        header_models = []

        for i, line in enumerate(lines):
            line = line.strip()
            if not line.startswith('|'):
                header_line = None
                header_models = []
                continue

            cells = [c.strip() for c in line.split('|')]
            # Remove empty first/last cells from leading/trailing pipes
            cells = [c for c in cells if c or c == '']

            # Skip separator rows
            if all(set(c) <= {'-', ':', ' '} for c in cells if c):
                continue

            # Detect header row: contains model names like gpt-4.1, o3-mini, etc.
            if any(re.search(r'gpt-|o[1-9]', c.lower()) for c in cells):
                # Check if first cell looks like a region header (Region, empty, etc.)
                first = cells[0].lower() if cells else ''
                if 'region' in first or first.startswith('**') or not re.search(r'gpt-|o[1-9]', first):
                    header_models = []
                    for cell in cells[1:]:
                        # Extract model name: "**gpt-4.1**, **2025-04-14**" -> "gpt-4.1"
                        model_matches = re.findall(r'(gpt-[\w\d][\w\d.-]*|o[1-9][\w-]*)', cell.lower())
                        header_models.append(model_matches[0] if model_matches else None)
                    header_line = i
                    continue

            # Data row: check for ✅ availability markers
            if header_line is not None and header_models:
                for j, cell in enumerate(cells[1:]):
                    if '✅' in cell and j < len(header_models) and header_models[j]:
                        gov_models.add(header_models[j])

        return _normalize_and_filter(gov_models, "openai", min_len=2)

    # ---- Google/Gemini ----

    def _extract_gemini_ids(self, text: str) -> Set[str]:
        """Extract and normalize Gemini model IDs from text."""
        raw_ids = set(re.findall(r'gemini-[\w\d][\w\d.-]*', text.lower()))
        return _normalize_and_filter(raw_ids, "google")

    def parse_gemini_docs(self, url: str) -> Set[str]:
        html = self._fetch(url)
        return self._extract_gemini_ids(html) if html else set()

    def parse_vertex_us_gemini(self, url: str) -> Set[str]:
        """Extract Gemini models available in US regions on Vertex AI."""
        html = self._fetch(url)
        if not html:
            return set()

        soup = BeautifulSoup(html, 'html.parser')
        us_models = set()

        # Look for tables mapping models to regions
        for table in soup.find_all('table'):
            headers = [th.get_text().strip().lower() for th in table.find_all('th')]

            us_col_indices = []
            for i, h in enumerate(headers):
                if any(region in h for region in US_REGIONS):
                    us_col_indices.append(i)

            if not us_col_indices:
                continue

            for row in table.find_all('tr'):
                cells = row.find_all(['td', 'th'])
                if not cells:
                    continue

                model_text = cells[0].get_text().lower()
                gemini_ids = re.findall(r'gemini-[\w\d][\w\d.-]*', model_text)
                if not gemini_ids:
                    continue

                has_us = False
                for col_idx in us_col_indices:
                    if col_idx < len(cells):
                        cell_text = cells[col_idx].get_text().strip().lower()
                        if cell_text and cell_text not in ['-', 'no', 'n/a', '', '—']:
                            has_us = True
                            break

                if has_us:
                    for raw_id in gemini_ids:
                        nid = normalize_model_id(raw_id, "google")
                        if not _is_junk_model_id(raw_id) and not _is_junk_model_id(nid):
                            if len(nid) > 7:
                                us_models.add(nid)

        # Fallback: prose extraction if table parsing failed
        if not us_models:
            text = soup.get_text().lower()
            if any(r in text for r in US_REGIONS):
                us_models = self._extract_gemini_ids(text)
                if us_models:
                    logger.warning(
                        f"Vertex US parser fell back to prose extraction — "
                        f"found {len(us_models)} models. Review page structure."
                    )

        return us_models


# =============================================================================
# Website Monitor
# =============================================================================

class WebsiteMonitor:
    """Monitors AI provider documentation for new model availability."""

    def __init__(self, config_path: str = "config.json",
                 dry_run: bool = False, seed: bool = False):
        self.config = self._load_config(config_path)
        self._override_with_env_vars()

        self.storage_dir = Path(self.config.get("storage_dir", "storage"))
        self.storage_dir.mkdir(exist_ok=True)

        self.llm = LLMClient(self.config.get('asksage_api', {}))

        self.github_token = os.getenv('GITHUB_TOKEN')
        self.github_repo = os.getenv('GITHUB_REPOSITORY')

        self.dry_run = dry_run
        self.seed = seed
        self.errors: List[str] = []

        self.fetch_delay_seconds = 1.5

        detection = self.config.get("detection", {})
        self.news_enrichment = detection.get("news_enrichment", True)
        self.active_platforms = detection.get("platforms", {
            "anthropic": ["direct_api", "gcp_vertex", "aws_bedrock"],
            "openai": ["azure", "azure_gov"],
            "google": ["direct_api", "vertex_us"],
        })

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': (
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            ),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        })

        self.parsers = PlatformParsers(self.session)

    # =========================================================================
    # Configuration
    # =========================================================================

    def _load_config(self, config_path: str) -> Dict:
        default_config = {
            "asksage_api": {
                "url": "https://api.asksage.ai/server/query",
                "token": "",
                "model": "google-claude-45-opus",
                "temperature": 0.1
            },
            "storage_dir": "storage",
            "detection": {
                "news_enrichment": True,
                "platforms": {
                    "anthropic": ["direct_api", "gcp_vertex", "aws_bedrock"],
                    "openai": ["azure", "azure_gov"],
                    "google": ["direct_api", "vertex_us"],
                }
            },
        }

        try:
            with open(config_path, 'r') as f:
                loaded = json.load(f)
                for key in default_config:
                    if key in loaded:
                        if isinstance(default_config[key], dict):
                            default_config[key].update(loaded[key])
                        else:
                            default_config[key] = loaded[key]
                return default_config
        except FileNotFoundError:
            logger.info(f"Config file {config_path} not found, using defaults")
            return default_config
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {config_path}: {e}")
            return default_config
        except Exception as e:
            logger.warning(f"Config load error: {e}, using defaults")
            return default_config

    def _override_with_env_vars(self):
        if os.getenv('ASKSAGE_API_TOKEN'):
            self.config['asksage_api']['token'] = os.getenv('ASKSAGE_API_TOKEN')
            logger.info("Using API token from environment")

    # =========================================================================
    # GitHub API
    # =========================================================================

    def _github_api(self, method: str, endpoint: str, **kwargs) -> Optional[requests.Response]:
        if self.dry_run:
            logger.info(f"[DRY RUN] Would {method} {endpoint}")
            return None

        if not self.github_token or not self.github_repo:
            logger.debug(f"GitHub API call skipped ({method} {endpoint}): not configured")
            return None

        url = f"https://api.github.com/repos/{self.github_repo}{endpoint}"
        headers = {
            'Authorization': f'token {self.github_token}',
            'Accept': 'application/vnd.github.v3+json'
        }

        try:
            resp = requests.request(method, url, headers=headers, timeout=30, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.exceptions.Timeout:
            logger.error(f"GitHub API timeout ({method} {endpoint})")
            return None
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else 'unknown'
            logger.error(f"GitHub API HTTP {status_code} ({method} {endpoint}): {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"GitHub API request failed ({method} {endpoint}): {e}")
            return None

    def _ensure_labels_exist(self):
        if self.dry_run or not self.github_token or not self.github_repo:
            return

        resp = self._github_api('GET', '/labels', params={'per_page': 100})
        if not resp:
            return

        existing = {label['name'].lower() for label in resp.json()}
        for label in REQUIRED_LABELS:
            if label['name'].lower() not in existing:
                result = self._github_api('POST', '/labels', json=label)
                if result:
                    logger.info(f"Created label: {label['name']}")

    # =========================================================================
    # Known Models Store (Schema v2)
    # =========================================================================

    def _models_path(self) -> Path:
        return self.storage_dir / "known_models.json"

    def _load_known_models(self) -> Dict[str, Dict[str, KnownModel]]:
        """Load known models. Handles v1 → v2 migration."""
        path = self._models_path()
        if not path.exists():
            return {"anthropic": {}, "openai": {}, "google": {}}

        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Could not load known models: {e}")
            return {"anthropic": {}, "openai": {}, "google": {}}

        schema_version = data.get("schema_version", 1)

        if schema_version >= 2:
            models = {}
            for provider, provider_models in data.get("models", {}).items():
                models[provider] = {}
                for model_id, model_data in provider_models.items():
                    models[provider][model_id] = KnownModel.from_dict(model_data)
            return models

        # Schema v1 migration: flat lists → platform-aware objects
        logger.info("Migrating known_models.json from schema v1 to v2")
        now = datetime.now(timezone.utc).isoformat()
        models = {}
        for provider, model_list in data.get("models", {}).items():
            models[provider] = {}
            if isinstance(model_list, list):
                for raw_id in model_list:
                    nid = normalize_model_id(raw_id, provider)
                    if nid not in models[provider]:
                        models[provider][nid] = KnownModel(
                            first_seen=now,
                            platforms={"direct_api": PlatformStatus(available=True, first_seen=now)},
                        )

        for p in ["anthropic", "openai", "google"]:
            if p not in models:
                models[p] = {}

        self._save_known_models(models)
        return models

    def _save_known_models(self, models: Dict[str, Dict[str, KnownModel]]):
        data = {
            "schema_version": SCHEMA_VERSION,
            "models": {},
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        for provider, provider_models in models.items():
            data["models"][provider] = {
                model_id: model.to_dict()
                for model_id, model in provider_models.items()
            }

        try:
            with open(self._models_path(), 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save known models: {e}")

    # =========================================================================
    # Phase 1: Deterministic Model Scraping + Diff
    # =========================================================================

    def _scrape_all_platforms(self) -> Dict[str, Dict[str, Set[str]]]:
        """Scrape all configured platform doc pages. Returns {provider: {platform: set(model_ids)}}."""
        results: Dict[str, Dict[str, Set[str]]] = {}

        for provider, platforms in PLATFORMS.items():
            results[provider] = {}
            active = self.active_platforms.get(provider, [])

            for platform_key, platform_info in platforms.items():
                if platform_key not in active:
                    continue

                parser_name = platform_info["parser"]
                parser_fn = getattr(self.parsers, parser_name, None)

                if not parser_fn:
                    logger.error(f"Parser not found: {parser_name}")
                    results[provider][platform_key] = set()
                    continue

                logger.info(f"Scraping {provider}/{platform_key}: {platform_info['url']}")
                try:
                    found = parser_fn(platform_info["url"])
                    results[provider][platform_key] = found
                    preview = sorted(found)[:5]
                    logger.info(f"  Found {len(found)} models: {preview}{'...' if len(found) > 5 else ''}")
                except Exception as e:
                    logger.error(f"  Parser error: {e}")
                    results[provider][platform_key] = set()

                time.sleep(self.fetch_delay_seconds)

        return results

    def _validate_scrape_results(
        self,
        scraped: Dict[str, Dict[str, Set[str]]],
        known: Dict[str, Dict[str, KnownModel]]
    ) -> Dict[str, Dict[str, Set[str]]]:
        """Validate scrape results. Use cache if a platform drops to 0 unexpectedly."""
        validated = {}

        for provider, platforms in scraped.items():
            validated[provider] = {}
            for platform_key, found_models in platforms.items():
                cached_count = sum(
                    1 for m in known.get(provider, {}).values()
                    if platform_key in m.platforms and m.platforms[platform_key].available
                )

                if cached_count > 0 and len(found_models) == 0:
                    logger.error(
                        f"{provider}/{platform_key}: Found 0 models but cache has {cached_count}. "
                        f"Possible page structure change. Using cached data."
                    )
                    cached_models = set()
                    for model_id, model in known.get(provider, {}).items():
                        if platform_key in model.platforms and model.platforms[platform_key].available:
                            cached_models.add(model_id)
                    validated[provider][platform_key] = cached_models
                elif cached_count > 0 and len(found_models) < cached_count * 0.5:
                    logger.warning(
                        f"{provider}/{platform_key}: Found {len(found_models)} models but cache "
                        f"has {cached_count}. Significant drop — review parser."
                    )
                    validated[provider][platform_key] = found_models
                else:
                    validated[provider][platform_key] = found_models

        return validated

    def _compute_deltas(
        self,
        scraped: Dict[str, Dict[str, Set[str]]],
        known: Dict[str, Dict[str, KnownModel]]
    ) -> List[ModelDelta]:
        """Compare scraped models against known to find new models or new platform availability."""
        deltas = []
        now = datetime.now(timezone.utc).isoformat()

        for provider, platforms in scraped.items():
            all_found: Dict[str, Dict[str, PlatformStatus]] = {}

            for platform_key, model_ids in platforms.items():
                for model_id in model_ids:
                    if model_id not in all_found:
                        all_found[model_id] = {}
                    all_found[model_id][platform_key] = PlatformStatus(
                        available=True, first_seen=now,
                    )

            provider_known = known.get(provider, {})

            for model_id, found_platforms in all_found.items():
                if model_id not in provider_known:
                    deltas.append(ModelDelta(
                        provider=provider,
                        model_id=model_id,
                        is_new_model=True,
                        new_platforms=list(found_platforms.keys()),
                        all_platforms=found_platforms,
                    ))
                else:
                    existing = provider_known[model_id]
                    new_platforms = []
                    for plat_key in found_platforms:
                        if plat_key not in existing.platforms or not existing.platforms[plat_key].available:
                            new_platforms.append(plat_key)

                    if new_platforms:
                        merged = {}
                        for k, v in existing.platforms.items():
                            merged[k] = v
                        for k, v in found_platforms.items():
                            if k not in merged or not merged[k].available:
                                merged[k] = v
                        deltas.append(ModelDelta(
                            provider=provider,
                            model_id=model_id,
                            is_new_model=False,
                            new_platforms=new_platforms,
                            all_platforms=merged,
                        ))

        return deltas

    def _apply_deltas(self, deltas: List[ModelDelta],
                       known: Dict[str, Dict[str, KnownModel]]):
        """Apply deltas to the known models store (in-memory)."""
        now = datetime.now(timezone.utc).isoformat()
        for delta in deltas:
            provider_known = known.setdefault(delta.provider, {})
            if delta.is_new_model:
                provider_known[delta.model_id] = KnownModel(
                    first_seen=now,
                    platforms={
                        k: PlatformStatus(available=True, first_seen=now)
                        for k in delta.new_platforms
                    },
                )
            else:
                existing = provider_known[delta.model_id]
                for plat_key in delta.new_platforms:
                    existing.platforms[plat_key] = PlatformStatus(
                        available=True, first_seen=now,
                    )

    # =========================================================================
    # Phase 2: LLM Enrichment
    # =========================================================================

    def _enrich_delta(self, delta: ModelDelta) -> str:
        """Use LLM to generate a description. Never asks "is this new?"."""
        if not self.news_enrichment:
            return ""

        prompt = f"""<task>
Describe the AI model "{delta.model_id}" from {delta.provider}.
</task>

<instructions>
Provide a brief (3-5 sentence) factual description of this model's capabilities,
intended use cases, and any notable features. If you are not familiar with this
specific model, say so and provide what you can infer from the naming convention.

Do NOT speculate about release dates, availability, or pricing.
Do NOT say whether this model is "new" — that determination has already been made.
Focus only on capabilities and technical characteristics.
</instructions>"""

        response, error = self.llm.query(prompt)
        if error:
            logger.warning(f"LLM enrichment failed for {delta.model_id}: {error}")
            return ""

        return response.strip() if response else ""

    # =========================================================================
    # Phase 3: Issue Management
    # =========================================================================

    def _format_availability_matrix(self, provider: str,
                                     platforms: Dict[str, PlatformStatus]) -> str:
        provider_platforms = PLATFORMS.get(provider, {})
        active = self.active_platforms.get(provider, [])

        lines = [
            "| Platform | Available | Since |",
            "|----------|-----------|-------|",
        ]

        for plat_key in active:
            plat_info = provider_platforms.get(plat_key, {})
            plat_name = plat_info.get("name", plat_key)

            if plat_key in platforms and platforms[plat_key].available:
                since = platforms[plat_key].first_seen[:10] if platforms[plat_key].first_seen else "—"
                lines.append(f"| {plat_name} | ✅ | {since} |")
            else:
                lines.append(f"| {plat_name} | ❌ | — |")

        return "\n".join(lines)

    def _provider_emoji(self, provider: str) -> str:
        return {"anthropic": "🟠", "openai": "🔵", "google": "🟡"}.get(provider, "⚪")

    def _provider_display_name(self, provider: str) -> str:
        return {"anthropic": "Anthropic", "openai": "OpenAI", "google": "Google"}.get(provider, provider)

    def _create_new_model_issue(self, delta: ModelDelta, enrichment: str) -> Optional[int]:
        emoji = self._provider_emoji(delta.provider)
        provider_name = self._provider_display_name(delta.provider)

        title = f"{emoji} New {provider_name} Model: {delta.model_id}"
        matrix = self._format_availability_matrix(delta.provider, delta.all_platforms)

        body = f"""## {emoji} New {provider_name} Model: {delta.model_id}

**Provider:** {provider_name}
**First Detected:** {datetime.now(timezone.utc).strftime('%Y-%m-%d')}
**Type:** API Model

### Availability Matrix

{matrix}
"""

        if enrichment:
            body += f"\n### Details\n\n{enrichment}\n"

        body += "\n### Source Links\n"
        for plat_key in delta.new_platforms:
            plat_info = PLATFORMS.get(delta.provider, {}).get(plat_key, {})
            if plat_info:
                body += f"- [{plat_info['name']}]({plat_info['url']})\n"

        body += "\n---\n*Detected by AI Model Availability Monitor v2*\n"
        labels = ["type:release", delta.provider, "auto-close-7d"]

        if self.dry_run:
            logger.info(f"[DRY RUN] Would create issue: {title}")
            logger.info(f"[DRY RUN] Labels: {labels}")
            return None

        resp = self._github_api('POST', '/issues', json={
            'title': title, 'body': body, 'labels': labels,
        })

        if resp:
            issue_number = resp.json()['number']
            logger.info(f"Created issue #{issue_number}: {title}")
            return issue_number

        self.errors.append(f"Failed to create issue for {delta.model_id}")
        return None

    def _update_existing_issue(self, delta: ModelDelta,
                                known_model: KnownModel) -> bool:
        if not known_model.issue_number:
            logger.warning(f"No issue number for {delta.model_id} — cannot post update")
            return False

        new_plat_names = []
        for plat_key in delta.new_platforms:
            plat_info = PLATFORMS.get(delta.provider, {}).get(plat_key, {})
            new_plat_names.append(plat_info.get("name", plat_key))

        matrix = self._format_availability_matrix(delta.provider, delta.all_platforms)

        body = f"""## 🆕 Platform Update

**{delta.model_id}** is now available on **{', '.join(new_plat_names)}**!

### Updated Availability Matrix

{matrix}

---
*Detected by AI Model Availability Monitor v2*
"""

        if self.dry_run:
            logger.info(
                f"[DRY RUN] Would comment on issue #{known_model.issue_number}: "
                f"{delta.model_id} → {', '.join(new_plat_names)}"
            )
            return True

        resp = self._github_api(
            'POST', f'/issues/{known_model.issue_number}/comments',
            json={'body': body}
        )

        if resp:
            logger.info(f"Updated issue #{known_model.issue_number}: {delta.model_id} → {', '.join(new_plat_names)}")
            return True
        return False

    def _close_old_issues(self):
        if self.dry_run:
            return

        resp = self._github_api('GET', '/issues', params={
            'labels': 'auto-close-7d', 'state': 'open', 'per_page': 100
        })
        if not resp:
            return

        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        for issue in resp.json():
            created = issue.get('created_at', '')
            try:
                created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                if created_dt < cutoff:
                    self._github_api('PATCH', f"/issues/{issue['number']}", json={'state': 'closed'})
                    logger.info(f"Auto-closed issue #{issue['number']}: {issue['title'][:50]}")
            except Exception:
                pass

    def _create_error_issue(self):
        if self.dry_run or not self.errors:
            return

        title = f"🐛 Monitor errors: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"
        body = "## Errors during monitoring run\n\n"
        for err in self.errors:
            body += f"- {err}\n"
        body += "\n---\n*Auto-generated by AI Model Availability Monitor v2*"

        self._github_api('POST', '/issues', json={
            'title': title, 'body': body, 'labels': ['type:bug'],
        })

    # =========================================================================
    # Watchlist Management (preserved from v1)
    # =========================================================================

    def _fetch_open_watchlist_issues(self) -> List[dict]:
        if self.dry_run:
            return []
        resp = self._github_api('GET', '/issues', params={
            'labels': 'type:watchlist', 'state': 'open', 'per_page': 100
        })
        return resp.json() if resp else []

    def _parse_watchlist_issue(self, issue: dict) -> Optional[WatchlistItem]:
        body = issue.get('body', '')
        if not body:
            return None

        watch_for = ""
        why = ""
        sources = []
        expires_days = 180

        watch_match = re.search(
            r'(?:What to watch for|### What to watch for)\s*\n+(.+?)(?=\n\s*(?:###|Why|$))',
            body, re.IGNORECASE | re.DOTALL
        )
        if watch_match:
            watch_for = watch_match.group(1).strip()

        why_match = re.search(
            r'(?:Why it matters|### Why it matters)\s*\n+(.+?)(?=\n\s*(?:###|Sources|$))',
            body, re.IGNORECASE | re.DOTALL
        )
        if why_match:
            why = why_match.group(1).strip()

        if 'all sources' in body.lower():
            sources = ['openai', 'anthropic', 'gemini']
        else:
            for s in ['openai', 'anthropic', 'gemini']:
                if s in body.lower():
                    sources.append(s)
        if not sources:
            sources = ['openai', 'anthropic', 'gemini']

        expires_match = re.search(r'Expires.*?(\d+)', body, re.IGNORECASE)
        if expires_match:
            try:
                expires_days = int(expires_match.group(1))
            except ValueError:
                pass

        created_at = issue.get('created_at', datetime.now(timezone.utc).isoformat())
        try:
            created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            expires_at = (created_dt + timedelta(days=expires_days)).isoformat()
        except Exception:
            expires_at = (datetime.now(timezone.utc) + timedelta(days=expires_days)).isoformat()

        if not watch_for:
            watch_for = issue.get('title', 'Unknown')

        return WatchlistItem(
            issue_number=issue['number'],
            title=issue.get('title', 'Unknown'),
            watch_for=watch_for,
            why=why,
            sources=sources,
            created_at=created_at,
            expires_at=expires_at,
        )

    def _generate_watchlist_md(self, items: List[WatchlistItem]) -> str:
        now = datetime.now(timezone.utc)
        lines = [
            "# Watchlist", "",
            "> Auto-generated from open GitHub issues with `type:watchlist` label.",
            "> Manage via GitHub issues. Do not edit manually.", ">",
            f"> Last updated: {now.isoformat()}", "",
        ]

        if not items:
            lines.append("*No active watchlist items.*")
        else:
            for item in items:
                try:
                    expires_dt = datetime.fromisoformat(item.expires_at.replace('Z', '+00:00'))
                    days_remaining = (expires_dt - now).days
                    expires_str = f"{expires_dt.strftime('%Y-%m-%d')} ({days_remaining} days remaining)"
                except Exception:
                    expires_str = item.expires_at

                lines.extend([
                    "---", "",
                    f"## {item.title}",
                    f"- **Issue:** #{item.issue_number}",
                    f"- **Watch for:** {item.watch_for}",
                ])
                if item.why:
                    lines.append(f"- **Why:** {item.why}")
                lines.extend([
                    f"- **Sources:** {', '.join(item.sources)}",
                    f"- **Expires:** {expires_str}", "",
                ])

        return '\n'.join(lines)

    def _close_expired_watchlist(self, items: List[WatchlistItem]):
        now = datetime.now(timezone.utc)
        for item in items:
            try:
                expires_dt = datetime.fromisoformat(item.expires_at.replace('Z', '+00:00'))
                if now > expires_dt:
                    self._github_api('POST', f'/issues/{item.issue_number}/comments', json={
                        'body': (
                            f"⏰ This watchlist item has expired "
                            f"({expires_dt.strftime('%Y-%m-%d')}).\n\n"
                            f"Reopening this issue will reset the expiration timer."
                        )
                    })
                    self._github_api('PATCH', f'/issues/{item.issue_number}', json={'state': 'closed'})
                    logger.info(f"Closed expired watchlist #{item.issue_number}")
            except Exception as e:
                logger.warning(f"Watchlist expiry check failed #{item.issue_number}: {e}")

    def _resolve_watchlist_item(self, watchlist_issue_number: int, release_issue_number: int):
        self._github_api('POST', f'/issues/{watchlist_issue_number}/comments', json={
            'body': f"✅ **Resolved!** See release issue #{release_issue_number}."
        })
        self._github_api('PATCH', f'/issues/{watchlist_issue_number}', json={'state': 'closed'})
        logger.info(f"Resolved watchlist #{watchlist_issue_number} via #{release_issue_number}")

    def _check_watchlist_resolution(self, delta: ModelDelta,
                                     watchlist_items: List[WatchlistItem]) -> Optional[int]:
        model_lower = delta.model_id.lower()
        for item in watchlist_items:
            if model_lower in item.watch_for.lower():
                return item.issue_number
        return None

    # =========================================================================
    # Mistake Tracking (preserved from v1)
    # =========================================================================

    def _process_false_positives(self):
        if self.dry_run:
            return

        resp = self._github_api('GET', '/issues', params={
            'labels': 'not-relevant', 'state': 'open', 'per_page': 100
        })
        if not resp:
            return

        for issue in resp.json():
            labels = [l['name'] for l in issue.get('labels', [])]
            if 'processed' not in labels:
                self._github_api('POST', f"/issues/{issue['number']}/labels", json={
                    'labels': ['processed']
                })
                self._github_api('PATCH', f"/issues/{issue['number']}", json={'state': 'closed'})
                logger.info(f"Processed false positive #{issue['number']}: {issue['title'][:50]}")

    # =========================================================================
    # Main Orchestration
    # =========================================================================

    def check_all(self):
        """Main entry point: scrape → diff → enrich → issue."""
        logger.info("=" * 60)
        logger.info("AI Model Availability Monitor v2")
        logger.info(f"Started: {datetime.now(timezone.utc)}")
        if self.dry_run:
            logger.info("MODE: DRY RUN (no GitHub changes)")
        if self.seed:
            logger.info("MODE: SEED (first-run baseline, no issues created)")
        logger.info("=" * 60)

        self._ensure_labels_exist()
        self._process_false_positives()

        # Watchlist management
        watchlist_issues = self._fetch_open_watchlist_issues()
        watchlist_items = [
            item for issue in watchlist_issues
            if (item := self._parse_watchlist_issue(issue)) is not None
        ]
        logger.info(f"Loaded {len(watchlist_items)} watchlist items")
        self._close_expired_watchlist(watchlist_items)

        # Regenerate watchlist.md
        watchlist_issues = self._fetch_open_watchlist_issues()
        watchlist_items = [
            item for issue in watchlist_issues
            if (item := self._parse_watchlist_issue(issue)) is not None
        ]
        watchlist_md = self._generate_watchlist_md(watchlist_items)
        watchlist_path = self.storage_dir / "watchlist.md"

        existing = watchlist_path.read_text() if watchlist_path.exists() else ""
        if re.sub(r'Last updated:.*', '', existing).strip() != re.sub(r'Last updated:.*', '', watchlist_md).strip():
            watchlist_path.write_text(watchlist_md)
            logger.info("Updated watchlist.md")

        self._close_old_issues()

        # =================================================================
        # PHASE 1: Deterministic model scraping
        # =================================================================
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: Scraping platform documentation pages")
        logger.info("=" * 60)

        known = self._load_known_models()
        total_known = sum(len(m) for m in known.values())
        logger.info(f"Loaded {total_known} known models from store")

        scraped = self._scrape_all_platforms()
        total_scraped = sum(len(m) for p in scraped.values() for m in p.values())
        logger.info(f"Scraped {total_scraped} model entries across all platforms")

        scraped = self._validate_scrape_results(scraped, known)

        # First-run seeding
        is_first_run = total_known == 0 or self.seed
        if is_first_run:
            logger.info("\n" + "-" * 40)
            logger.info("FIRST RUN: Seeding baseline. No issues will be created.")
            logger.info("-" * 40)

            now = datetime.now(timezone.utc).isoformat()
            for provider, platforms in scraped.items():
                for platform_key, model_ids in platforms.items():
                    for model_id in model_ids:
                        if model_id not in known.get(provider, {}):
                            known.setdefault(provider, {})[model_id] = KnownModel(
                                first_seen=now, platforms={},
                            )
                        known[provider][model_id].platforms[platform_key] = PlatformStatus(
                            available=True, first_seen=now,
                        )

            self._save_known_models(known)
            total_seeded = sum(len(m) for m in known.values())
            logger.info(f"Seeded {total_seeded} models as known baseline")

            if os.getenv('GITHUB_ACTIONS'):
                print(f"::notice::First run: seeded {total_seeded} models.")
            logger.info("=" * 60)
            logger.info("✅ Seed complete. No issues created.")
            logger.info("=" * 60)
            return

        # Compute deltas
        deltas = self._compute_deltas(scraped, known)
        logger.info(f"\nFound {len(deltas)} changes:")
        for d in deltas:
            tag = "NEW MODEL" if d.is_new_model else "NEW PLATFORM"
            logger.info(f"  {tag}: {d.provider}/{d.model_id} → {', '.join(d.new_platforms)}")

        if not deltas:
            logger.info("\n" + "=" * 60)
            logger.info("✅ No new models or platform changes detected")
            logger.info("=" * 60)
            if os.getenv('GITHUB_ACTIONS'):
                print("::notice::No new models or platform changes detected")
            return

        # =================================================================
        # PHASE 2: LLM Enrichment
        # =================================================================
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: LLM Enrichment")
        logger.info("=" * 60)

        enrichments: Dict[str, str] = {}
        new_model_deltas = [d for d in deltas if d.is_new_model]

        if new_model_deltas and self.news_enrichment:
            for delta in new_model_deltas:
                logger.info(f"Enriching {delta.provider}/{delta.model_id}...")
                enrichments[delta.model_id] = self._enrich_delta(delta)
        else:
            logger.info("No new models to enrich (or enrichment disabled)")

        # =================================================================
        # PHASE 3: Issue Management
        # =================================================================
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: Creating/updating GitHub issues")
        logger.info("=" * 60)

        issues_created = 0
        issues_updated = 0

        for delta in deltas:
            if delta.is_new_model:
                enrichment = enrichments.get(delta.model_id, "")
                issue_num = self._create_new_model_issue(delta, enrichment)

                if issue_num:
                    issues_created += 1
                    if delta.model_id in known.get(delta.provider, {}):
                        known[delta.provider][delta.model_id].issue_number = issue_num
                    resolved = self._check_watchlist_resolution(delta, watchlist_items)
                    if resolved:
                        self._resolve_watchlist_item(resolved, issue_num)
                elif self.dry_run:
                    issues_created += 1
            else:
                existing_model = known.get(delta.provider, {}).get(delta.model_id)
                if existing_model and self._update_existing_issue(delta, existing_model):
                    issues_updated += 1

        self._apply_deltas(deltas, known)
        self._save_known_models(known)

        if self.errors:
            logger.warning(f"⚠️ {len(self.errors)} errors occurred")
            self._create_error_issue()

        if os.getenv('GITHUB_ACTIONS'):
            if issues_created or issues_updated:
                print(f"::notice::Created {issues_created}, updated {issues_updated} issue(s)")
            else:
                print("::notice::No issues created or updated")
            for error in self.errors:
                print(f"::warning::{error}")

        logger.info("\n" + "=" * 60)
        if issues_created or issues_updated:
            logger.info(f"✅ Created {issues_created} issue(s), updated {issues_updated} issue(s)")
        else:
            logger.info("✅ No issues created or updated")
        logger.info("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AI Model Availability Monitor v2 — "
                    "Deterministic model detection via doc-page scraping."
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show what would be created/updated without touching GitHub'
    )
    parser.add_argument(
        '--seed', action='store_true',
        help='Force first-run seeding: scrape all, save baseline, create no issues'
    )
    parser.add_argument(
        '--config', default='config.json',
        help='Path to config file (default: config.json)'
    )

    args = parser.parse_args()

    try:
        monitor = WebsiteMonitor(
            config_path=args.config,
            dry_run=args.dry_run,
            seed=args.seed,
        )
        monitor.check_all()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Fatal: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
