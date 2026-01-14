#!/usr/bin/env python3
"""
AI-Powered Website Change Monitor
Creates GitHub Issues when product/API changes are detected.
"""

import os
import json
import hashlib
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
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
# Data Classes
# =============================================================================

@dataclass
class ContentItem:
    """A single news/changelog item."""
    id: str
    title: str
    description: str
    date: str
    link: str
    source_name: str
    full_content: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass 
class HandledRelease:
    """A release we've already created an issue for."""
    id: str
    title: str
    link: str
    issue_number: int
    detected_at: str
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'HandledRelease':
        return cls(**data)


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


@dataclass
class AnalysisResult:
    """Result of analyzing a content item."""
    item: ContentItem
    is_relevant: bool
    summary: str
    issue_body: str
    resolves_watchlist: Optional[int] = None


# =============================================================================
# Required Labels
# =============================================================================
# Note: We use labels with "type:" prefix for categorization (e.g., "type:release").
# This is a naming convention, NOT GitHub's "issue types" feature which requires
# organization-level configuration and GitHub Projects. Labels work universally
# across all repository types and don't require special permissions.

REQUIRED_LABELS = [
    {"name": "type:release", "color": "0E8A16", "description": "AI release/update detected"},
    {"name": "type:watchlist", "color": "1D76DB", "description": "Watching for future availability"},
    {"name": "type:bug", "color": "D73A4A", "description": "System error"},
    {"name": "openai", "color": "412991", "description": "OpenAI source"},
    {"name": "anthropic", "color": "D97706", "description": "Anthropic source"},
    {"name": "gemini", "color": "FBBC04", "description": "Google Gemini source"},
    {"name": "not-relevant", "color": "E99695", "description": "False positive"},
    {"name": "auto-close-7d", "color": "BFD4F2", "description": "Auto-closes after 7 days"},
]


# =============================================================================
# LLM Client
# =============================================================================

class LLMClient:
    """Handles all LLM API interactions."""
    
    def __init__(self, config: dict):
        self.api_url = config.get('url', '')
        self.api_token = config.get('token', '')
        self.model = config.get('model', 'google-gemini-2.5-pro')
        self.temperature = config.get('temperature', 0.1)
    
    def query(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Send a prompt to the LLM.
        
        Returns:
            Tuple of (response_text, error_message)
        """
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
            
            # Extract from 'message' field (contains actual LLM response)
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
# Website Monitor
# =============================================================================

class WebsiteMonitor:
    """Monitors websites for AI product releases."""
    
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self._override_with_env_vars()
        
        self.storage_dir = Path(self.config.get("storage_dir", "storage"))
        self.storage_dir.mkdir(exist_ok=True)
        
        self.llm = LLMClient(self.config.get('asksage_api', {}))
        
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.github_repo = os.getenv('GITHUB_REPOSITORY')
        
        self.errors: List[str] = []
        
        # Limits: 7 days OR 10 items, whichever comes first
        self.time_window_days = 7
        self.max_items_per_source = 10
        
        # Parallelization settings
        self.max_fetch_workers = 5
        self.max_llm_workers = 3
        
        # Token budget settings
        self.chars_per_token = 3  # Conservative estimate
        self.response_buffer_tokens = 10_000
        self.max_combined_context = 160_000
        
        # Cache for known models
        self._known_models_cache: Optional[Dict[str, List[str]]] = None
        
        # Cache for context files
        self._watchlist_content: Optional[str] = None
        self._mistakes_content: Optional[str] = None
        
        # Browser-like session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        
        # Model documentation URLs
        self.model_docs = {
            "anthropic": "https://docs.anthropic.com/en/docs/about-claude/models",
            "openai": "https://platform.openai.com/docs/models",
            "google": "https://ai.google.dev/gemini-api/docs/models"
        }
        
        self.websites = {
            "openai": {
                "url": "https://openai.com/news/rss.xml",
                "name": "OpenAI",
                "type": "rss",
                "emoji": "🔵",
                "label": "openai",
                "base_url": "https://openai.com"
            },
            "anthropic": {
                "url": "https://www.anthropic.com/news",
                "name": "Anthropic",
                "type": "html_news",
                "emoji": "🟠", 
                "label": "anthropic",
                "base_url": "https://www.anthropic.com"
            },
            "gemini": {
                "url": "https://ai.google.dev/gemini-api/docs/changelog",
                "name": "Google Gemini",
                "type": "html_changelog",
                "emoji": "🟡",
                "label": "gemini",
                "base_url": "https://ai.google.dev"
            },
        }
    
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
            "storage_dir": "storage"
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
    # Utilities
    # =========================================================================
    
    def _generate_id(self, *args) -> str:
        combined = '|'.join(str(a).strip() for a in args if a)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        if not date_str:
            return None
        try:
            from dateutil import parser
            dt = parser.parse(date_str, fuzzy=True)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None
    
    def _is_recent(self, date: Optional[datetime]) -> bool:
        if not date:
            return True
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.time_window_days)
        return date >= cutoff
    
    def _get_provider_key(self, source_name: str) -> str:
        """Map source name to provider key for model lookup."""
        name_lower = source_name.lower()
        if 'anthropic' in name_lower or 'claude' in name_lower:
            return 'anthropic'
        elif 'openai' in name_lower:
            return 'openai'
        elif 'google' in name_lower or 'gemini' in name_lower:
            return 'google'
        return ''
    
    def _github_api(self, method: str, endpoint: str, **kwargs) -> Optional[requests.Response]:
        """
        Make a GitHub API request.
        
        Args:
            method: HTTP method (GET, POST, PATCH, etc.)
            endpoint: API endpoint (e.g., '/issues')
            **kwargs: Additional arguments passed to requests.request()
        
        Returns:
            Response object on success, None on failure.
            Failures are logged with details but not added to self.errors
            (callers decide if the error is workflow-critical).
        """
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
            if status_code == 403:
                logger.warning("GitHub API 403: possible rate limit or insufficient permissions")
            elif status_code == 404:
                logger.warning(f"GitHub API 404: endpoint or resource not found: {endpoint}")
            elif status_code == 422:
                # Unprocessable entity - often means validation error
                try:
                    error_body = e.response.json()
                    logger.error(f"GitHub API validation error: {error_body}")
                except Exception:
                    pass
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"GitHub API connection error ({method} {endpoint}): {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"GitHub API request failed ({method} {endpoint}): {e}")
            return None
    
    # =========================================================================
    # Label Management
    # =========================================================================
    
    def _ensure_labels_exist(self):
        """Create required labels if they don't exist."""
        if not self.github_token or not self.github_repo:
            return
        
        resp = self._github_api('GET', '/labels', params={'per_page': 100})
        if not resp:
            logger.warning("Could not fetch existing labels, skipping label creation")
            return
        
        existing = {label['name'].lower() for label in resp.json()}
        
        for label in REQUIRED_LABELS:
            if label['name'].lower() not in existing:
                result = self._github_api('POST', '/labels', json=label)
                if result:
                    logger.info(f"Created label: {label['name']}")
                else:
                    logger.warning(f"Failed to create label: {label['name']}")
    
    # =========================================================================
    # Known Models
    # =========================================================================
        
    def _load_cached_models(self) -> Dict[str, List[str]]:
        """Load cached models from storage."""
        cache_path = self.storage_dir / "known_models.json"
        try:
            if cache_path.exists():
                with open(cache_path) as f:
                    data = json.load(f)
                    return data.get('models', {})
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in cached models file: {e}")
        except Exception as e:
            logger.warning(f"Could not load cached models: {e}")
        return {}

    def _save_cached_models(self, models: Dict[str, List[str]]):
        """Save models to cache."""
        cache_path = self.storage_dir / "known_models.json"
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    'models': models,
                    'updated': datetime.now(timezone.utc).isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save cached models: {e}")

    def _fetch_known_models(self) -> Dict[str, List[str]]:
        """
        Fetch current model lists from each provider's documentation.
        Uses cached models as base and merges in newly discovered models.
        """
        if self._known_models_cache is not None:
            return self._known_models_cache
        
        # Start with cached models
        models = self._load_cached_models()
        if not models:
            models = {'anthropic': [], 'openai': [], 'google': []}
        
        models_updated = False
        
        # Anthropic models
        try:
            resp = self.session.get(self.model_docs['anthropic'], timeout=20)
            if resp.ok:
                text = resp.text.lower()
                anthropic_pattern = r'claude-[\w\d.-]+'
                found = set(re.findall(anthropic_pattern, text))
                new_models = found - set(models.get('anthropic', []))
                if new_models:
                    models['anthropic'] = sorted(set(models.get('anthropic', [])) | found)
                    models_updated = True
                    logger.info(f"Found {len(new_models)} new Anthropic models")
                logger.info(f"Anthropic: {len(models['anthropic'])} models total")
            else:
                logger.warning(f"Anthropic models fetch failed: HTTP {resp.status_code}")
        except requests.exceptions.Timeout:
            logger.warning("Anthropic models fetch timed out")
        except Exception as e:
            logger.warning(f"Failed to fetch Anthropic models: {e}")
        
        # OpenAI models
        try:
            resp = self.session.get(self.model_docs['openai'], timeout=20)
            if resp.ok:
                text = resp.text.lower()
                openai_patterns = [
                    r'gpt-[\w\d.-]+',
                    r'o[1-9]-[\w-]+',
                    r'o[1-9](?!\w)',
                    r'davinci-[\w\d-]+',
                    r'codex-[\w\d-]+',
                ]
                found = set()
                for pattern in openai_patterns:
                    found.update(re.findall(pattern, text))
                new_models = found - set(models.get('openai', []))
                if new_models:
                    models['openai'] = sorted(set(models.get('openai', [])) | found)
                    models_updated = True
                    logger.info(f"Found {len(new_models)} new OpenAI models")
                logger.info(f"OpenAI: {len(models['openai'])} models total")
            else:
                logger.warning(f"OpenAI models fetch failed: HTTP {resp.status_code}")
        except requests.exceptions.Timeout:
            logger.warning("OpenAI models fetch timed out")
        except Exception as e:
            logger.warning(f"Failed to fetch OpenAI models: {e}")
        
        # Google Gemini models
        try:
            resp = self.session.get(self.model_docs['google'], timeout=20)
            if resp.ok:
                text = resp.text.lower()
                gemini_pattern = r'gemini-[\w\d.-]+'
                found = set(re.findall(gemini_pattern, text))
                new_models = found - set(models.get('google', []))
                if new_models:
                    models['google'] = sorted(set(models.get('google', [])) | found)
                    models_updated = True
                    logger.info(f"Found {len(new_models)} new Google models")
                logger.info(f"Google: {len(models['google'])} models total")
            else:
                logger.warning(f"Google models fetch failed: HTTP {resp.status_code}")
        except requests.exceptions.Timeout:
            logger.warning("Google models fetch timed out")
        except Exception as e:
            logger.warning(f"Failed to fetch Google models: {e}")
        
        # Apply fallbacks only if we have NO models for a provider
        if not models.get('anthropic'):
            models['anthropic'] = [
                'claude-4.5-opus', 'claude-4.5-sonnet', 'claude-4.5-haiku',
                'claude-4-opus', 'claude-4-sonnet',
                'claude-3.5-opus', 'claude-3.5-sonnet', 'claude-3.5-haiku',
                'claude-3-opus', 'claude-3-sonnet', 'claude-3-haiku'
            ]
            logger.info("Using fallback Anthropic model list")
            models_updated = True
        
        if not models.get('openai'):
            models['openai'] = [
                'gpt-5', 'gpt-5-turbo', 'gpt-4.5', 'gpt-4.5-turbo',
                'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4',
                'o1', 'o1-preview', 'o1-mini', 'o3', 'o3-mini',
                'gpt-3.5-turbo'
            ]
            logger.info("Using fallback OpenAI model list")
            models_updated = True
        
        if not models.get('google'):
            models['google'] = [
                'gemini-3-pro', 'gemini-3-flash',
                'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite',
                'gemini-2.0-flash', 'gemini-2.0-flash-lite',
                'gemini-1.5-pro', 'gemini-1.5-flash'
            ]
            logger.info("Using fallback Google model list")
            models_updated = True
        
        # Save if updated
        if models_updated:
            self._save_cached_models(models)
        
        self._known_models_cache = models
        return models
    
    # =========================================================================
    # Watchlist Management
    # =========================================================================
    
    def _fetch_open_watchlist_issues(self) -> List[dict]:
        """Fetch all open issues with type:watchlist label."""
        resp = self._github_api('GET', '/issues', params={
            'labels': 'type:watchlist',
            'state': 'open',
            'per_page': 100
        })
        if not resp:
            return []
        return resp.json()
    
    def _parse_watchlist_issue(self, issue: dict) -> Optional[WatchlistItem]:
        """Parse a watchlist issue into a WatchlistItem."""
        body = issue.get('body', '')
        if not body:
            return None
        
        watch_for = ""
        why = ""
        sources = []
        expires_days = 180
        
        # Extract "What to watch for" section
        watch_match = re.search(
            r'(?:What to watch for|### What to watch for)\s*\n+(.+?)(?=\n\s*(?:###|Why|$))',
            body, re.IGNORECASE | re.DOTALL
        )
        if watch_match:
            watch_for = watch_match.group(1).strip()
        
        # Extract "Why it matters" section
        why_match = re.search(
            r'(?:Why it matters|### Why it matters)\s*\n+(.+?)(?=\n\s*(?:###|Sources|$))',
            body, re.IGNORECASE | re.DOTALL
        )
        if why_match:
            why = why_match.group(1).strip()
        
        # Extract sources
        if 'all sources' in body.lower():
            sources = ['openai', 'anthropic', 'gemini']
        else:
            if 'openai' in body.lower():
                sources.append('openai')
            if 'anthropic' in body.lower():
                sources.append('anthropic')
            if 'gemini' in body.lower():
                sources.append('gemini')
        
        if not sources:
            sources = ['openai', 'anthropic', 'gemini']
        
        # Extract expiration
        expires_match = re.search(r'Expires.*?(\d+)', body, re.IGNORECASE)
        if expires_match:
            try:
                expires_days = int(expires_match.group(1))
            except ValueError:
                pass
        
        created_at = issue.get('created_at', datetime.now(timezone.utc).isoformat())
        try:
            created_dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            expires_dt = created_dt + timedelta(days=expires_days)
            expires_at = expires_dt.isoformat()
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
            expires_at=expires_at
        )
    
    def _generate_watchlist_md(self, items: List[WatchlistItem]) -> str:
        """Generate watchlist.md content from watchlist items."""
        now = datetime.now(timezone.utc)
        
        lines = [
            "# Watchlist",
            "",
            "> Auto-generated from open GitHub issues with `type:watchlist` label.",
            "> Manage via GitHub issues. Do not edit manually.",
            ">",
            f"> Last updated: {now.isoformat()}",
            "",
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
                    "---",
                    "",
                    f"## {item.title}",
                    f"- **Issue:** #{item.issue_number}",
                    f"- **Watch for:** {item.watch_for}",
                ])
                if item.why:
                    lines.append(f"- **Why:** {item.why}")
                lines.extend([
                    f"- **Sources:** {', '.join(item.sources)}",
                    f"- **Expires:** {expires_str}",
                    "",
                ])
        
        return '\n'.join(lines)
    
    def _close_expired_watchlist(self, items: List[WatchlistItem]):
        """Close watchlist items that have expired."""
        now = datetime.now(timezone.utc)
        
        for item in items:
            try:
                expires_dt = datetime.fromisoformat(item.expires_at.replace('Z', '+00:00'))
                if now > expires_dt:
                    self._github_api('POST', f'/issues/{item.issue_number}/comments', json={
                        'body': f"⏰ This watchlist item has expired after reaching its expiration date ({expires_dt.strftime('%Y-%m-%d')}).\n\nReopening this issue will reset the expiration timer."
                    })
                    self._github_api('PATCH', f'/issues/{item.issue_number}', json={
                        'state': 'closed'
                    })
                    logger.info(f"Closed expired watchlist item #{item.issue_number}")
            except Exception as e:
                logger.warning(f"Failed to check/close watchlist item #{item.issue_number}: {e}")
    
    def _resolve_watchlist_item(self, watchlist_issue_number: int, release_issue_number: int):
        """Mark a watchlist item as resolved."""
        self._github_api('POST', f'/issues/{watchlist_issue_number}/comments', json={
            'body': f"✅ **Resolved!**\n\nThis watchlist item has been resolved by release issue #{release_issue_number}."
        })
        
        self._github_api('PATCH', f'/issues/{watchlist_issue_number}', json={
            'state': 'closed'
        })
        
        logger.info(f"Resolved watchlist item #{watchlist_issue_number} via #{release_issue_number}")
    
    # =========================================================================
    # Mistakes / Learning Management
    # =========================================================================
    
    def _fetch_unprocessed_mistakes(self) -> List[dict]:
        """Fetch issues marked as not-relevant that haven't been processed yet."""
        resp = self._github_api('GET', '/issues', params={
            'labels': 'not-relevant',
            'state': 'open',
            'per_page': 100
        })
        if not resp:
            return []
        
        issues = []
        for issue in resp.json():
            labels = [l['name'] for l in issue.get('labels', [])]
            if 'type:release' in labels:
                issues.append(issue)
        
        return issues
    
    def _extract_user_feedback(self, issue: dict) -> Optional[str]:
        """Extract the user's comment explaining why the issue is not relevant."""
        issue_number = issue['number']
        
        resp = self._github_api('GET', f'/issues/{issue_number}/comments')
        if not resp:
            return None
        
        comments = resp.json()
        
        for comment in reversed(comments):
            user = comment.get('user', {})
            if user.get('type') != 'Bot' and '[bot]' not in user.get('login', ''):
                return comment.get('body', '')
        
        return None
    
    def _generate_lesson(self, issue: dict, feedback: str) -> Optional[Tuple[str, str]]:
        """
        Generate a lesson from a false positive.
        
        Returns:
            Tuple of (lesson_text, category) or None if generation fails
        """
        body = issue.get('body', '')
        title = issue.get('title', '')
        
        labels = [l['name'] for l in issue.get('labels', [])]
        source = 'Unknown'
        for label in labels:
            if label in ['openai', 'anthropic', 'gemini']:
                source = label.capitalize()
                break
        
        prompt = f"""<task>
Learn from a classification mistake to prevent similar errors in the future.
</task>

<false_positive>
<title>{title}</title>
<source>{source}</source>
<issue_body>
{body[:2500]}
</issue_body>
</false_positive>

<user_feedback>
{feedback}
</user_feedback>

<instructions>
Generate a concise lesson (2-4 sentences) that will prevent this type of mistake.

Structure your lesson as:
1. WHAT went wrong (the pattern that caused the error)
2. WHAT TO DO instead (positive guidance)
3. WHAT NOT TO DO (negative guidance)

Be specific - reference the exact type of content, keywords, or patterns to watch for.
</instructions>

<categories>
Choose the single best category:
- hallucinated_model: LLM imagined a model that doesn't exist
- existing_model: Treated an existing model as new
- consumer_not_api: Confused consumer product for API announcement  
- case_study: Flagged a customer story as a release
- partnership: Flagged business news as technical release
- policy_research: Flagged policy/research as technical release
- changelog_noise: Flagged minor changelog update that wasn't significant
- other: None of the above
</categories>

<output_format>
LESSON: [Your 2-4 sentence lesson with specific patterns to watch for]
CATEGORY: [single category from list above]
</output_format>"""

        response, error = self.llm.query(prompt)
        
        if error:
            logger.error(f"Failed to generate lesson: {error}")
            return None
        
        lesson = ""
        category = "other"
        
        lesson_match = re.search(r'LESSON:\s*(.+?)(?=CATEGORY:|$)', response, re.DOTALL | re.IGNORECASE)
        if lesson_match:
            lesson = lesson_match.group(1).strip()
        
        category_match = re.search(r'CATEGORY:\s*(\w+)', response, re.IGNORECASE)
        if category_match:
            category = category_match.group(1).lower()
        
        if not lesson:
            lesson = response.strip()
        
        return lesson, category
    
    def _append_to_mistakes_md(self, issue: dict, lesson: str, category: str):
        """Append a lesson to mistakes.md."""
        mistakes_path = self.storage_dir / "mistakes.md"
        
        existing_content = ""
        if mistakes_path.exists():
            existing_content = mistakes_path.read_text()
        
        now = datetime.now(timezone.utc)
        
        if not existing_content.strip():
            existing_content = f"""# Lessons Learned

> Auto-generated from issues marked `not-relevant`.
> These lessons inform future classification decisions.
>
> Last updated: {now.isoformat()}

"""
        
        # Update timestamp
        existing_content = re.sub(
            r'Last updated:.*',
            f'Last updated: {now.isoformat()}',
            existing_content
        )
        
        title = issue.get('title', 'Unknown')
        issue_number = issue['number']
        created_at = issue.get('created_at', '')[:10]
        
        labels = [l['name'] for l in issue.get('labels', [])]
        source = 'Unknown'
        for label in labels:
            if label in ['openai', 'anthropic', 'gemini']:
                source = label.capitalize()
                break
        
        lesson_title = lesson.split('.')[0][:60] if lesson else "Lesson learned"
        
        new_entry = f"""
---

## {lesson_title}
- **Issue:** #{issue_number}
- **Source:** {source}
- **Date:** {created_at}
- **Category:** `{category}`

{lesson}
"""
        
        full_content = existing_content.rstrip() + "\n" + new_entry
        
        mistakes_path.write_text(full_content)
        logger.info(f"Added lesson to mistakes.md from issue #{issue_number}")
    
    def _process_false_positives(self):
        """Process all unprocessed false positive issues."""
        issues = self._fetch_unprocessed_mistakes()
        
        if not issues:
            logger.info("No unprocessed false positives")
            return
        
        logger.info(f"Processing {len(issues)} false positive(s)")
        
        for issue in issues:
            issue_number = issue['number']
            
            feedback = self._extract_user_feedback(issue)
            if not feedback:
                self._github_api('POST', f'/issues/{issue_number}/comments', json={
                    'body': "⚠️ Please add a comment explaining why this issue is not relevant, so I can learn from this mistake."
                })
                logger.info(f"Requested feedback on #{issue_number}")
                continue
            
            result = self._generate_lesson(issue, feedback)
            if not result:
                self.errors.append(f"Failed to generate lesson for #{issue_number}")
                continue
            
            lesson, category = result
            
            self._append_to_mistakes_md(issue, lesson, category)
            
            self._github_api('POST', f'/issues/{issue_number}/comments', json={
                'body': f"📚 **Lesson learned!**\n\n> {lesson}\n\nThis has been added to my knowledge base to prevent similar mistakes."
            })
            self._github_api('PATCH', f'/issues/{issue_number}', json={
                'state': 'closed'
            })
            
            logger.info(f"Processed false positive #{issue_number}")
    
    # =========================================================================
    # Context Management
    # =========================================================================
    
    def _load_context_files(self) -> Tuple[str, str]:
        """Load watchlist.md and mistakes.md content."""
        watchlist_content = ""
        mistakes_content = ""
        
        watchlist_path = self.storage_dir / "watchlist.md"
        if watchlist_path.exists():
            watchlist_content = watchlist_path.read_text()
        
        mistakes_path = self.storage_dir / "mistakes.md"
        if mistakes_path.exists():
            mistakes_content = mistakes_path.read_text()
        
        return watchlist_content, mistakes_content
    
    def _calculate_token_budget(self, article_content: str, base_prompt: str,
                                watchlist: str, mistakes: str) -> str:
        """
        Determine whether to use combined or split query strategy.
        
        Returns:
            'combined' or 'split'
        """
        base_tokens = (len(base_prompt) + len(article_content)) // self.chars_per_token
        base_total = base_tokens + self.response_buffer_tokens
        
        context_tokens = (len(watchlist) + len(mistakes)) // self.chars_per_token
        
        if base_total + context_tokens < self.max_combined_context:
            return 'combined'
        else:
            return 'split'
    
    def _format_watchlist_for_prompt(self, watchlist_content: str, source: str) -> str:
        """Format watchlist content for inclusion in prompt, filtered by source."""
        if not watchlist_content.strip():
            return ""
        
        items = []
        current_item = None
        
        for line in watchlist_content.split('\n'):
            if line.startswith('## ') and not line.startswith('## Active'):
                if current_item:
                    items.append(current_item)
                current_item = {'title': line[3:].strip(), 'lines': []}
            elif current_item and line.strip():
                current_item['lines'].append(line)
        
        if current_item:
            items.append(current_item)
        
        source_lower = source.lower()
        relevant_items = []
        for item in items:
            item_text = '\n'.join(item['lines'])
            if source_lower in item_text.lower() or 'all' in item_text.lower():
                relevant_items.append(item)
        
        if not relevant_items:
            return ""
        
        lines = ["WATCHLIST - Things we are actively watching for:"]
        for i, item in enumerate(relevant_items, 1):
            issue_match = re.search(r'#(\d+)', '\n'.join(item['lines']))
            issue_num = issue_match.group(1) if issue_match else '?'
            
            watch_for = ""
            why = ""
            for line in item['lines']:
                if 'Watch for:' in line:
                    watch_for = line.split('Watch for:')[1].strip()
                elif 'Why:' in line:
                    why = line.split('Why:')[1].strip()
            
            lines.append(f"{i}. [Issue #{issue_num}] {item['title']}: {watch_for}")
            if why:
                lines.append(f"   Reason: {why}")
        
        lines.append("")
        lines.append("If this article announces that a watchlist item is NOW AVAILABLE or RESOLVED,")
        lines.append("mark it as RELEVANT and include 'RESOLVES_WATCHLIST: #<issue_number>' in your response.")
        
        return '\n'.join(lines)
    
    def _format_mistakes_for_prompt(self, mistakes_content: str) -> str:
        """Format mistakes content for inclusion in prompt."""
        if not mistakes_content.strip():
            return ""
        
        lessons = []
        current_lesson = None
        
        for line in mistakes_content.split('\n'):
            if line.startswith('## '):
                if current_lesson and current_lesson.get('text'):
                    lessons.append(current_lesson)
                current_lesson = {'title': line[3:].strip(), 'text': '', 'category': ''}
            elif current_lesson:
                if line.startswith('- **Category:**'):
                    cat_match = re.search(r'`(\w+)`', line)
                    if cat_match:
                        current_lesson['category'] = cat_match.group(1)
                elif not line.startswith('- **') and not line.startswith('---') and line.strip():
                    current_lesson['text'] += ' ' + line.strip()
        
        if current_lesson and current_lesson.get('text'):
            lessons.append(current_lesson)
        
        if not lessons:
            return ""
        
        # Limit to most recent lessons to save tokens
        lessons = lessons[-10:]
        
        lines = ["LESSONS FROM PAST MISTAKES - Do not repeat these errors:"]
        for i, lesson in enumerate(lessons, 1):
            cat = f"[{lesson['category']}]" if lesson['category'] else ""
            lines.append(f"{i}. {cat} {lesson['text'].strip()}")
        
        return '\n'.join(lines)
    
    # =========================================================================
    # Content Extraction
    # =========================================================================
    
    def _extract_rss(self, url: str, source_name: str) -> List[ContentItem]:
        """Extract items from RSS feed (max 10, within 7 days)."""
        items = []
        try:
            resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, 'xml')
            
            for item in soup.find_all('item'):
                # Stop if we hit max items
                if len(items) >= self.max_items_per_source:
                    break
                
                title = item.find('title')
                title = title.get_text(strip=True) if title else ""
                
                link = item.find('link')
                link = link.get_text(strip=True).rstrip(':') if link else ""
                
                desc = item.find('description')
                description = ""
                if desc:
                    desc_soup = BeautifulSoup(desc.get_text(), 'html.parser')
                    description = desc_soup.get_text(separator=' ', strip=True)
                
                date_el = item.find('pubDate')
                date_str = date_el.get_text(strip=True) if date_el else ""
                date = self._parse_date(date_str)
                
                # Skip if outside time window
                if not self._is_recent(date):
                    continue
                
                guid = item.find('guid')
                item_id = self._generate_id(
                    guid.get_text(strip=True) if guid else "", link, title
                )
                
                items.append(ContentItem(
                    id=item_id,
                    title=title,
                    description=description[:1500],
                    date=date_str,
                    link=link,
                    source_name=source_name
                ))
            
            logger.info(f"RSS: {len(items)} items from {source_name} (limit: {self.max_items_per_source}, window: {self.time_window_days}d)")
            
        except requests.exceptions.Timeout:
            logger.error(f"RSS extraction timed out for {url}")
            self.errors.append(f"RSS timeout ({source_name})")
        except requests.exceptions.RequestException as e:
            logger.error(f"RSS extraction failed for {url}: {e}")
            self.errors.append(f"RSS error ({source_name}): {e}")
        except Exception as e:
            logger.error(f"RSS parsing failed for {url}: {e}")
            self.errors.append(f"RSS parse error ({source_name}): {e}")
        
        return items
    
    def _extract_anthropic_news(self, url: str, source_name: str, base_url: str) -> List[ContentItem]:
        """Extract news from Anthropic's news page."""
        items = []
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            seen_urls = set()
            
            for a in soup.find_all('a', href=True):
                if len(items) >= self.max_items_per_source:
                    break
                
                href = a['href']
                
                if not href or href in ['/', '/news', '/news/']:
                    continue
                if href.startswith('#') or href.startswith('mailto:'):
                    continue
                if '/news/' not in href:
                    continue
                
                full_url = urljoin(base_url, href)
                
                if full_url in seen_urls:
                    continue
                seen_urls.add(full_url)
                
                title = a.get_text(separator=' ', strip=True)
                
                if len(title) < 10:
                    parent = a.find_parent(['article', 'div', 'li', 'section'])
                    if parent:
                        heading = parent.find(['h1', 'h2', 'h3', 'h4', 'h5'])
                        if heading:
                            title = heading.get_text(strip=True)
                
                if not title or len(title) < 5:
                    continue
                
                description = ""
                date_str = ""
                parent = a.find_parent(['article', 'div', 'li', 'section'])
                if parent:
                    p = parent.find('p')
                    if p:
                        description = p.get_text(strip=True)
                    time_el = parent.find('time')
                    if time_el:
                        date_str = time_el.get('datetime', '') or time_el.get_text(strip=True)
                
                # Check if within time window
                date = self._parse_date(date_str)
                if not self._is_recent(date):
                    continue
                
                items.append(ContentItem(
                    id=self._generate_id(full_url),
                    title=title[:200],
                    description=description[:500],
                    date=date_str,
                    link=full_url,
                    source_name=source_name
                ))
            
            logger.info(f"Anthropic: {len(items)} items (limit: {self.max_items_per_source}, window: {self.time_window_days}d)")
            
        except requests.exceptions.Timeout:
            logger.error(f"Anthropic extraction timed out")
            self.errors.append(f"Anthropic timeout")
        except requests.exceptions.RequestException as e:
            logger.error(f"Anthropic extraction failed: {e}")
            self.errors.append(f"Anthropic error: {e}")
        except Exception as e:
            logger.error(f"Anthropic parsing failed: {e}")
            self.errors.append(f"Anthropic parse error: {e}")
        
        return items
    
    def _extract_gemini_changelog(self, url: str, source_name: str) -> List[ContentItem]:
        """Extract changelog entries from Gemini docs."""
        items = []
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            date_pattern = re.compile(
                r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}',
                re.IGNORECASE
            )
            
            for heading in soup.find_all(['h2', 'h3']):
                if len(items) >= self.max_items_per_source:
                    break
                
                text = heading.get_text(strip=True)
                
                if not date_pattern.search(text):
                    continue
                
                date = self._parse_date(text)
                if not self._is_recent(date):
                    continue
                
                content_parts = []
                sibling = heading.find_next_sibling()
                while sibling and sibling.name not in ['h2', 'h3']:
                    t = sibling.get_text(strip=True)
                    if t:
                        content_parts.append(t)
                    sibling = sibling.find_next_sibling()
                
                content = ' '.join(content_parts)
                
                items.append(ContentItem(
                    id=self._generate_id(text, content[:200]),
                    title=text,
                    description=content[:1500],
                    date=text,
                    link=url,
                    source_name=source_name
                ))
            
            logger.info(f"Gemini: {len(items)} items (limit: {self.max_items_per_source}, window: {self.time_window_days}d)")
            
        except requests.exceptions.Timeout:
            logger.error(f"Gemini extraction timed out")
            self.errors.append(f"Gemini timeout")
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini extraction failed: {e}")
            self.errors.append(f"Gemini error: {e}")
        except Exception as e:
            logger.error(f"Gemini parsing failed: {e}")
            self.errors.append(f"Gemini parse error: {e}")
        
        return items
    
    def _fetch_article_content(self, url: str, base_url: str) -> Optional[str]:
        """Try to fetch full article content."""
        if not url or not url.startswith('http'):
            return None
        
        try:
            headers = dict(self.session.headers)
            headers['Referer'] = base_url
            
            resp = self.session.get(url, headers=headers, timeout=15)
            if resp.status_code == 403:
                return None
            
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                tag.decompose()
            
            main = soup.find('main') or soup.find('article') or soup.body
            if main:
                return main.get_text(separator='\n', strip=True)[:8000]
            
        except Exception:
            pass
        
        return None
    
    def _fetch_articles_parallel(self, items: List[ContentItem], base_url: str) -> List[ContentItem]:
        """Fetch article content for multiple items in parallel."""
        if not items:
            return items
        
        def fetch_one(item: ContentItem) -> ContentItem:
            if item.link:
                item.full_content = self._fetch_article_content(item.link, base_url)
            return item
        
        logger.info(f"Fetching {len(items)} articles in parallel (max {self.max_fetch_workers} workers)...")
        
        with ThreadPoolExecutor(max_workers=self.max_fetch_workers) as executor:
            results = list(executor.map(fetch_one, items))
        
        fetched_count = sum(1 for item in results if item.full_content)
        logger.info(f"Fetched content for {fetched_count}/{len(items)} articles")
        
        return results
    
    # =========================================================================
    # Handled Releases
    # =========================================================================
    
    def _load_handled(self) -> List[HandledRelease]:
        db = self.storage_dir / "handled_releases.json"
        try:
            if db.exists():
                with open(db) as f:
                    data = json.load(f)
                    return [HandledRelease.from_dict(r) for r in data.get('releases', [])]
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in handled releases: {e}")
        except Exception as e:
            logger.warning(f"Could not load handled releases: {e}")
        return []
    
    def _save_handled(self, releases: List[HandledRelease]):
        db = self.storage_dir / "handled_releases.json"
        try:
            # Keep only last 10 days
            cutoff = datetime.now(timezone.utc) - timedelta(days=10)
            recent = []
            for r in releases:
                try:
                    dt = datetime.fromisoformat(r.detected_at.replace('Z', '+00:00'))
                    if dt >= cutoff:
                        recent.append(r)
                except Exception:
                    recent.append(r)
            
            with open(db, 'w') as f:
                json.dump({
                    'releases': [r.to_dict() for r in recent],
                    'updated': datetime.now(timezone.utc).isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save handled releases: {e}")
    
    def _is_handled(self, item: ContentItem, handled: List[HandledRelease]) -> bool:
        for h in handled:
            if h.id == item.id or (h.link and item.link and h.link == item.link):
                return True
        return False
    
    # =========================================================================
    # LLM Analysis
    # =========================================================================
    
    def _analyze_item(self, item: ContentItem, 
                      watchlist_items: List[WatchlistItem]) -> AnalysisResult:
        """Analyze a single item to determine if it's a relevant release."""
        
        # Build article content block
        article_content = f"""<article>
<title>{item.title}</title>
<source>{item.source_name}</source>
<date>{item.date or 'Unknown'}</date>
<link>{item.link}</link>
<description>{item.description or ''}</description>
"""
        if item.full_content:
            article_content += f"<full_content>\n{item.full_content[:6000]}\n</full_content>\n"
        article_content += "</article>"
        
        # Query 1: Summarize
        prompt1 = f"""<task>
Summarize what this article is announcing or discussing. Be factual and specific.
</task>

{article_content}

<instructions>
In 2-3 sentences, state:
1. What type of content this is (announcement, case study, blog post, changelog, policy update, etc.)
2. The main topic or news
3. Any specific product names, model names, or version numbers mentioned

Be precise. Do not infer or assume - only state what is explicitly in the article.
</instructions>"""

        response1, error1 = self.llm.query(prompt1)
        
        if error1:
            logger.error(f"Query 1 failed for {item.title[:40]}: {error1}")
            return AnalysisResult(item=item, is_relevant=False, summary="", issue_body="")
        
        logger.info(f"Summary: {response1[:100]}...")
        
        # Get known models for context
        known_models = self._fetch_known_models()
        provider_key = self._get_provider_key(item.source_name)
        provider_models = known_models.get(provider_key, [])
        
        # Build models context
        models_list = ', '.join(provider_models[:40]) if provider_models else 'Unable to fetch - use fallback judgment'
        
        # Format watchlist and mistakes for prompt
        watchlist_context = self._format_watchlist_for_prompt(
            self._watchlist_content or "", 
            item.source_name
        )
        mistakes_context = self._format_mistakes_for_prompt(
            self._mistakes_content or ""
        )
        
        # Check token budget
        base_prompt_size = len(response1) + len(models_list) + 2000  # estimate
        strategy = self._calculate_token_budget(
            article_content, str(base_prompt_size), watchlist_context, mistakes_context
        )
        
        # Build extra context based on strategy
        extra_context = ""
        if strategy == 'combined':
            if watchlist_context:
                extra_context += f"\n{watchlist_context}\n"
            if mistakes_context:
                extra_context += f"\n{mistakes_context}\n"
        
        # Query 2: Classification with strict exclusion logic
        prompt2 = f"""<role>
You are a strict classifier for an API developer notification system.
Your job is to filter OUT irrelevant content. When uncertain, reject.
False positives waste developer time. False negatives can be caught next cycle.
</role>

<article_summary>
Title: {item.title}
Source: {item.source_name}
Summary: {response1}
</article_summary>

<critical_exclusions>
IMMEDIATELY REJECT if ANY of these apply:

1. CASE STUDY: Article title contains "How [Company]...", "[Company]'s lessons...", 
   "Building with...", or describes a customer's experience using AI
   → These describe existing capabilities, NOT new releases

2. PARTNERSHIP/BUSINESS: Announces partnerships, investments, data centers, 
   hiring, acquisitions, or corporate news
   → Not relevant to API developers

3. CONSUMER PRODUCT: About ChatGPT, Claude App, Gemini App, AI Studio, 
   or subscription tiers (Plus, Pro, Team, Enterprise plans for consumers)
   → We only care about API/developer platform changes

4. POLICY/RESEARCH: Safety policies, research papers, responsible AI, 
   governance frameworks, or thought leadership
   → Not actionable for developers

5. VERTICAL SOLUTIONS: Healthcare solutions, enterprise offerings, 
   industry-specific products WITHOUT new underlying API capabilities
   → Marketing, not technical releases

6. EXISTING MODELS IN NEW CONTEXT: Article mentions existing models 
   being used in a new way, new region, or new integration
   → Model must be genuinely NEW, not existing model in new context
</critical_exclusions>

<known_models>
Models that ALREADY EXIST for {item.source_name}:
{models_list}

IMPORTANT: If an article MENTIONS a model not in this list, that does NOT 
automatically mean it's a new release. Case studies and blog posts often 
contain typos or refer to internal model versions. Only flag as new if 
the article is an OFFICIAL ANNOUNCEMENT of a new model release.
</known_models>
{extra_context}
<relevant_criteria>
ONLY mark as RELEVANT if the article is an OFFICIAL ANNOUNCEMENT containing:

- New model release: A genuinely new model announced by {item.source_name} 
  (not mentioned in a case study or customer story)
- New API capability: New endpoints, parameters, features, or rate limits
- Pricing changes: Changes to API pricing (not subscription tiers)
- SDK/library release: New developer tools or SDK versions
- Deprecation notice: Models or API features being deprecated
- Breaking changes: Changes requiring developer action
- Availability changes: Models moving from preview to GA (especially in new regions)
</relevant_criteria>

<output_format>
Think step by step:

EXCLUSION_CHECK: [Check each exclusion 1-6. Does any apply? State which one or "None"]
ARTICLE_TYPE: [case_study | announcement | changelog | blog_post | policy | other]
RELEVANT: [YES or NO]
REASON: [One specific sentence explaining your decision]
RESOLVES_WATCHLIST: #[issue_number] (only if applicable, otherwise omit this line)
</output_format>"""

        response2, error2 = self.llm.query(prompt2)
        
        if error2:
            logger.error(f"Query 2 failed for {item.title[:40]}: {error2}")
            return AnalysisResult(item=item, is_relevant=False, summary=response1, issue_body="")
        
        # Handle split strategy - additional queries for watchlist and mistakes
        if strategy == 'split':
            if watchlist_context:
                watchlist_prompt = f"""<context>
Article: {item.title}
Source: {item.source_name}
Summary: {response1}
</context>

{watchlist_context}

<task>
Does this article resolve any watchlist item?
</task>

<output>
RESOLVES_WATCHLIST: #[issue_number] or NONE
</output>"""
                
                wl_response, wl_error = self.llm.query(watchlist_prompt)
                if not wl_error and wl_response:
                    response2 += "\n" + wl_response
            
            if mistakes_context:
                mistakes_prompt = f"""<context>
Article: {item.title}
Source: {item.source_name}  
Summary: {response1}
Classification so far: {response2[:500]}
</context>

{mistakes_context}

<task>
Would marking this article as RELEVANT repeat any past mistake listed above?
</task>

<output>
REPEAT_MISTAKE: YES or NO
WHICH_MISTAKE: [If yes, which one]
</output>"""
                
                m_response, m_error = self.llm.query(mistakes_prompt)
                if not m_error and m_response and 'YES' in m_response.upper():
                    logger.info(f"Rejected by mistakes check: {item.title[:40]}")
                    return AnalysisResult(
                        item=item, 
                        is_relevant=False, 
                        summary=response1, 
                        issue_body=""
                    )
        
        # Parse response with improved flexibility
        is_relevant = False
        reason = response2
        resolves_watchlist = None
        article_type = "unknown"
        
        # Check for exclusion applied
        exclusion_match = re.search(r'EXCLUSION_CHECK:\s*(.+?)(?=\n|ARTICLE_TYPE)', response2, re.IGNORECASE | re.DOTALL)
        if exclusion_match:
            exclusion_text = exclusion_match.group(1).strip().lower()
            # If any exclusion was found (not "none"), reject
            if exclusion_text and 'none' not in exclusion_text and len(exclusion_text) > 5:
                logger.info(f"Excluded by rule: {item.title[:40]}... - {exclusion_text[:50]}")
                return AnalysisResult(
                    item=item,
                    is_relevant=False,
                    summary=response1,
                    issue_body=""
                )
        
        # Extract article type
        type_match = re.search(r'ARTICLE_TYPE:\s*(\w+)', response2, re.IGNORECASE)
        if type_match:
            article_type = type_match.group(1).lower()
        
        # Case studies and blog posts are automatically not relevant
        if article_type in ['case_study', 'blog_post', 'policy']:
            logger.info(f"Rejected by type ({article_type}): {item.title[:40]}")
            return AnalysisResult(
                item=item,
                is_relevant=False,
                summary=response1,
                issue_body=""
            )
        
        # Parse RELEVANT field with flexible regex
        # Handles: "RELEVANT: YES", "RELEVANT:YES", "RELEVANT: YES - because...", "RELEVANT: yes"
        relevant_match = re.search(r'RELEVANT:\s*(YES|NO)\b', response2, re.IGNORECASE)
        if relevant_match:
            is_relevant = relevant_match.group(1).upper() == 'YES'
        else:
            # RELEVANT field not found - check for unstructured response indicating relevance
            # This is a warning case - the LLM didn't follow instructions
            # Check if response starts with YES/NO without proper format
            stripped_response = response2.strip()
            if re.match(r'^YES\b', stripped_response, re.IGNORECASE):
                logger.warning(
                    f"LLM response missing 'RELEVANT:' prefix but starts with YES - "
                    f"treating as relevant with warning: {item.title[:40]}"
                )
                if os.getenv('GITHUB_ACTIONS'):
                    print(f"::warning title=LLM Format Issue::Response missing RELEVANT: prefix for: {item.title[:60]}")
                is_relevant = True
            elif re.match(r'^NO\b', stripped_response, re.IGNORECASE):
                logger.warning(
                    f"LLM response missing 'RELEVANT:' prefix but starts with NO - "
                    f"treating as not relevant: {item.title[:40]}"
                )
                is_relevant = False
            else:
                # Can't determine relevance from response format
                logger.warning(
                    f"LLM response missing 'RELEVANT:' field entirely - "
                    f"defaulting to NOT relevant for safety: {item.title[:40]}"
                )
                if os.getenv('GITHUB_ACTIONS'):
                    print(f"::warning title=LLM Format Issue::Cannot parse RELEVANT field for: {item.title[:60]}")
                is_relevant = False
        
        # Validate response format when marking as relevant
        if is_relevant:
            has_exclusion = bool(re.search(r'EXCLUSION_CHECK:', response2, re.IGNORECASE))
            has_type = bool(re.search(r'ARTICLE_TYPE:', response2, re.IGNORECASE))
            has_relevant = bool(re.search(r'RELEVANT:', response2, re.IGNORECASE))
            has_reason = bool(re.search(r'REASON:', response2, re.IGNORECASE))
            
            missing_fields = []
            if not has_exclusion:
                missing_fields.append('EXCLUSION_CHECK')
            if not has_type:
                missing_fields.append('ARTICLE_TYPE')
            if not has_relevant:
                missing_fields.append('RELEVANT')
            if not has_reason:
                missing_fields.append('REASON')
            
            if missing_fields:
                logger.warning(
                    f"LLM response missing expected fields ({', '.join(missing_fields)}): {item.title[:40]}"
                )
                if os.getenv('GITHUB_ACTIONS'):
                    print(f"::warning title=Incomplete LLM Response::Missing {', '.join(missing_fields)} for: {item.title[:60]}")
        
        # Extract reason
        if 'REASON:' in response2.upper():
            idx = response2.upper().index('REASON:')
            end_idx = response2.upper().find('RESOLVES_WATCHLIST:', idx)
            if end_idx == -1:
                end_idx = len(response2)
            reason = response2[idx + 7:end_idx].strip()
        
        # Check for watchlist resolution
        watchlist_match = re.search(r'RESOLVES_WATCHLIST:\s*#?(\d+)', response2)
        if watchlist_match:
            resolves_watchlist = int(watchlist_match.group(1))
        
        if not is_relevant:
            logger.info(f"Not relevant: {item.title[:40]}... - {reason[:50]}")
            return AnalysisResult(
                item=item, 
                is_relevant=False, 
                summary=response1, 
                issue_body=""
            )
        
        logger.info(f"✓ RELEVANT ({article_type}): {item.title[:40]}... - {reason[:50]}")
        if resolves_watchlist:
            logger.info(f"  Resolves watchlist item #{resolves_watchlist}")
        
        # Query 3: Generate issue body
        prompt3 = f"""<task>
Create a GitHub issue body for this AI API release notification.
</task>

<article>
Title: {item.title}
Source: {item.source_name}
Link: {item.link}
Summary: {response1}
</article>

<content>
{item.full_content[:5000] if item.full_content else item.description}
</content>

<format>
Write a clear, professional Markdown summary including:

## What's New
[Specific details: model names, API endpoints, version numbers, capabilities]

## Key Details  
[Technical specifics relevant to API developers: pricing, rate limits, parameters]

## Developer Impact
[What developers need to know or do]

## Platform Availability
[Note if this mentions specific platforms: direct API, AWS Bedrock, Google Cloud Vertex AI, Azure]
</format>

<rules>
- Be factual and specific
- Include exact model names and version numbers
- Note any action items or migration requirements
- Keep it concise but complete
</rules>"""

        response3, error3 = self.llm.query(prompt3)
        
        if error3:
            logger.error(f"Query 3 failed for {item.title[:40]}: {error3}")
            response3 = f"""## {item.title}

**Summary:** {response1}

**Source:** [{item.source_name}]({item.link})

{item.description if item.description else 'See link for full details.'}"""
        
        return AnalysisResult(
            item=item,
            is_relevant=True,
            summary=response1,
            issue_body=response3,
            resolves_watchlist=resolves_watchlist
        )
    
    def _analyze_items_parallel(self, items: List[ContentItem], 
                                handled: List[HandledRelease], 
                                watchlist_items: List[WatchlistItem]) -> List[AnalysisResult]:
        """Analyze multiple items in parallel."""
        # Filter out already handled items first
        to_analyze = [item for item in items if not self._is_handled(item, handled)]
        
        if not to_analyze:
            return []
        
        logger.info(f"Analyzing {len(to_analyze)} items in parallel (max {self.max_llm_workers} workers)...")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_llm_workers) as executor:
            future_to_item = {
                executor.submit(self._analyze_item, item, watchlist_items): item 
                for item in to_analyze
            }
            
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    if result.is_relevant:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Analysis failed for {item.title[:40]}: {e}")
        
        logger.info(f"Found {len(results)} relevant items")
        return results
    
    # =========================================================================
    # GitHub Issue Management
    # =========================================================================
    
    def _create_issue(self, site_info: dict, result: AnalysisResult) -> Optional[int]:
        """Create a GitHub issue for a relevant release."""
        if not self.github_token or not self.github_repo:
            logger.error("GitHub not configured")
            return None
        
        try:
            item = result.item
            title = f"{site_info['emoji']} {item.source_name}: {item.title[:70]}"
            
            watchlist_note = ""
            if result.resolves_watchlist:
                watchlist_note = f"\n\n> 🎯 **Resolves watchlist item #{result.resolves_watchlist}**\n"
            
            full_body = f"""## New Release Detected

**Source:** [{item.source_name}]({item.link})  
**Detected:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
{watchlist_note}
---

{result.issue_body}

---

### Quick Summary
{result.summary}

### Actions
- [ ] Review the release
- [ ] Update affected code/docs if needed

*Auto-generated by AI Website Monitor*
"""
            
            labels = ['type:release', site_info['label'], 'auto-close-7d']
            
            resp = self._github_api('POST', '/issues', json={
                'title': title,
                'body': full_body,
                'labels': labels
            })
            
            if resp:
                issue = resp.json()
                logger.info(f"✅ Created issue #{issue['number']}")
                return issue['number']
            else:
                self.errors.append(f"Issue creation failed for: {item.title[:50]}")
            
        except Exception as e:
            logger.error(f"Issue creation failed: {e}")
            self.errors.append(f"Issue creation: {e}")
        
        return None
    
    def _close_old_issues(self):
        """Auto-close issues older than 7 days with auto-close-7d label."""
        resp = self._github_api('GET', '/issues', params={
            'labels': 'auto-close-7d',
            'state': 'open',
            'per_page': 100
        })
        
        if not resp:
            return
        
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        for issue in resp.json():
            try:
                created = datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00'))
                if created < cutoff:
                    result = self._github_api('PATCH', f'/issues/{issue["number"]}', json={
                        'state': 'closed'
                    })
                    if result:
                        logger.info(f"Auto-closed issue #{issue['number']}")
                    else:
                        logger.warning(f"Failed to auto-close issue #{issue['number']}")
            except Exception as e:
                logger.warning(f"Failed to close issue #{issue.get('number')}: {e}")
    
    def _create_error_issue(self):
        """Create an issue summarizing errors from this run."""
        if not self.errors or not self.github_token or not self.github_repo:
            return
        
        try:
            resp = self._github_api('POST', '/issues', json={
                'title': '⚠️ Monitor Errors',
                'body': f"**Errors from run at {datetime.now(timezone.utc).isoformat()}:**\n\n" + 
                        '\n'.join(f"- {e}" for e in self.errors),
                'labels': ['type:bug']
            })
            if resp:
                logger.info(f"Created error issue #{resp.json()['number']}")
            else:
                logger.error("Failed to create error issue")
        except Exception as e:
            logger.error(f"Error issue creation failed: {e}")
    
    # =========================================================================
    # Main Processing
    # =========================================================================
    
    def _process_site(self, site_key: str, site_info: dict, 
                      handled: List[HandledRelease], 
                      watchlist_items: List[WatchlistItem]) -> List[AnalysisResult]:
        """Process a single site: extract, fetch articles, analyze."""
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing: {site_info['name']}")
        logger.info(f"{'='*50}")
        
        # Extract items (respects 7 day / 10 item limits)
        if site_info['type'] == 'rss':
            items = self._extract_rss(site_info['url'], site_info['name'])
        elif site_info['type'] == 'html_news':
            items = self._extract_anthropic_news(
                site_info['url'], site_info['name'], site_info['base_url']
            )
        elif site_info['type'] == 'html_changelog':
            items = self._extract_gemini_changelog(site_info['url'], site_info['name'])
        else:
            logger.warning(f"Unknown type: {site_info['type']}")
            return []
        
        if not items:
            logger.info(f"No items from {site_info['name']}")
            return []
        
        # Filter out already handled
        new_items = [item for item in items if not self._is_handled(item, handled)]
        logger.info(f"Found {len(items)} items, {len(new_items)} are new")
        
        if not new_items:
            return []
        
        # Fetch article content in parallel
        new_items = self._fetch_articles_parallel(new_items, site_info['base_url'])
        
        # Analyze items in parallel
        results = self._analyze_items_parallel(new_items, handled, watchlist_items)
        
        return results
    
    def check_all_websites(self):
        """Main entry point: check all websites for new releases."""
        logger.info("=" * 60)
        logger.info(f"Monitor started: {datetime.now(timezone.utc)}")
        logger.info(f"Limits: {self.time_window_days} days OR {self.max_items_per_source} items per source")
        logger.info("=" * 60)
        
        # Ensure labels exist
        self._ensure_labels_exist()
        
        # Process false positives from previous runs
        self._process_false_positives()
        
        # Fetch and process watchlist
        watchlist_issues = self._fetch_open_watchlist_issues()
        watchlist_items = []
        for issue in watchlist_issues:
            item = self._parse_watchlist_issue(issue)
            if item:
                watchlist_items.append(item)
        
        logger.info(f"Loaded {len(watchlist_items)} watchlist items")
        
        # Close expired watchlist items
        self._close_expired_watchlist(watchlist_items)
        
        # Regenerate watchlist.md from remaining open items
        watchlist_issues = self._fetch_open_watchlist_issues()
        watchlist_items = []
        for issue in watchlist_issues:
            item = self._parse_watchlist_issue(issue)
            if item:
                watchlist_items.append(item)
        
        watchlist_md = self._generate_watchlist_md(watchlist_items)
        watchlist_path = self.storage_dir / "watchlist.md"
        
        existing_watchlist = ""
        if watchlist_path.exists():
            existing_watchlist = watchlist_path.read_text()
        
        existing_no_ts = re.sub(r'Last updated:.*', '', existing_watchlist)
        new_no_ts = re.sub(r'Last updated:.*', '', watchlist_md)
        
        if existing_no_ts.strip() != new_no_ts.strip():
            watchlist_path.write_text(watchlist_md)
            logger.info("Updated watchlist.md")
        
        # Load context files for analysis
        self._watchlist_content, self._mistakes_content = self._load_context_files()
        
        # Pre-fetch known models
        logger.info("Fetching current model lists from provider documentation...")
        known_models = self._fetch_known_models()
        total_models = sum(len(v) for v in known_models.values())
        logger.info(f"Loaded {total_models} known models across all providers")
        
        # Close old issues
        self._close_old_issues()
        
        # Load handled releases
        handled = self._load_handled()
        logger.info(f"Loaded {len(handled)} previously handled releases")
        
        # Process all sites
        all_results: List[Tuple[str, dict, AnalysisResult]] = []
        
        with ThreadPoolExecutor(max_workers=len(self.websites)) as executor:
            future_to_site = {
                executor.submit(
                    self._process_site, key, info, handled, watchlist_items
                ): (key, info)
                for key, info in self.websites.items()
            }
            
            for future in as_completed(future_to_site):
                site_key, site_info = future_to_site[future]
                try:
                    results = future.result()
                    for result in results:
                        all_results.append((site_key, site_info, result))
                except Exception as e:
                    logger.error(f"Site processing failed for {site_info['name']}: {e}")
                    self.errors.append(f"{site_info['name']}: {e}")
        
        # Create issues
        issues_created = 0
        for site_key, site_info, result in all_results:
            issue_num = self._create_issue(site_info, result)
            if issue_num:
                issues_created += 1
                handled.append(HandledRelease(
                    id=result.item.id,
                    title=result.item.title,
                    link=result.item.link,
                    issue_number=issue_num,
                    detected_at=datetime.now(timezone.utc).isoformat()
                ))
                
                if result.resolves_watchlist:
                    self._resolve_watchlist_item(result.resolves_watchlist, issue_num)
        
        self._save_handled(handled)
        
        if self.errors:
            logger.warning(f"⚠️ {len(self.errors)} errors occurred")
            self._create_error_issue()
        
        # GitHub Actions annotations
        if os.getenv('GITHUB_ACTIONS'):
            if issues_created:
                print(f"::notice::Created {issues_created} issue(s)")
                for site_key, site_info, result in all_results:
                    print(f"::notice title={site_info['name']}::{result.item.title[:80]}")
            else:
                print("::notice::No new releases detected")
            
            if self.errors:
                for error in self.errors:
                    print(f"::warning::{error}")
        
        logger.info("=" * 60)
        if issues_created:
            logger.info(f"✅ Created {issues_created} issue(s)")
        else:
            logger.info("✅ No new releases detected")
        logger.info("=" * 60)


def main():
    try:
        monitor = WebsiteMonitor()
        monitor.check_all_websites()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Fatal: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
