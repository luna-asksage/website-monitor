#!/usr/bin/env python3
"""
AI-Powered Website Change Monitor
Creates GitHub Issues when product/API changes are detected.
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup
from dateutil import parser as date_parser


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ContentItem:
    """Represents a single content item."""
    id: str
    title: str
    content: str
    date: Optional[datetime]
    link: str
    source: str
    source_name: str = ""
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['date'] = self.date.isoformat() if self.date else None
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ContentItem':
        if data.get('date'):
            data['date'] = datetime.fromisoformat(data['date'])
        return cls(**data)


@dataclass
class HandledRelease:
    """Represents a release we've already created an issue for."""
    id: str
    title: str
    url: str
    detected_at: datetime
    issue_number: int
    source: str
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d['detected_at'] = self.detected_at.isoformat()
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> 'HandledRelease':
        data['detected_at'] = datetime.fromisoformat(data['detected_at'])
        return cls(**data)


class WebContentFetcher:
    """Handles web content fetching with anti-bot measures."""
    
    def __init__(self):
        self.session = requests.Session()
        # Rotate through different user agents
        self.user_agents = [
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        ]
        self.current_ua_index = 0
    
    def _get_headers(self, referer: Optional[str] = None) -> Dict[str, str]:
        """Get headers that mimic a real browser."""
        ua = self.user_agents[self.current_ua_index % len(self.user_agents)]
        self.current_ua_index += 1
        
        headers = {
            'User-Agent': ua,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
            'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"macOS"',
        }
        
        if referer:
            headers['Referer'] = referer
            headers['Sec-Fetch-Site'] = 'same-origin'
        
        return headers
    
    def fetch(self, url: str, referer: Optional[str] = None) -> Optional[requests.Response]:
        """Fetch URL with retry logic and anti-bot measures."""
        for attempt in range(3):
            try:
                headers = self._get_headers(referer)
                response = self.session.get(
                    url, 
                    headers=headers, 
                    timeout=30, 
                    allow_redirects=True
                )
                
                if response.status_code == 403:
                    logger.warning(f"403 on attempt {attempt + 1} for {url}, retrying...")
                    continue
                
                response.raise_for_status()
                response.encoding = response.apparent_encoding or 'utf-8'
                return response
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Fetch attempt {attempt + 1} failed for {url}: {e}")
                if attempt == 2:
                    return None
        
        return None
    
    def fetch_rss(self, url: str) -> Optional[str]:
        """Fetch RSS feed content."""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; AIMonitor/1.0)'}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.content.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Failed to fetch RSS {url}: {e}")
            return None


class WebsiteMonitor:
    """Monitors websites for changes and creates GitHub Issues."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the monitor with configuration."""
        self.config = self._load_config(config_path)
        self._override_with_env_vars()
        self.storage_dir = Path(self.config.get("storage_dir", "storage"))
        self.storage_dir.mkdir(exist_ok=True)
        
        self.fetcher = WebContentFetcher()
        
        # GitHub configuration
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.github_repo = os.getenv('GITHUB_REPOSITORY')
        
        # Track errors for reporting
        self.errors: List[str] = []
        
        # Time window for items (7 days)
        self.time_window = timedelta(days=7)
        self.max_items_per_source = 10
        
        # Websites to monitor
        self.websites = {
            "openai_news": {
                "url": "https://openai.com/news/rss.xml",
                "name": "OpenAI News",
                "type": "rss",
                "emoji": "🔵",
                "label": "openai",
                "base_url": "https://openai.com"
            },
            "anthropic_rss": {
                "url": "https://www.anthropic.com/rss.xml",
                "name": "Anthropic News",
                "type": "rss",
                "emoji": "🟠",
                "label": "anthropic",
                "base_url": "https://www.anthropic.com"
            },
            "gemini_changelog": {
                "url": "https://ai.google.dev/gemini-api/docs/changelog",
                "name": "Gemini API Changelog",
                "type": "html_changelog",
                "selectors": [".devsite-article-body", "article", "main"],
                "emoji": "🟡",
                "label": "gemini",
                "base_url": "https://ai.google.dev"
            },
            "google_ai_blog": {
                "url": "https://blog.google/technology/ai/rss/",
                "name": "Google AI Blog",
                "type": "rss",
                "emoji": "🟡",
                "label": "google-ai",
                "base_url": "https://blog.google"
            },
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        default_config = {
            "asksage_api": {
                "url": "https://api.asksage.ai/server/query",
                "token": "",
                "model": "google-gemini-2.5-pro",
                "temperature": 0.1
            },
            "storage_dir": "storage",
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
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Config load issue: {e}, using defaults")
            return default_config
    
    def _override_with_env_vars(self):
        """Override config with environment variables."""
        if os.getenv('ASKSAGE_API_TOKEN'):
            self.config['asksage_api']['token'] = os.getenv('ASKSAGE_API_TOKEN')
            logger.info("Using API token from environment variable")
    
    def _generate_id(self, *args) -> str:
        """Generate a unique ID from input strings."""
        combined = '|'.join(str(arg).strip() for arg in args if arg)
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats into datetime."""
        if not date_str:
            return None
        try:
            dt = date_parser.parse(date_str, fuzzy=True)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            return None
    
    def _is_within_time_window(self, dt: Optional[datetime]) -> bool:
        """Check if datetime is within our monitoring window."""
        if not dt:
            return True  # Include items without dates
        now = datetime.now(timezone.utc)
        cutoff = now - self.time_window
        return dt >= cutoff
    
    def _clean_text(self, text: str) -> str:
        """Clean text content."""
        if not text:
            return ""
        import re
        text = re.sub(r'[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000\uFEFF]', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def _clean_url(self, url: str) -> str:
        """Clean URL artifacts."""
        if not url:
            return ""
        url = url.strip()
        while url.endswith(':'):
            url = url[:-1]
        return url
    
    # =========================================================================
    # Content Extraction
    # =========================================================================
    
    def _extract_rss_items(self, url: str, source_name: str, base_url: str) -> List[ContentItem]:
        """Extract items from RSS feed."""
        items = []
        content = self.fetcher.fetch_rss(url)
        if not content:
            self.errors.append(f"Failed to fetch RSS: {url}")
            return items
        
        try:
            soup = BeautifulSoup(content, 'xml')
            
            for item in soup.find_all('item'):
                title = item.find('title')
                title = title.get_text(strip=True) if title else ""
                
                link = item.find('link')
                link = self._clean_url(link.get_text(strip=True) if link else "")
                
                # Fix relative links
                if link and not link.startswith('http'):
                    link = urljoin(base_url, link)
                
                guid = item.find('guid')
                guid = guid.get_text(strip=True) if guid else ""
                
                pub_date = item.find('pubDate')
                date = self._parse_date(pub_date.get_text(strip=True) if pub_date else "")
                
                desc = item.find('description')
                description = ""
                if desc:
                    desc_soup = BeautifulSoup(desc.get_text(), 'html.parser')
                    description = desc_soup.get_text(separator=' ', strip=True)
                
                item_id = self._generate_id(guid or link or title)
                
                items.append(ContentItem(
                    id=item_id,
                    title=title,
                    content=self._clean_text(description),
                    date=date,
                    link=link,
                    source=url,
                    source_name=source_name
                ))
            
            logger.info(f"Extracted {len(items)} items from {source_name} RSS")
            
        except Exception as e:
            logger.error(f"Failed to parse RSS {url}: {e}")
            self.errors.append(f"RSS parse error for {url}: {str(e)}")
        
        return items
    
    def _extract_changelog_items(self, url: str, source_name: str, 
                                  selectors: List[str], base_url: str) -> List[ContentItem]:
        """Extract changelog entries from HTML."""
        items = []
        response = self.fetcher.fetch(url)
        if not response:
            self.errors.append(f"Failed to fetch changelog: {url}")
            return items
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Check for JS-rendered page
            body_text = soup.get_text(strip=True)
            if len(body_text) < 200 or body_text.count('Loading') > 3:
                logger.warning(f"Page appears JS-rendered: {url}")
                self.errors.append(f"JS-rendered page: {url}")
                return items
            
            # Remove clutter
            for el in soup(["script", "style", "nav", "footer", "header", "aside"]):
                el.decompose()
            
            # Find main content
            main = None
            for sel in selectors:
                main = soup.select_one(sel)
                if main:
                    break
            if not main:
                main = soup.body or soup
            
            # Find date-based sections
            import re
            date_pattern = re.compile(
                r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|'
                r'\d{4}-\d{2}-\d{2}',
                re.IGNORECASE
            )
            
            for heading in main.find_all(['h2', 'h3']):
                heading_text = heading.get_text(strip=True)
                date_match = date_pattern.search(heading_text)
                
                if not date_match:
                    continue
                
                # Get content until next heading
                content_parts = []
                sibling = heading.find_next_sibling()
                while sibling and sibling.name not in ['h2', 'h3']:
                    text = sibling.get_text(strip=True)
                    if text:
                        content_parts.append(text)
                    sibling = sibling.find_next_sibling()
                
                content = ' '.join(content_parts)
                date = self._parse_date(date_match.group())
                item_id = self._generate_id(heading_text, content[:200])
                
                items.append(ContentItem(
                    id=item_id,
                    title=heading_text,
                    content=self._clean_text(content[:3000]),
                    date=date,
                    link=url,
                    source=url,
                    source_name=source_name
                ))
            
            logger.info(f"Extracted {len(items)} changelog entries from {source_name}")
            
        except Exception as e:
            logger.error(f"Failed to parse changelog {url}: {e}")
            self.errors.append(f"Changelog parse error: {str(e)}")
        
        return items
    
    def _fetch_article_content(self, url: str, base_url: str) -> Optional[str]:
        """Fetch full article content for a linked page."""
        if not url or not url.startswith('http'):
            return None
        
        response = self.fetcher.fetch(url, referer=base_url)
        if not response:
            return None
        
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for el in soup(["script", "style", "nav", "footer", "header", "aside", "iframe"]):
                el.decompose()
            
            # Try to find main content
            main = None
            for sel in ['article', 'main', '.post-content', '.article-content', '[role="main"]']:
                main = soup.select_one(sel)
                if main:
                    break
            
            if not main:
                main = soup.body or soup
            
            content = self._clean_text(main.get_text(separator='\n', strip=True))
            return content[:15000]
            
        except Exception as e:
            logger.warning(f"Failed to parse article {url}: {e}")
            return None
    
    # =========================================================================
    # Handled Releases Database
    # =========================================================================
    
    def _load_handled_releases(self) -> List[HandledRelease]:
        """Load the list of releases we've already created issues for."""
        db_file = self.storage_dir / "handled_releases.json"
        try:
            if db_file.exists():
                with open(db_file, 'r') as f:
                    data = json.load(f)
                    return [HandledRelease.from_dict(r) for r in data.get('releases', [])]
        except Exception as e:
            logger.warning(f"Error loading handled releases: {e}")
        return []
    
    def _save_handled_releases(self, releases: List[HandledRelease]):
        """Save the handled releases database."""
        db_file = self.storage_dir / "handled_releases.json"
        try:
            # Keep only releases from the last 30 days
            cutoff = datetime.now(timezone.utc) - timedelta(days=30)
            recent = [r for r in releases if r.detected_at >= cutoff]
            
            with open(db_file, 'w') as f:
                json.dump({
                    'releases': [r.to_dict() for r in recent],
                    'updated': datetime.now(timezone.utc).isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving handled releases: {e}")
    
    def _is_already_handled(self, item: ContentItem, handled: List[HandledRelease]) -> bool:
        """Check if we've already created an issue for this release."""
        for release in handled:
            # Match by ID
            if release.id == item.id:
                return True
            # Match by URL
            if release.url and item.link and release.url == item.link:
                return True
            # Match by similar title (fuzzy)
            if release.title and item.title:
                if release.title.lower().strip() == item.title.lower().strip():
                    return True
        return False
    
    # =========================================================================
    # LLM Analysis (Multi-step)
    # =========================================================================
    
    def _call_llm(self, system_prompt: str, user_message: str) -> Optional[str]:
        """Make a single LLM API call."""
        try:
            api_url = self.config['asksage_api']['url']
            api_token = self.config['asksage_api']['token']
            
            if not api_token:
                logger.error("API token not configured")
                return None
            
            files = {
                'model': (None, self.config['asksage_api']['model']),
                'temperature': (None, str(self.config['asksage_api']['temperature'])),
                'message': (None, json.dumps([{"user": "me", "message": user_message}])),
                'system_prompt': (None, system_prompt),
                'dataset': (None, 'none'),
                'usage': (None, 'True')
            }
            
            headers = {'x-access-tokens': api_token}
            
            response = requests.post(api_url, headers=headers, files=files, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            if isinstance(result, dict):
                return result.get('response', result.get('message', '')).strip()
            return str(result).strip()
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None
    
    def _check_relevance(self, item: ContentItem, article_content: Optional[str]) -> Tuple[bool, str]:
        """
        Step 1: Check if an item is a relevant product/API release.
        Returns (is_relevant, reason).
        """
        system_prompt = """You are an AI release detector. Determine if this announcement is about a PRODUCT or API RELEASE that developers need to know about.

RELEVANT (answer YES):
- New AI model releases (GPT-5, Claude 4, Gemini 2.5, etc.)
- Model version updates or improvements
- New API features, endpoints, or capabilities
- Pricing changes for models/APIs
- New SDKs or developer tools
- Model deprecations or breaking changes
- Significant capability updates (context length, speed, multimodal)

NOT RELEVANT (answer NO):
- Research papers or academic announcements
- Company news (funding, hiring, partnerships, office news)
- Policy, safety, or ethics announcements
- General thought leadership or opinion pieces
- Event or conference announcements
- Marketing content without new features

Answer format:
YES: [one sentence explaining what release/update this is]
or
NO: [one sentence explaining why this isn't a product release]"""

        content_preview = item.content[:1000] if item.content else ""
        article_preview = article_content[:3000] if article_content else ""
        
        user_message = f"""Analyze this announcement:

TITLE: {item.title}
DATE: {item.date.strftime('%Y-%m-%d') if item.date else 'Unknown'}
SOURCE: {item.source_name}
URL: {item.link}

PREVIEW:
{content_preview}

FULL ARTICLE CONTENT:
{article_preview}

Is this a relevant product/API release announcement?"""

        response = self._call_llm(system_prompt, user_message)
        
        if not response:
            return False, "LLM call failed"
        
        response_upper = response.upper()
        if response_upper.startswith('YES'):
            reason = response[3:].strip().lstrip(':').strip()
            return True, reason
        else:
            reason = response[2:].strip().lstrip(':').strip() if response_upper.startswith('NO') else response
            return False, reason
    
    def _generate_issue_summary(self, item: ContentItem, article_content: Optional[str], 
                                 relevance_reason: str) -> str:
        """
        Step 2: Generate a detailed issue summary for a relevant release.
        """
        system_prompt = """You are a technical writer creating a GitHub issue about an AI product/API release.

Create a clear, actionable summary that developers can use. Include:
1. What was released/updated
2. Key features or changes
3. Technical details (model names, versions, capabilities, pricing if mentioned)
4. Any breaking changes or deprecations
5. Links to documentation if mentioned

Format as GitHub-flavored Markdown."""

        content_preview = item.content[:2000] if item.content else ""
        article_preview = article_content[:8000] if article_content else ""
        
        user_message = f"""Create a GitHub issue summary for this release:

TITLE: {item.title}
DATE: {item.date.strftime('%Y-%m-%d') if item.date else 'Unknown'}
SOURCE: {item.source_name}
URL: {item.link}
DETECTED AS: {relevance_reason}

PREVIEW:
{content_preview}

FULL ARTICLE:
{article_preview}

Generate a detailed but concise summary for developers."""

        response = self._call_llm(system_prompt, user_message)
        
        if not response or len(response) < 50:
            # Fallback summary
            return f"""### {item.title}

**Detected:** {relevance_reason}

**Date:** {item.date.strftime('%Y-%m-%d') if item.date else 'Unknown'}

**Source:** [{item.source_name}]({item.link})

**Preview:**
{content_preview[:500]}

---
*Please review the source link for full details.*"""
        
        return response
    
    # =========================================================================
    # GitHub Integration
    # =========================================================================
    
    def _create_github_issue(self, site_info: dict, item: ContentItem, 
                              summary: str) -> Optional[int]:
        """Create a GitHub Issue and return the issue number."""
        if not self.github_token or not self.github_repo:
            logger.error("GitHub credentials not configured")
            return None
        
        try:
            title = f"{site_info['emoji']} {item.title[:80]}"
            
            body = f"""## {site_info['name']} - Release Detected

**Detection Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
**Source:** [{item.source_name}]({item.link})
**Published:** {item.date.strftime('%Y-%m-%d') if item.date else 'Unknown'}

---

{summary}

---

### Source Link
🔗 {item.link}

### Actions
- [ ] Review the release
- [ ] Update affected code/docs if needed
- [ ] This issue will auto-close after 7 days

---
*Automated by AI Website Monitor*
"""
            
            api_url = f"https://api.github.com/repos/{self.github_repo}/issues"
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            data = {
                'title': title,
                'body': body,
                'labels': ['ai-release', site_info['label'], 'auto-close-7d']
            }
            
            response = requests.post(api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            issue = response.json()
            logger.info(f"✅ Created issue #{issue['number']}: {issue['html_url']}")
            return issue['number']
            
        except Exception as e:
            logger.error(f"Failed to create issue: {e}")
            self.errors.append(f"Issue creation failed: {str(e)}")
            return None
    
    def _close_old_issues(self):
        """Close issues older than 7 days with the auto-close label."""
        if not self.github_token or not self.github_repo:
            return
        
        try:
            # Find open issues with auto-close label
            api_url = f"https://api.github.com/repos/{self.github_repo}/issues"
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            params = {
                'labels': 'auto-close-7d',
                'state': 'open',
                'per_page': 100
            }
            
            response = requests.get(api_url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            issues = response.json()
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(days=7)
            
            for issue in issues:
                created = datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00'))
                if created < cutoff:
                    # Close the issue
                    close_url = f"{api_url}/{issue['number']}"
                    close_data = {
                        'state': 'closed',
                        'state_reason': 'completed'
                    }
                    requests.patch(close_url, headers=headers, json=close_data, timeout=30)
                    
                    # Add a comment
                    comment_url = f"{api_url}/{issue['number']}/comments"
                    comment_data = {
                        'body': '🤖 Auto-closed after 7 days. Reopen if action is still needed.'
                    }
                    requests.post(comment_url, headers=headers, json=comment_data, timeout=30)
                    
                    logger.info(f"Auto-closed issue #{issue['number']}")
            
        except Exception as e:
            logger.warning(f"Failed to close old issues: {e}")
    
    def _create_error_issue(self):
        """Create an issue for errors if any occurred."""
        if not self.errors or not self.github_token or not self.github_repo:
            return
        
        try:
            title = "⚠️ Monitor Errors Detected"
            body = f"""## Website Monitor Error Report

**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

### Errors
{"".join(f"- {e}" + chr(10) for e in self.errors)}

---
*Automated by AI Website Monitor*
"""
            
            api_url = f"https://api.github.com/repos/{self.github_repo}/issues"
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            data = {
                'title': title,
                'body': body,
                'labels': ['error', 'automated']
            }
            
            response = requests.post(api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            logger.info(f"Created error issue #{response.json()['number']}")
            
        except Exception as e:
            logger.error(f"Failed to create error issue: {e}")
    
    # =========================================================================
    # Main Processing Logic
    # =========================================================================
    
    def _process_site(self, site_key: str, site_info: dict) -> List[Tuple[ContentItem, str, str]]:
        """
        Process a single site and return list of (item, relevance_reason, summary) 
        for relevant items.
        """
        logger.info(f"Processing {site_info['name']}...")
        
        # Extract items based on type
        if site_info['type'] == 'rss':
            items = self._extract_rss_items(
                site_info['url'], 
                site_info['name'],
                site_info['base_url']
            )
        elif site_info['type'] == 'html_changelog':
            items = self._extract_changelog_items(
                site_info['url'],
                site_info['name'],
                site_info.get('selectors', ['main']),
                site_info['base_url']
            )
        else:
            logger.warning(f"Unknown type: {site_info['type']}")
            return []
        
        if not items:
            logger.info(f"No items from {site_info['name']}")
            return []
        
        # Filter to items within time window
        recent_items = [
            item for item in items 
            if self._is_within_time_window(item.date)
        ][:self.max_items_per_source]
        
        logger.info(f"Found {len(recent_items)} recent items from {site_info['name']}")
        
        # Load handled releases
        handled = self._load_handled_releases()
        
        # Filter out already handled
        new_items = [
            item for item in recent_items 
            if not self._is_already_handled(item, handled)
        ]
        
        if not new_items:
            logger.info(f"No new unhandled items from {site_info['name']}")
            return []
        
        logger.info(f"Checking {len(new_items)} new items from {site_info['name']}")
        
        relevant_items = []
        
        for item in new_items:
            # Fetch full article content
            article_content = None
            if item.link:
                logger.info(f"Fetching article: {item.link}")
                article_content = self._fetch_article_content(item.link, site_info['base_url'])
            
            # Step 1: Check relevance
            is_relevant, reason = self._check_relevance(item, article_content)
            
            if is_relevant:
                logger.info(f"✓ Relevant: {item.title[:50]}... - {reason}")
                
                # Step 2: Generate summary
                summary = self._generate_issue_summary(item, article_content, reason)
                relevant_items.append((item, reason, summary))
            else:
                logger.info(f"✗ Not relevant: {item.title[:50]}... - {reason}")
        
        return relevant_items
    
    def check_all_websites(self):
        """Main entry point - check all websites in parallel."""
        logger.info("=" * 60)
        logger.info(f"Starting monitor at {datetime.now(timezone.utc)}")
        logger.info("=" * 60)
        
        # First, close old issues
        logger.info("Checking for issues to auto-close...")
        self._close_old_issues()
        
        # Load handled releases once
        handled_releases = self._load_handled_releases()
        
        # Process all sites in parallel
        all_relevant: List[Tuple[str, dict, ContentItem, str, str]] = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self._process_site, key, info): (key, info)
                for key, info in self.websites.items()
            }
            
            for future in as_completed(futures):
                site_key, site_info = futures[future]
                try:
                    results = future.result()
                    for item, reason, summary in results:
                        all_relevant.append((site_key, site_info, item, reason, summary))
                except Exception as e:
                    logger.error(f"Error processing {site_info['name']}: {e}")
                    self.errors.append(f"Processing error for {site_info['name']}: {str(e)}")
        
        # Create issues for relevant items
        issues_created = 0
        for site_key, site_info, item, reason, summary in all_relevant:
            issue_num = self._create_github_issue(site_info, item, summary)
            
            if issue_num:
                issues_created += 1
                # Add to handled releases
                handled_releases.append(HandledRelease(
                    id=item.id,
                    title=item.title,
                    url=item.link,
                    detected_at=datetime.now(timezone.utc),
                    issue_number=issue_num,
                    source=site_key
                ))
        
        # Save updated handled releases
        self._save_handled_releases(handled_releases)
        
        # Handle errors
        if self.errors:
            logger.warning(f"⚠️ {len(self.errors)} errors occurred")
            self._create_error_issue()
        
        # Summary
        if issues_created > 0:
            logger.info(f"\n✅ Created {issues_created} issue(s)")
        else:
            logger.info("\n✅ No new relevant releases detected")
        
        logger.info("=" * 60)
        logger.info("Monitor completed")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    try:
        monitor = WebsiteMonitor()
        monitor.check_all_websites()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
