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
        
        # Browser-like session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        
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
        except Exception as e:
            logger.warning(f"Config load error: {e}, using defaults")
            return default_config
    
    def _override_with_env_vars(self):
        if os.getenv('ASKSAGE_API_TOKEN'):
            self.config['asksage_api']['token'] = os.getenv('ASKSAGE_API_TOKEN')
            logger.info("Using API token from environment")
    
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
            
        except Exception as e:
            logger.error(f"RSS extraction failed for {url}: {e}")
            self.errors.append(f"RSS error ({source_name}): {e}")
        
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
            
        except Exception as e:
            logger.error(f"Anthropic extraction failed: {e}")
            self.errors.append(f"Anthropic error: {e}")
        
        return items
    
    def _extract_gemini_changelog(self, url: str, source_name: str) -> List[ContentItem]:
        """Extract changelog entries from Gemini docs."""
        items = []
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            
            import re
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
            
        except Exception as e:
            logger.error(f"Gemini extraction failed: {e}")
            self.errors.append(f"Gemini error: {e}")
        
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
                except:
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
    
    def _analyze_item(self, item: ContentItem) -> Tuple[bool, str, str]:
        """
        Analyze a single item to determine if it's a relevant release.
        
        Returns:
            Tuple of (is_relevant, summary, issue_body)
        """
        content = f"TITLE: {item.title}\n"
        content += f"DATE: {item.date}\n" if item.date else ""
        content += f"SOURCE: {item.source_name}\n"
        content += f"LINK: {item.link}\n"
        content += f"\nDESCRIPTION:\n{item.description}\n" if item.description else ""
        
        if item.full_content:
            content += f"\nFULL ARTICLE CONTENT:\n{item.full_content}\n"
        
        # Query 1: Summarize
        prompt1 = f"""Read the following article and tell me what it is announcing or discussing.

{content}

In 2-3 sentences, what is the main topic or announcement of this article? Be specific - mention any product names, model names, version numbers, or features discussed."""

        response1, error1 = self.llm.query(prompt1)
        
        if error1:
            logger.error(f"Query 1 failed for {item.title[:40]}: {error1}")
            return False, "", ""
        
        logger.info(f"Summary: {response1[:100]}...")
        
        # Query 2: Classify
        prompt2 = f"""Based on this summary of an article from {item.source_name}:

Title: {item.title}
Summary: {response1}

Is this article announcing or describing ANY of the following?
1. A new AI model or new version of an existing model
2. New API features, endpoints, or capabilities
3. Pricing changes for AI models or APIs
4. New SDKs, libraries, or developer tools
5. Model deprecations or breaking API changes
6. Significant capability improvements (context window, speed, multimodal support, etc.)

Answer with EXACTLY this format:
RELEVANT: YES or NO
REASON: One sentence explaining your answer"""

        response2, error2 = self.llm.query(prompt2)
        
        if error2:
            logger.error(f"Query 2 failed for {item.title[:40]}: {error2}")
            return False, "", ""
        
        # Parse response
        is_relevant = False
        reason = response2
        
        response2_upper = response2.upper()
        if 'RELEVANT: YES' in response2_upper or 'RELEVANT:YES' in response2_upper:
            is_relevant = True
        elif response2_upper.strip().startswith('YES'):
            is_relevant = True
        
        if 'REASON:' in response2.upper():
            idx = response2.upper().index('REASON:')
            reason = response2[idx + 7:].strip()
        
        if not is_relevant:
            logger.info(f"Not relevant: {item.title[:40]}... - {reason[:50]}")
            return False, response1, ""
        
        logger.info(f"✓ RELEVANT: {item.title[:40]}... - {reason[:50]}")
        
        # Query 3: Generate issue body
        prompt3 = f"""Create a GitHub issue body summarizing this AI product/API release.

Title: {item.title}
Source: {item.source_name}
Link: {item.link}
Summary: {response1}

{f"Full content: {item.full_content[:5000]}" if item.full_content else f"Description: {item.description}"}

Write a clear, concise summary in Markdown format including:
- What was released/announced
- Key features or changes
- Technical details if available (model names, versions, capabilities, pricing)
- Any action items for developers

Keep it professional and focused on the facts."""

        response3, error3 = self.llm.query(prompt3)
        
        if error3:
            logger.error(f"Query 3 failed for {item.title[:40]}: {error3}")
            response3 = f"""### {item.title}

**Summary:** {response1}

**Source:** [{item.source_name}]({item.link})

{item.description if item.description else 'See link for full details.'}"""
        
        return True, response1, response3
    
    def _analyze_items_parallel(self, items: List[ContentItem], 
                                 handled: List[HandledRelease]) -> List[Tuple[ContentItem, str, str]]:
        """Analyze multiple items in parallel."""
        # Filter out already handled items first
        to_analyze = [item for item in items if not self._is_handled(item, handled)]
        
        if not to_analyze:
            return []
        
        logger.info(f"Analyzing {len(to_analyze)} items in parallel (max {self.max_llm_workers} workers)...")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_llm_workers) as executor:
            future_to_item = {
                executor.submit(self._analyze_item, item): item 
                for item in to_analyze
            }
            
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    is_relevant, summary, body = future.result()
                    if is_relevant:
                        results.append((item, summary, body))
                except Exception as e:
                    logger.error(f"Analysis failed for {item.title[:40]}: {e}")
        
        logger.info(f"Found {len(results)} relevant items")
        return results
    
    # =========================================================================
    # GitHub
    # =========================================================================
    
    def _create_issue(self, site_info: dict, item: ContentItem, summary: str, body: str) -> Optional[int]:
        if not self.github_token or not self.github_repo:
            logger.error("GitHub not configured")
            return None
        
        try:
            title = f"{site_info['emoji']} {item.source_name}: {item.title[:70]}"
            
            full_body = f"""## New Release Detected

**Source:** [{item.source_name}]({item.link})  
**Detected:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

---

{body}

---

### Quick Summary
{summary}

### Actions
- [ ] Review the release
- [ ] Update affected code/docs if needed

*Auto-generated by AI Website Monitor*
"""
            
            resp = requests.post(
                f"https://api.github.com/repos/{self.github_repo}/issues",
                headers={
                    'Authorization': f'token {self.github_token}',
                    'Accept': 'application/vnd.github.v3+json'
                },
                json={
                    'title': title,
                    'body': full_body,
                    'labels': ['ai-release', site_info['label'], 'auto-close-7d']
                },
                timeout=30
            )
            resp.raise_for_status()
            
            issue = resp.json()
            logger.info(f"✅ Created issue #{issue['number']}")
            return issue['number']
            
        except Exception as e:
            logger.error(f"Issue creation failed: {e}")
            self.errors.append(f"Issue creation: {e}")
            return None
    
    def _close_old_issues(self):
        if not self.github_token or not self.github_repo:
            return
        
        try:
            resp = requests.get(
                f"https://api.github.com/repos/{self.github_repo}/issues",
                headers={
                    'Authorization': f'token {self.github_token}',
                    'Accept': 'application/vnd.github.v3+json'
                },
                params={'labels': 'auto-close-7d', 'state': 'open'},
                timeout=30
            )
            resp.raise_for_status()
            
            cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            for issue in resp.json():
                created = datetime.fromisoformat(issue['created_at'].replace('Z', '+00:00'))
                if created < cutoff:
                    requests.patch(
                        f"https://api.github.com/repos/{self.github_repo}/issues/{issue['number']}",
                        headers={
                            'Authorization': f'token {self.github_token}',
                            'Accept': 'application/vnd.github.v3+json'
                        },
                        json={'state': 'closed'},
                        timeout=30
                    )
                    logger.info(f"Auto-closed issue #{issue['number']}")
        except Exception as e:
            logger.warning(f"Close old issues error: {e}")
    
    def _create_error_issue(self):
        if not self.errors or not self.github_token or not self.github_repo:
            return
        
        try:
            requests.post(
                f"https://api.github.com/repos/{self.github_repo}/issues",
                headers={
                    'Authorization': f'token {self.github_token}',
                    'Accept': 'application/vnd.github.v3+json'
                },
                json={
                    'title': '⚠️ Monitor Errors',
                    'body': f"**Errors:**\n" + '\n'.join(f"- {e}" for e in self.errors),
                    'labels': ['error']
                },
                timeout=30
            )
        except Exception as e:
            logger.error(f"Error issue creation failed: {e}")
    
    # =========================================================================
    # Main
    # =========================================================================
    
    def _process_site(self, site_key: str, site_info: dict, 
                       handled: List[HandledRelease]) -> List[Tuple[ContentItem, str, str]]:
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
        results = self._analyze_items_parallel(new_items, handled)
        
        return results
    
    def check_all_websites(self):
        logger.info("=" * 60)
        logger.info(f"Monitor started: {datetime.now(timezone.utc)}")
        logger.info(f"Limits: {self.time_window_days} days OR {self.max_items_per_source} items per source")
        logger.info("=" * 60)
        
        self._close_old_issues()
        
        handled = self._load_handled()
        logger.info(f"Loaded {len(handled)} previously handled releases")
        
        # Process all sites in parallel
        all_results: List[Tuple[str, dict, ContentItem, str, str]] = []
        
        with ThreadPoolExecutor(max_workers=len(self.websites)) as executor:
            future_to_site = {
                executor.submit(self._process_site, key, info, handled): (key, info)
                for key, info in self.websites.items()
            }
            
            for future in as_completed(future_to_site):
                site_key, site_info = future_to_site[future]
                try:
                    results = future.result()
                    for item, summary, body in results:
                        all_results.append((site_key, site_info, item, summary, body))
                except Exception as e:
                    logger.error(f"Site processing failed for {site_info['name']}: {e}")
                    self.errors.append(f"{site_info['name']}: {e}")
        
        # Create issues
        issues_created = 0
        for site_key, site_info, item, summary, body in all_results:
            issue_num = self._create_issue(site_info, item, summary, body)
            if issue_num:
                issues_created += 1
                handled.append(HandledRelease(
                    id=item.id,
                    title=item.title,
                    link=item.link,
                    issue_number=issue_num,
                    detected_at=datetime.now(timezone.utc).isoformat()
                ))
        
        self._save_handled(handled)
        
        if self.errors:
            logger.warning(f"⚠️ {len(self.errors)} errors occurred")
            self._create_error_issue()
        
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
