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
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
import requests
from bs4 import BeautifulSoup


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
    """Represents a single content item (RSS entry, changelog section, etc.)."""
    id: str
    title: str
    content: str
    date: str
    link: str
    source: str
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ContentItem':
        return cls(**data)


class WebsiteMonitor:
    """Monitors websites for changes and creates GitHub Issues."""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the monitor with configuration."""
        self.config = self._load_config(config_path)
        self._override_with_env_vars()
        self.storage_dir = Path(self.config.get("storage_dir", "storage"))
        self.storage_dir.mkdir(exist_ok=True)
        
        # GitHub configuration
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.github_repo = os.getenv('GITHUB_REPOSITORY')
        
        # Track errors for reporting
        self.errors: List[str] = []
        
        # Websites to monitor - improved sources
        self.websites = {
            "openai_news": {
                "url": "https://openai.com/news/rss.xml",
                "name": "OpenAI News & Releases",
                "type": "rss",
                "follow_links": True,
                "emoji": "🔵",
                "label": "openai"
            },
            "gemini_changelog": {
                "url": "https://ai.google.dev/gemini-api/docs/changelog",
                "name": "Google Gemini API Changelog",
                "type": "html",
                "selectors": ["main", "article", ".devsite-article-body"],
                "follow_links": True,
                "emoji": "🟡",
                "label": "gemini"
            },
            "anthropic_news": {
                "url": "https://www.anthropic.com/news",
                "name": "Anthropic News",
                "type": "html",
                "selectors": ["main", "article", ".content"],
                "follow_links": True,
                "emoji": "🟠",
                "label": "anthropic"
            },
            "anthropic_api": {
                "url": "https://docs.anthropic.com/en/release-notes/overview",
                "name": "Anthropic API Release Notes",
                "type": "html",
                "selectors": ["main", "article", ".content"],
                "follow_links": False,
                "emoji": "🟠",
                "label": "anthropic"
            }
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file with fallback to defaults."""
        default_config = {
            "asksage_api": {
                "url": "https://api.asksage.ai/server/query",
                "token": "",
                "model": "google-claude-45-opus",
                "temperature": 0.2
            },
            "storage_dir": "storage",
            "max_content_length": 50000,
            "max_linked_content_length": 20000
        }
        
        try:
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                for key in default_config:
                    if key in loaded_config:
                        if isinstance(default_config[key], dict):
                            default_config[key].update(loaded_config[key])
                        else:
                            default_config[key] = loaded_config[key]
                return default_config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return default_config
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config: {e}")
            return default_config
    
    def _override_with_env_vars(self):
        """Override config with environment variables if present."""
        if os.getenv('ASKSAGE_API_TOKEN'):
            self.config['asksage_api']['token'] = os.getenv('ASKSAGE_API_TOKEN')
            logger.info("Using API token from environment variable")
    
    def _generate_item_id(self, *args) -> str:
        """Generate a unique ID for a content item based on its attributes."""
        combined = '|'.join(str(arg) for arg in args if arg)
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()[:16]
    
    def _clean_text(self, text: str) -> str:
        """Clean text while preserving meaningful content including dates."""
        if not text:
            return ""
        # Replace non-breaking spaces with regular spaces
        text = re.sub(r'[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000\uFEFF]', ' ', text)
        # Normalize whitespace but preserve newlines for structure
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def _extract_rss_items(self, url: str) -> List[ContentItem]:
        """
        Extract individual items from an RSS feed with unique IDs.
        
        Returns:
            List of ContentItem objects
        """
        items = []
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; AIMonitor/1.0)'}
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'xml')
            
            for item in soup.find_all('item'):
                title_elem = item.find('title')
                desc_elem = item.find('description')
                link_elem = item.find('link')
                guid_elem = item.find('guid')
                pub_date_elem = item.find('pubDate')
                
                title = title_elem.get_text(strip=True) if title_elem else ""
                link = link_elem.get_text(strip=True) if link_elem else ""
                guid = guid_elem.get_text(strip=True) if guid_elem else ""
                pub_date = pub_date_elem.get_text(strip=True) if pub_date_elem else ""
                
                # Parse description HTML
                description = ""
                if desc_elem:
                    desc_soup = BeautifulSoup(desc_elem.get_text(), 'html.parser')
                    description = desc_soup.get_text(separator='\n', strip=True)
                
                # Generate unique ID from guid, link, or title
                item_id = self._generate_item_id(guid or link or title)
                
                items.append(ContentItem(
                    id=item_id,
                    title=title,
                    content=self._clean_text(description),
                    date=pub_date,
                    link=link,
                    source=url
                ))
            
            logger.info(f"Extracted {len(items)} items from RSS feed")
            
        except Exception as e:
            logger.error(f"Failed to extract RSS items from {url}: {e}")
            self.errors.append(f"RSS extraction error for {url}: {str(e)}")
        
        return items
    
    def _extract_html_items(self, url: str, selectors: List[str]) -> List[ContentItem]:
        """
        Extract content sections from HTML pages.
        Attempts to identify individual news items, changelog entries, etc.
        
        Returns:
            List of ContentItem objects
        """
        items = []
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            
            response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
                element.decompose()
            
            # Try to find the main content area
            main_content = None
            for selector in selectors:
                try:
                    main_content = soup.select_one(selector)
                    if main_content:
                        break
                except Exception:
                    continue
            
            if not main_content:
                main_content = soup.body if soup.body else soup
            
            # Try to find individual items (articles, sections, changelog entries)
            item_selectors = [
                'article',
                '.changelog-entry',
                '.release-note',
                '.news-item',
                'section[data-date]',
                '.update-item',
                'h2, h3'  # Fallback: treat headings as item delimiters
            ]
            
            found_items = []
            for selector in item_selectors:
                try:
                    found = main_content.select(selector)
                    if found and len(found) > 1:
                        found_items = found
                        break
                except Exception:
                    continue
            
            if found_items:
                # Extract individual items
                for elem in found_items[:50]:  # Limit to 50 most recent
                    title = ""
                    # Try to find a title
                    title_elem = elem.find(['h1', 'h2', 'h3', 'h4'])
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                    elif elem.name in ['h2', 'h3']:
                        title = elem.get_text(strip=True)
                    
                    content = self._clean_text(elem.get_text(separator='\n', strip=True))
                    
                    # Find link if available
                    link = ""
                    link_elem = elem.find('a', href=True)
                    if link_elem:
                        link = link_elem['href']
                        if link.startswith('/'):
                            from urllib.parse import urljoin
                            link = urljoin(url, link)
                    
                    # Find date if available
                    date = ""
                    date_elem = elem.find(['time', '.date', '[datetime]'])
                    if date_elem:
                        date = date_elem.get('datetime', '') or date_elem.get_text(strip=True)
                    
                    # Generate ID from title and content
                    item_id = self._generate_item_id(title, content[:200])
                    
                    if title or content:
                        items.append(ContentItem(
                            id=item_id,
                            title=title,
                            content=content,
                            date=date,
                            link=link,
                            source=url
                        ))
            else:
                # Fallback: treat entire content as one item
                full_content = self._clean_text(main_content.get_text(separator='\n', strip=True))
                item_id = self._generate_item_id(full_content[:500])
                items.append(ContentItem(
                    id=item_id,
                    title="Full Page Content",
                    content=full_content,
                    date=datetime.now().isoformat(),
                    link=url,
                    source=url
                ))
            
            logger.info(f"Extracted {len(items)} items from HTML page")
            
        except Exception as e:
            logger.error(f"Failed to extract HTML items from {url}: {e}")
            self.errors.append(f"HTML extraction error for {url}: {str(e)}")
        
        return items
    
    def _fetch_linked_content(self, url: str) -> Optional[str]:
        """
        Fetch the full content from a linked page for additional context.
        
        Returns:
            Cleaned text content or None if fetch fails
        """
        if not url or not url.startswith('http'):
            return None
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml',
            }
            
            response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
                element.decompose()
            
            # Try to find main content
            main_content = None
            for selector in ['main', 'article', '.content', '.post-content', '.blog-post', '[role="main"]']:
                try:
                    main_content = soup.select_one(selector)
                    if main_content:
                        break
                except Exception:
                    continue
            
            if not main_content:
                main_content = soup.body if soup.body else soup
            
            content = self._clean_text(main_content.get_text(separator='\n', strip=True))
            
            max_length = self.config.get('max_linked_content_length', 20000)
            return content[:max_length]
            
        except Exception as e:
            logger.warning(f"Failed to fetch linked content from {url}: {e}")
            return None
    
    def _load_seen_items(self, site_key: str) -> Set[str]:
        """Load the set of previously seen item IDs."""
        seen_file = self.storage_dir / f"{site_key}_seen.json"
        try:
            if seen_file.exists():
                with open(seen_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('seen_ids', []))
        except Exception as e:
            logger.warning(f"Error loading seen items for {site_key}: {e}")
        return set()
    
    def _save_seen_items(self, site_key: str, seen_ids: Set[str], items: List[ContentItem]):
        """Save the set of seen item IDs and recent items for context."""
        seen_file = self.storage_dir / f"{site_key}_seen.json"
        items_file = self.storage_dir / f"{site_key}_items.json"
        
        try:
            # Save seen IDs (keep last 500 to prevent unbounded growth)
            seen_list = list(seen_ids)[-500:]
            with open(seen_file, 'w') as f:
                json.dump({'seen_ids': seen_list, 'updated': datetime.now().isoformat()}, f)
            
            # Save recent items for context (keep last 50)
            items_data = [item.to_dict() for item in items[:50]]
            with open(items_file, 'w') as f:
                json.dump({'items': items_data, 'updated': datetime.now().isoformat()}, f, indent=2)
            
            logger.info(f"Saved {len(seen_list)} seen IDs for {site_key}")
        except Exception as e:
            logger.error(f"Error saving seen items for {site_key}: {e}")
    
    def _analyze_new_items_with_llm(self, site_name: str, new_items: List[ContentItem], 
                                     linked_contents: Dict[str, str]) -> Optional[str]:
        """
        Use LLM to analyze NEW items and determine relevance.
        
        Args:
            site_name: Name of the website
            new_items: List of new ContentItem objects
            linked_contents: Dict mapping item links to their full content
        
        Returns:
            Summary of relevant changes or None if no relevant changes
        """
        if not new_items:
            return None
        
        system_prompt = """You are an expert AI/ML technical analyst. Your job is to identify NEW AI product releases and API changes that developers need to know about.

CRITICAL: You MUST flag these types of announcements - DO NOT MISS THEM:

🚨 HIGH PRIORITY - ALWAYS FLAG:
- New model releases (GPT-5, Claude 4, Gemini 2.5, etc.)
- New model versions (GPT-4.1, Claude 3.5 Sonnet v2, etc.)
- Model capability announcements (new context window, vision, audio, etc.)
- New API endpoints or features
- Pricing changes for models or APIs
- Model deprecations or sunset announcements
- New SDKs or official libraries
- Rate limit changes
- New developer tools or playgrounds

⚠️ MEDIUM PRIORITY - FLAG IF DEVELOPER-RELEVANT:
- Research announcements that include model access
- Beta/preview program announcements
- Significant documentation updates about capabilities
- Integration announcements (new platforms, partnerships with dev impact)

❌ DO NOT FLAG:
- Pure research papers with no product
- Corporate news (funding, hiring, office locations)
- Policy/safety announcements with no product changes
- Opinion pieces or thought leadership
- Event announcements without product reveals

IMPORTANT RULES:
1. When in doubt, FLAG IT. False positives are acceptable; missed releases are not.
2. Look for model names, version numbers, "introducing", "announcing", "now available", "launching"
3. Check the FULL linked content, not just the preview
4. Multiple items may be relevant - include ALL of them

RESPONSE FORMAT for relevant items:

### 🚀 Summary
[One-line summary of the most important announcement]

### New Releases & Updates

#### [Product/Model Name]
- **What's New:** [Clear description]
- **Key Details:** [Model specs, pricing, availability, etc.]
- **Developer Impact:** [How this affects developers]

[Repeat for each relevant item]

### Quick Links
- [Link 1]
- [Link 2]

---

If NOTHING is relevant, respond with exactly: "None"
"""

        # Build content for analysis - be generous with content
        items_text = ""
        for i, item in enumerate(new_items, 1):
            items_text += f"\n{'='*60}\n"
            items_text += f"## NEW ITEM {i}\n"
            items_text += f"**Title:** {item.title}\n"
            items_text += f"**Date:** {item.date}\n"
            items_text += f"**Link:** {item.link}\n"
            items_text += f"**Content Preview:**\n{item.content[:3000]}\n"
            
            # Add linked full content if available - this is critical for full details
            if item.link in linked_contents and linked_contents[item.link]:
                items_text += f"\n**=== FULL ARTICLE CONTENT ===**\n{linked_contents[item.link][:25000]}\n"
                items_text += f"**=== END FULL CONTENT ===**\n"
        
        user_message = f"""Website: {site_name}

I detected {len(new_items)} NEW item(s) on this AI company's website. Analyze each one carefully.

{items_text}

{'='*60}

TASK:
1. Read EVERY new item carefully, including the full article content
2. Identify ANY announcements about new models, API changes, or developer-relevant updates
3. Look for keywords: "introducing", "announcing", "new", "launching", "available", version numbers
4. If you find relevant items, provide a detailed summary
5. If nothing is relevant, respond with exactly "None"

Remember: It's better to flag something that might be relevant than to miss a major release.

Your analysis:"""

        try:
            api_url = self.config['asksage_api']['url']
            api_token = self.config['asksage_api']['token']
            
            if not api_token:
                error_msg = "API token not configured"
                logger.error(error_msg)
                self.errors.append(error_msg)
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
            
            logger.info(f"Analyzing {len(new_items)} new items for {site_name} with LLM...")
            response = requests.post(api_url, headers=headers, files=files, timeout=180)
            response.raise_for_status()
            
            result = response.json()
            
            if isinstance(result, dict):
                response_text = result.get('response', result.get('message', str(result)))
            else:
                response_text = str(result)
            
            response_text = response_text.strip()
            
            logger.info(f"LLM response length: {len(response_text)} chars")
            logger.debug(f"LLM response preview: {response_text[:500]}...")
            
            # Check if LLM detected relevant changes
            if response_text.lower() in ['none', 'false', 'no changes', 'no relevant changes']:
                logger.info(f"No relevant changes in new items for {site_name}")
                return None
            
            if len(response_text) < 50:
                logger.warning(f"LLM response too short for {site_name}: {response_text}")
                return None
            
            logger.info(f"Relevant changes detected for {site_name}")
            return response_text
            
        except requests.exceptions.RequestException as e:
            error_msg = f"API request failed for {site_name}: {e}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return None
        except Exception as e:
            error_msg = f"Error analyzing changes for {site_name}: {e}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return None
    
    def _create_github_issue(self, site_key: str, site_info: dict, summary: str, 
                             new_items: List[ContentItem]):
        """Create a GitHub Issue for detected changes."""
        if not self.github_token or not self.github_repo:
            logger.error("GitHub token or repository not configured")
            return
        
        try:
            # Create descriptive title from first new item
            first_item_title = new_items[0].title[:60] if new_items else "Updates"
            title = f"{site_info['emoji']} {site_info['name']}: {first_item_title}"
            if len(new_items) > 1:
                title = f"{site_info['emoji']} {site_info['name']}: {len(new_items)} New Updates"
            
            # Build links section
            links_section = ""
            if new_items:
                links_section = "\n### Source Links\n"
                for item in new_items[:10]:
                    if item.link:
                        links_section += f"- [{item.title or 'Link'}]({item.link})\n"
            
            body = f"""## {site_info['name']} - New Updates Detected

**Detection Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Items Detected:** {len(new_items)}
**Primary Source:** {site_info['url']}

---

{summary}

{links_section}

---

### Actions
- [ ] Review changes in detail
- [ ] Update any affected code/documentation
- [ ] Test compatibility if needed
- [ ] Close this issue when complete

---
*This issue was automatically created by the AI Website Monitor*
"""
            
            api_url = f"https://api.github.com/repos/{self.github_repo}/issues"
            
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            
            data = {
                'title': title,
                'body': body,
                'labels': ['ai-update', site_info['label'], 'automated']
            }
            
            response = requests.post(api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            issue_data = response.json()
            logger.info(f"Created issue #{issue_data['number']}: {issue_data['html_url']}")
            
        except Exception as e:
            logger.error(f"Failed to create GitHub issue: {e}")
            self.errors.append(f"GitHub issue creation failed: {str(e)}")
    
    def _create_error_issue(self):
        """Create a GitHub Issue for errors encountered during monitoring."""
        if not self.errors or not self.github_token or not self.github_repo:
            return
        
        try:
            title = "⚠️ Website Monitor - Errors Detected"
            error_list = '\n'.join([f"- {error}" for error in self.errors])
            
            body = f"""## Website Monitor Error Report

**Detection Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}

The following errors occurred during the monitoring run:

{error_list}

### Recommended Actions
- [ ] Check API credentials
- [ ] Verify website URLs are accessible
- [ ] Review error logs in workflow artifacts
- [ ] Close this issue when resolved

---
*This issue was automatically created by the AI Website Monitor*
"""
            
            api_url = f"https://api.github.com/repos/{self.github_repo}/issues"
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            data = {
                'title': title,
                'body': body,
                'labels': ['error', 'automated', 'monitor-health']
            }
            
            response = requests.post(api_url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            logger.info(f"Created error issue #{response.json()['number']}")
            
        except Exception as e:
            logger.error(f"Failed to create error issue: {e}")
    
    def check_website(self, site_key: str, site_info: dict) -> int:
        """
        Check a single website for new content.
        
        Returns:
            Number of issues created (0 or 1)
        """
        logger.info(f"\nChecking {site_info['name']}...")
        
        # Extract items based on type
        if site_info.get('type') == 'rss':
            current_items = self._extract_rss_items(site_info['url'])
        else:
            current_items = self._extract_html_items(
                site_info['url'], 
                site_info.get('selectors', ['main', 'article'])
            )
        
        if not current_items:
            logger.warning(f"No items extracted from {site_info['name']}")
            return 0
        
        # Load previously seen items
        seen_ids = self._load_seen_items(site_key)
        
        # Find new items
        new_items = [item for item in current_items if item.id not in seen_ids]
        
        if not new_items:
            logger.info(f"No new items for {site_info['name']}")
            # Still save to update timestamp
            all_ids = seen_ids.union({item.id for item in current_items})
            self._save_seen_items(site_key, all_ids, current_items)
            return 0
        
        logger.info(f"Found {len(new_items)} new items for {site_info['name']}")
        
        # Check if this is first run (no previous data)
        if not seen_ids:
            logger.info(f"First run for {site_info['name']}, saving baseline without analysis...")
            all_ids = {item.id for item in current_items}
            self._save_seen_items(site_key, all_ids, current_items)
            return 0
        
        # Fetch linked content for new items if enabled
        linked_contents: Dict[str, str] = {}
        if site_info.get('follow_links', False):
            for item in new_items[:5]:  # Limit to 5 links to avoid rate limiting
                if item.link:
                    logger.info(f"Fetching linked content: {item.link}")
                    linked_content = self._fetch_linked_content(item.link)
                    if linked_content:
                        linked_contents[item.link] = linked_content
        
        # Analyze with LLM
        summary = self._analyze_new_items_with_llm(site_info['name'], new_items, linked_contents)
        
        issues_created = 0
        if summary:
            self._create_github_issue(site_key, site_info, summary, new_items)
            issues_created = 1
        
        # Update seen items
        all_ids = seen_ids.union({item.id for item in current_items})
        self._save_seen_items(site_key, all_ids, current_items)
        
        return issues_created
    
    def check_all_websites(self):
        """Main method to check all websites for changes."""
        logger.info("=" * 60)
        logger.info(f"Starting website check at {datetime.now()}")
        logger.info("=" * 60)
        
        issues_created = 0
        
        for site_key, site_info in self.websites.items():
            try:
                issues_created += self.check_website(site_key, site_info)
            except Exception as e:
                logger.error(f"Error checking {site_info['name']}: {e}")
                self.errors.append(f"Check failed for {site_info['name']}: {str(e)}")
        
        # Create error issue if any errors occurred
        if self.errors:
            logger.warning(f"\n⚠️  {len(self.errors)} error(s) occurred during monitoring")
            self._create_error_issue()
        
        # Summary
        if issues_created > 0:
            logger.info(f"\n✅ Created {issues_created} GitHub issue(s) for detected changes")
        else:
            logger.info("\n✅ No relevant new content detected across all monitored websites")
        
        logger.info("=" * 60)
        logger.info("Website check completed")
        logger.info("=" * 60)


def main():
    """Main entry point for the script."""
    try:
        monitor = WebsiteMonitor()
        monitor.check_all_websites()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
