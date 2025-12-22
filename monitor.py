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
from typing import Dict, Optional, Tuple
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
        self.errors = []
        
        # Websites to monitor
        self.websites = {
            "openai": {
                "url": "https://openai.com/news/rss.xml",
                "name": "OpenAI Product Releases",
                "type": "rss",
                "selectors": ["item"],
                "emoji": "🔵",
                "label": "openai"
            },
            "gemini": {
                "url": "https://ai.google.dev/gemini-api/docs/changelog.md.txt",
                "name": "Google Gemini API Changelog",
                "type": "text",
                "selectors": [],
                "emoji": "🟡",
                "label": "gemini"
            },
            "anthropic": {
                "url": "https://www.anthropic.com/news",
                "name": "Anthropic News",
                "type": "html",
                "selectors": ["main", "article", ".content"],
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
                "model": "google-gemini-2.5-pro",
                "temperature": 0.1
            },
            "storage_dir": "storage"
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
    
    def _normalize_content(self, content: str) -> str:
        """
        Normalize content to ignore insignificant changes.
        
        This removes:
        - Extra whitespace
        - Timestamps and dates
        - Session IDs and tracking parameters
        - Non-breaking spaces and special characters that might vary
        """
        # Replace non-breaking spaces and other Unicode spaces with regular spaces
        content = re.sub(r'[\u00A0\u1680\u2000-\u200B\u202F\u205F\u3000\uFEFF]', ' ', content)
        
        # Remove common dynamic elements
        # Remove ISO timestamps
        content = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[.\d]*Z?', '', content)
        
        # Remove common date formats
        content = re.sub(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}', '', content)
        content = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}', '', content)
        
        # Remove time stamps
        content = re.sub(r'\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM|am|pm)?', '', content)
        
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()
        
        # Remove any remaining problematic Unicode characters
        content = content.encode('ascii', 'ignore').decode('ascii')
        
        return content
    
    def _fetch_rss_content(self, url: str) -> Optional[str]:
        """
        Fetch and parse RSS feed content.
        
        Args:
            url: The RSS feed URL
        
        Returns:
            Clean text content from RSS items or None if fetch fails
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; AIMonitor/1.0)',
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'xml')
            
            # Extract all items
            items = soup.find_all('item')
            
            # Build content from items (title + description)
            content_parts = []
            for item in items:
                title = item.find('title')
                description = item.find('description')
                pub_date = item.find('pubDate')
                
                if title:
                    content_parts.append(f"TITLE: {title.get_text(strip=True)}")
                if description:
                    # Parse HTML in description
                    desc_soup = BeautifulSoup(description.get_text(), 'html.parser')
                    desc_text = desc_soup.get_text(separator='\n', strip=True)
                    content_parts.append(f"DESCRIPTION: {desc_text}")
                if pub_date:
                    content_parts.append(f"DATE: {pub_date.get_text(strip=True)}")
                content_parts.append("---")
            
            content = '\n'.join(content_parts)
            return self._normalize_content(content)
            
        except Exception as e:
            logger.error(f"Failed to fetch RSS feed {url}: {e}")
            self.errors.append(f"RSS fetch error for {url}: {str(e)}")
            return None
    
    def _fetch_html_content(self, url: str, selectors: list) -> Optional[str]:
        """
        Fetch and extract clean text content from HTML website.
        
        Args:
            url: The URL to fetch
            selectors: List of CSS selectors to try (in order of preference)
        
        Returns:
            Clean text content or None if fetch fails
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
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
            }
            
            response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
            response.raise_for_status()
            
            # Handle encoding properly
            response.encoding = response.apparent_encoding or 'utf-8'
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
                element.decompose()
            
            # Try selectors in order
            content = None
            for selector in selectors:
                try:
                    element = soup.select_one(selector)
                    if element:
                        content = element.get_text(separator='\n', strip=True)
                        break
                except Exception:
                    continue
            
            # Fallback to body
            if not content:
                body = soup.body if soup.body else soup
                content = body.get_text(separator='\n', strip=True)
            
            # Clean up whitespace
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            content = '\n'.join(lines)
            
            # Normalize content to ignore insignificant changes
            return self._normalize_content(content)
            
        except Exception as e:
            logger.error(f"Failed to fetch HTML {url}: {e}")
            self.errors.append(f"HTML fetch error for {url}: {str(e)}")
            return None

    def _fetch_text_content(self, url: str) -> Optional[str]:
        """
        Fetch plain text content from a URL.
        
        Args:
            url: The URL to fetch
        
        Returns:
            Clean text content or None if fetch fails
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; AIMonitor/1.0)',
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Handle encoding properly
            response.encoding = response.apparent_encoding or 'utf-8'
            
            # Get the text content
            content = response.text
            
            # Clean up whitespace while preserving structure
            lines = [line.rstrip() for line in content.split('\n')]
            content = '\n'.join(lines)
            
            # Normalize content to ignore insignificant changes
            return self._normalize_content(content)
            
        except Exception as e:
            logger.error(f"Failed to fetch text {url}: {e}")
            self.errors.append(f"Text fetch error for {url}: {str(e)}")
            return None

    def _fetch_website_content(self, url: str, content_type: str, selectors: list) -> Optional[str]:
        """
        Fetch website content based on type.
        
        Args:
            url: The URL to fetch
            content_type: Type of content ('html', 'rss', 'text')
            selectors: List of CSS selectors (for HTML/RSS)
        
        Returns:
            Clean text content or None if fetch fails
        """
        if content_type == 'rss':
            return self._fetch_rss_content(url)
        elif content_type == 'text':
            return self._fetch_text_content(url)
        elif content_type == 'html':
            return self._fetch_html_content(url, selectors)
        else:
            # Default to HTML
            return self._fetch_html_content(url, selectors)
    
    def _get_content_hash(self, content: str) -> str:
        """Generate SHA256 hash of content for change detection."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def _load_previous_content(self, site_key: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Load previous content and hash for a website.
        
        Returns:
            Tuple of (content, hash) or (None, None) if no previous content exists
        """
        content_file = self.storage_dir / f"{site_key}_content.txt"
        hash_file = self.storage_dir / f"{site_key}_hash.txt"
        
        try:
            if content_file.exists() and hash_file.exists():
                with open(content_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                with open(hash_file, 'r') as f:
                    content_hash = f.read().strip()
                return content, content_hash
        except Exception as e:
            logger.warning(f"Error loading previous content for {site_key}: {e}")
        
        return None, None
    
    def _save_content(self, site_key: str, content: str, content_hash: str):
        """Save current content and hash to storage."""
        try:
            content_file = self.storage_dir / f"{site_key}_content.txt"
            hash_file = self.storage_dir / f"{site_key}_hash.txt"
            
            with open(content_file, 'w', encoding='utf-8') as f:
                f.write(content)
            with open(hash_file, 'w') as f:
                f.write(content_hash)
                
            logger.info(f"Saved content for {site_key}")
        except Exception as e:
            logger.error(f"Error saving content for {site_key}: {e}")
    
    def _analyze_with_llm(self, site_name: str, old_content: str, new_content: str) -> Optional[str]:
        """
        Use LLM to analyze changes and determine if notification is needed.
        
        Returns:
            Summary of changes or None if no relevant changes detected
        """
        system_prompt = """You are an expert technical analyst monitoring AI product releases and API changes.

Your task is to compare old and new website content and determine if there are any SIGNIFICANT product or API-related changes that developers should know about.

IMPORTANT: Only flag changes in these categories:
- New AI models or model versions
- Model pricing changes
- New API features or capabilities
- API or model deprecations or breaking changes
- Significant model capability improvements
- New SDKs or development tools
- New API endpoints or parameters

DO NOT flag:
- Blog posts about general AI topics
- Company news or partnerships (unless directly related to new products/APIs)
- Minor documentation updates
- Policy or research announcements
- General website updates
- UI/UX changes

If you detect relevant changes, respond with a clear, concise summary in GitHub-flavored Markdown format.

Start your response with:

### Summary
Brief 1-2 sentence overview of what changed.

### Changes Detected

#### [Category Name]
- **[Specific change]:** Brief description with details
- **[Specific change]:** Brief description with details

### Impact
Brief note on who/what this affects and recommended actions.

---

If NO relevant changes are detected, respond with exactly: "None"

Be strict - only report changes that would impact developers using these AI products/APIs. Provide specific details like model names, feature names, dates when available."""

        user_message = f"""Website: {site_name}

I'm comparing two versions of this website's content. Analyze the differences and identify ONLY product/API changes as defined above.

=== PREVIOUS CONTENT (15000 chars max) ===
{old_content[:15000]}

=== CURRENT CONTENT (15000 chars max) ===
{new_content[:15000]}

=== END OF CONTENT ===

Instructions:
1. Compare the two versions carefully
2. Identify what changed
3. Determine if the changes are product/API-related (as defined above)
4. If yes, provide a detailed summary in the format specified
5. If no, respond with exactly "None"

Your response:"""

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
            
            headers = {
                'x-access-tokens': api_token
            }
            
            logger.info(f"Analyzing changes for {site_name} with LLM...")
            response = requests.post(api_url, headers=headers, files=files, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract the response text
            if isinstance(result, dict):
                response_text = result.get('response', result.get('message', str(result)))
            else:
                response_text = str(result)
            
            response_text = response_text.strip()
            
            # Log the full LLM response for debugging
            logger.info(f"LLM response length: {len(response_text)} chars")
            logger.debug(f"LLM response: {response_text[:500]}...")
            
            # Check if LLM detected changes
            if response_text.lower() in ['none', 'false', 'no changes', 'no relevant changes', 'ok']:
                logger.info(f"No relevant changes detected for {site_name}")
                return None
            
            # Verify we got a real summary (not just "None" buried in text)
            if len(response_text) < 20:
                logger.warning(f"LLM response too short for {site_name}: {response_text}")
                return None
            
            logger.info(f"Changes detected for {site_name}")
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
    
    def _create_github_issue(self, site_key: str, site_info: dict, summary: str):
        """
        Create a GitHub Issue for detected changes.
        
        Args:
            site_key: The site identifier key
            site_info: Site configuration dict
            summary: LLM-generated summary of changes
        """
        if not self.github_token or not self.github_repo:
            logger.error("GitHub token or repository not configured")
            return
        
        try:
            # Prepare issue content
            title = f"{site_info['emoji']} {site_info['name']} - Updates Detected"
            
            body = f"""## {site_info['name']} - API/Product Changes Detected

**Detection Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Source:** {site_info['url']}

---

{summary}

---

### Actions
- [ ] Review changes
- [ ] Update any affected code/documentation
- [ ] Close this issue when complete

---
*This issue was automatically created by the AI Website Monitor*
"""
            
            # Create issue via GitHub API
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
            issue_number = issue_data['number']
            issue_url = issue_data['html_url']
            
            logger.info(f"Created issue #{issue_number}: {issue_url}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create GitHub issue: {e}")
        except Exception as e:
            logger.error(f"Error creating GitHub issue: {e}")
    
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
            
            issue_data = response.json()
            logger.info(f"Created error issue #{issue_data['number']}: {issue_data['html_url']}")
            
        except Exception as e:
            logger.error(f"Failed to create error issue: {e}")
    
    def check_all_websites(self):
        """Main method to check all websites for changes."""
        logger.info("="*60)
        logger.info(f"Starting website check at {datetime.now()}")
        logger.info("="*60)
        
        issues_created = 0
        
        for site_key, site_info in self.websites.items():
            logger.info(f"\nChecking {site_info['name']}...")
            
            # Fetch current content
            current_content = self._fetch_website_content(
                site_info['url'], 
                site_info.get('type', 'html'),
                site_info['selectors']
            )
            
            if not current_content:
                logger.warning(f"Failed to fetch content for {site_info['name']}, skipping...")
                continue
            
            # Calculate hash
            current_hash = self._get_content_hash(current_content)
            
            # Load previous content
            previous_content, previous_hash = self._load_previous_content(site_key)
            
            # Check if content changed
            if previous_hash == current_hash:
                logger.info(f"No changes detected for {site_info['name']}")
                continue
            
            logger.info(f"Content changed for {site_info['name']}")
            
            # If this is the first run, just save the content
            if previous_content is None:
                logger.info(f"First run for {site_info['name']}, saving baseline...")
                self._save_content(site_key, current_content, current_hash)
                continue
            
            # Analyze changes with LLM
            summary = self._analyze_with_llm(site_info['name'], previous_content, current_content)
            
            if summary:
                # Create GitHub Issue
                self._create_github_issue(site_key, site_info, summary)
                issues_created += 1
            
            # Save current content as new baseline
            self._save_content(site_key, current_content, current_hash)
        
        # Create error issue if any errors occurred
        if self.errors:
            logger.warning(f"\n⚠️  {len(self.errors)} error(s) occurred during monitoring")
            self._create_error_issue()
        
        # Summary
        if issues_created > 0:
            logger.info(f"\n✅ Created {issues_created} GitHub issue(s) for detected changes")
        else:
            logger.info("\n✅ No relevant changes detected across all monitored websites")
        
        logger.info("="*60)
        logger.info("Website check completed")
        logger.info("="*60)


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
