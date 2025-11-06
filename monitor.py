#!/usr/bin/env python3
"""
AI-Powered Website Change Monitor
Creates GitHub Issues when product/API changes are detected.
"""

import os
import json
import hashlib
import logging
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
        
        # Websites to monitor
        self.websites = {
            "openai": {
                "url": "https://openai.com/news/product-releases/?display=list",
                "name": "OpenAI Product Releases",
                "selectors": ["main", "article", ".content"],
                "emoji": "🔵",
                "label": "openai"
            },
            "gemini": {
                "url": "https://ai.google.dev/gemini-api/docs/changelog",
                "name": "Google Gemini API Changelog",
                "selectors": ["main", "article", ".content"],
                "emoji": "🟡",
                "label": "gemini"
            },
            "anthropic": {
                "url": "https://www.anthropic.com/news",
                "name": "Anthropic News",
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
                "model": "gemini-2.5-flash",
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
    
    def _fetch_website_content(self, url: str, selectors: list) -> Optional[str]:
        """
        Fetch and extract clean text content from a website.
        
        Args:
            url: The URL to fetch
            selectors: List of CSS selectors to try (in order of preference)
        
        Returns:
            Clean text content or None if fetch fails
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
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
            return '\n'.join(lines)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error processing content from {url}: {e}")
            return None
    
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
- API deprecations or breaking changes
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

If you detect relevant changes, respond with a clear, concise summary in GitHub-flavored Markdown format:

### Summary
Brief 1-2 sentence overview of what changed.

### Changes Detected

#### [Category Name]
- **[Specific change]:** Brief description
- **[Specific change]:** Brief description

#### [Another Category]
- **[Specific change]:** Brief description

### Impact
Brief note on who/what this affects.

---

If NO relevant changes are detected, respond with exactly: "None"

Be strict - only report changes that would impact developers using these AI products/APIs."""

        user_message = f"""Website: {site_name}
Analyze these two versions and identify ONLY product/API changes as defined above.

=== PREVIOUS CONTENT ===
{old_content[:15000]}

=== CURRENT CONTENT ===
{new_content[:15000]}

=== END ===

Respond with "None" if no relevant changes, or a detailed summary in the format specified if changes detected."""

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
            
            # Check if LLM detected changes
            if response_text.lower() in ['none', 'false', 'no changes', 'no relevant changes']:
                logger.info(f"No relevant changes detected for {site_name}")
                return None
            
            logger.info(f"Changes detected for {site_name}")
            return response_text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed for {site_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error analyzing changes for {site_name}: {e}")
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
    
    def check_all_websites(self):
        """Main method to check all websites for changes."""
        logger.info("="*60)
        logger.info(f"Starting website check at {datetime.now()}")
        logger.info("="*60)
        
        issues_created = 0
        
        for site_key, site_info in self.websites.items():
            logger.info(f"\nChecking {site_info['name']}...")
            
            # Fetch current content
            current_content = self._fetch_website_content(site_info['url'], site_info['selectors'])
            
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