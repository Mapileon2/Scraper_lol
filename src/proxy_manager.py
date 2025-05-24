"""
Proxy Manager for rotating proxies in web scraping operations.
This module provides functionality to manage and rotate between multiple proxies.
"""
import random
from typing import List, Dict, Optional, Union
import logging
from dataclasses import dataclass
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

@dataclass
class ProxyConfig:
    """Configuration for a proxy server."""
    http: str
    https: str
    username: Optional[str] = None
    password: Optional[str] = None
    fail_count: int = 0
    last_used: float = 0.0
    
    def to_dict(self) -> Dict[str, str]:
        """Convert proxy config to dictionary format for requests."""
        proxy_dict = {
            'http': self.http,
            'https': self.https
        }
        
        if self.username and self.password:
            proxy_dict['http'] = proxy_dict['http'].replace('//', f'//{self.username}:{self.password}@')
            proxy_dict['https'] = proxy_dict['https'].replace('//', f'//{self.username}:{self.password}@')
            
        return proxy_dict

class ProxyManager:
    """
    Manages a pool of proxy servers with rotation and failover capabilities.
    """
    
    def __init__(self, proxies: Optional[List[Union[Dict, str]]] = None, 
                 max_failures: int = 3, cooldown: int = 300):
        """
        Initialize the proxy manager.
        
        Args:
            proxies: List of proxy configurations (dict or str)
            max_failures: Maximum number of failures before a proxy is temporarily disabled
            cooldown: Cooldown period in seconds before a failed proxy can be used again
        """
        self.proxies: List[ProxyConfig] = []
        self.max_failures = max_failures
        self.cooldown = cooldown
        self.current_proxy_index = 0
        
        if proxies:
            self.add_proxies(proxies)
    
    def add_proxy(self, proxy: Union[Dict, str]) -> None:
        """Add a single proxy to the pool."""
        if isinstance(proxy, str):
            proxy = {'http': proxy, 'https': proxy}
            
        if 'http' not in proxy or 'https' not in proxy:
            logger.warning("Invalid proxy configuration. Must include 'http' and 'https' keys.")
            return
            
        self.proxies.append(ProxyConfig(
            http=proxy['http'],
            https=proxy['https'],
            username=proxy.get('username'),
            password=proxy.get('password')
        ))
    
    def add_proxies(self, proxies: List[Union[Dict, str]]) -> None:
        """Add multiple proxies to the pool."""
        for proxy in proxies:
            self.add_proxy(proxy)
    
    def get_proxy(self, strategy: str = 'round_robin') -> Optional[Dict[str, str]]:
        """
        Get the next available proxy based on the specified strategy.
        
        Args:
            strategy: Strategy for selecting the next proxy ('round_robin', 'random', 'least_used')
            
        Returns:
            Dictionary with proxy configuration or None if no proxies are available
        """
        if not self.proxies:
            return None
            
        # Filter out proxies that have exceeded max failures
        available_proxies = [
            p for p in self.proxies 
            if p.fail_count < self.max_failures or 
               (time.time() - p.last_used) > self.cooldown
        ]
        
        if not available_proxies:
            logger.warning("All proxies are temporarily unavailable")
            return None
        
        # Select proxy based on strategy
        if strategy == 'random':
            proxy = random.choice(available_proxies)
        elif strategy == 'least_used':
            proxy = min(available_proxies, key=lambda p: p.last_used)
        else:  # round_robin
            proxy = available_proxies[self.current_proxy_index % len(available_proxies)]
            self.current_proxy_index = (self.current_proxy_index + 1) % len(available_proxies)
        
        proxy.last_used = time.time()
        return proxy.to_dict()
    
    def mark_failure(self, proxy_config: Dict[str, str]) -> None:
        """Mark a proxy as failed."""
        for proxy in self.proxies:
            proxy_dict = proxy.to_dict()
            if (proxy_dict['http'] == proxy_config.get('http') and 
                proxy_dict['https'] == proxy_config.get('https')):
                proxy.fail_count += 1
                proxy.last_used = time.time()
                logger.warning(f"Proxy {proxy.http} marked as failed (failures: {proxy.fail_count})")
                break
    
    def mark_success(self, proxy_config: Dict[str, str]) -> None:
        """Mark a proxy as successful (reset failure count)."""
        for proxy in self.proxies:
            proxy_dict = proxy.to_dict()
            if (proxy_dict['http'] == proxy_config.get('http') and 
                proxy_dict['https'] == proxy_config.get('https')):
                if proxy.fail_count > 0:
                    proxy.fail_count = 0
                    logger.info(f"Proxy {proxy.http} marked as successful")
                break
    
    def get_session(self, proxy_config: Optional[Dict[str, str]] = None, 
                   retries: int = 3, backoff_factor: float = 0.5) -> requests.Session:
        """
        Get a requests Session configured with retry logic and optionally a proxy.
        
        Args:
            proxy_config: Proxy configuration to use (if None, no proxy is used)
            retries: Number of retries for failed requests
            backoff_factor: Backoff factor for retries
            
        Returns:
            Configured requests.Session object
        """
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[500, 502, 503, 504, 429],
            allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
        )
        
        # Mount the retry strategy to both http and https
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set proxy if provided
        if proxy_config:
            session.proxies.update(proxy_config)
            
        return session

def test_proxy(proxy_config: Dict[str, str], test_url: str = 'https://httpbin.org/ip', 
              timeout: int = 10) -> bool:
    """
    Test if a proxy is working by making a request to a test URL.
    
    Args:
        proxy_config: Proxy configuration to test
        test_url: URL to test the proxy against
        timeout: Request timeout in seconds
        
    Returns:
        True if the proxy is working, False otherwise
    """
    try:
        pm = ProxyManager()
        session = pm.get_session(proxy_config=proxy_config, retries=1)
        
        response = session.get(test_url, timeout=timeout)
        response.raise_for_status()
        
        # Verify the response contains the proxy IP
        if 'origin' in response.json():
            logger.info(f"Proxy test successful. Using IP: {response.json()['origin']}")
            return True
            
    except Exception as e:
        logger.warning(f"Proxy test failed: {str(e)}")
        
    return False
