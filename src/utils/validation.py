"""Validation and sanitization utilities for the Web Scraper application."""
import re
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from urllib.parse import urlparse, urljoin

from ..config import SCRAPER_CONFIG

class ValidationError(ValueError):
    """Custom exception for validation errors."""
    pass

def is_valid_url(url: str, allowed_domains: Optional[List[str]] = None) -> Tuple[bool, str]:
    """Check if a URL is valid and optionally if it's from an allowed domain.
    
    Args:
        url: The URL to validate
        allowed_domains: Optional list of allowed domains (e.g., ['example.com'])
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url:
        return False, "URL cannot be empty"
    
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            return False, f"Invalid URL format: {url}"
            
        # Check if domain is allowed
        if allowed_domains:
            domain = parsed.netloc.lower()
            if not any(d.lower() in domain for d in allowed_domains):
                return False, f"Domain not allowed: {domain}"
                
        return True, ""
        
    except Exception as e:
        return False, f"URL validation error: {str(e)}"

def sanitize_input(
    value: Any, 
    field_type: type = str,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    allowed_values: Optional[list] = None,
    regex: Optional[str] = None,
    required: bool = True,
    default: Any = None,
    trim: bool = True
) -> Any:
    """Sanitize and validate input data.
    
    Args:
        value: The input value to sanitize
        field_type: Expected type (str, int, float, bool, list, dict)
        min_length: Minimum length for strings/lists
        max_length: Maximum length for strings/lists
        allowed_values: List of allowed values
        regex: Regular expression pattern to match against strings
        required: Whether the field is required
        default: Default value if field is empty and not required
        trim: Whether to trim whitespace from strings
        
    Returns:
        Sanitized value
        
    Raises:
        ValidationError: If validation fails
    """
    # Handle None/empty values
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise ValidationError("This field is required")
        return default
    
    # Convert to target type
    try:
        if field_type is str:
            if not isinstance(value, str):
                value = str(value)
            if trim:
                value = value.strip()
                
        elif field_type is int:
            value = int(value)
            
        elif field_type is float:
            value = float(value)
            
        elif field_type is bool:
            if isinstance(value, str):
                value = value.lower() in ('true', '1', 'yes', 'y', 't')
            else:
                value = bool(value)
                
        elif field_type is list:
            if not isinstance(value, list):
                if isinstance(value, str):
                    value = [item.strip() for item in value.split(',') if item.strip()]
                else:
                    value = list(value) if hasattr(value, '__iter__') else [value]
                    
        elif field_type is dict:
            if not isinstance(value, dict):
                raise ValidationError("Expected a dictionary")
                
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Invalid {field_type.__name__} value: {value}") from e
    
    # Validate length for strings and sequences
    if isinstance(value, (str, list)) and (min_length is not None or max_length is not None):
        length = len(value)
        if min_length is not None and length < min_length:
            raise ValidationError(f"Must be at least {min_length} characters")
        if max_length is not None and length > max_length:
            raise ValidationError(f"Must be at most {max_length} characters")
    
    # Validate against allowed values
    if allowed_values is not None and value not in allowed_values:
        raise ValidationError(f"Must be one of: {', '.join(map(str, allowed_values))}")
    
    # Validate against regex pattern
    if regex is not None and isinstance(value, str):
        if not re.match(regex, value):
            raise ValidationError("Invalid format")
    
    return value

def validate_config(config: Dict[str, Any], schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Validate a configuration dictionary against a schema.
    
    Args:
        config: Configuration dictionary to validate
        schema: Schema definition with field names as keys and validation rules as values
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ValidationError: If validation fails
    """
    validated = {}
    
    for field, rules in schema.items():
        value = config.get(field)
        field_type = rules.get('type', str)
        
        try:
            validated[field] = sanitize_input(
                value=value,
                field_type=field_type,
                min_length=rules.get('min_length'),
                max_length=rules.get('max_length'),
                allowed_values=rules.get('allowed_values'),
                regex=rules.get('regex'),
                required=rules.get('required', True),
                default=rules.get('default'),
                trim=rules.get('trim', True)
            )
        except ValidationError as e:
            raise ValidationError(f"{field}: {str(e)}") from e
    
    return validated

def validate_css_selector(selector: str) -> bool:
    """Validate a CSS selector.
    
    Args:
        selector: CSS selector to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not selector or not isinstance(selector, str):
        return False
        
    # Basic validation - can be expanded with more complex rules
    forbidden_patterns = [
        r'[\x00-\x1F\x7F]',  # Control characters
        r'[{}<>]',  # HTML/XML tags
        r'javascript:',  # JavaScript protocol
        r'data:',  # Data URLs
        r'expression\(',  # CSS expressions
    ]
    
    for pattern in forbidden_patterns:
        if re.search(pattern, selector, re.IGNORECASE):
            return False
            
    return True

def validate_html(html: str) -> bool:
    """Basic HTML validation.
    
    Args:
        html: HTML content to validate
        
    Returns:
        bool: True if HTML appears valid, False otherwise
    """
    if not html or not isinstance(html, str):
        return False
        
    # Check for basic HTML structure
    if not re.search(r'<[a-z][\s\S]*>', html, re.IGNORECASE):
        return False
        
    # Check for balanced tags (simplistic check)
    tags = re.findall(r'<(/?)(\w+)', html, re.IGNORECASE)
    stack = []
    
    for is_closing, tag in tags:
        if not is_closing:
            stack.append(tag.lower())
        elif stack and stack[-1] == tag.lower():
            stack.pop()
    
    return len(stack) == 0
