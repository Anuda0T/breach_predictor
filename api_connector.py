"""
API Connector Framework
Phase 1: Data Architecture Foundation

Provides a robust, extensible framework for connecting to various data sources
with authentication, rate limiting, retry logic, and error handling.
"""

import requests
import time
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
from enum import Enum
import hashlib
import hmac
import base64
from urllib.parse import urlencode, quote
import aiohttp
import asyncio
from concurrent.futures import ThreadPoolExecutor
import backoff

class AuthType(Enum):
    """Authentication types supported"""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    HMAC = "hmac"
    CUSTOM = "custom"

class RequestMethod(Enum):
    """HTTP request methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"

@dataclass
class RateLimit:
    """Rate limiting configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10

@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    backoff_factor: float = 2.0
    max_delay: int = 60
    retry_on_status_codes: List[int] = field(default_factory=lambda: [429, 500, 502, 503, 504])

@dataclass
class APIConfig:
    """API configuration"""
    base_url: str
    auth_type: AuthType = AuthType.NONE
    auth_credentials: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    rate_limit: RateLimit = field(default_factory=RateLimit)
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    user_agent: str = "BreachPredictor/1.0"

@dataclass
class APIResponse:
    """Standardized API response"""
    success: bool
    status_code: int
    data: Any = None
    error_message: str = ""
    response_time: float = 0.0
    rate_limit_remaining: int = -1
    rate_limit_reset: int = -1

class RateLimiter:
    """Thread-safe rate limiter"""

    def __init__(self, rate_limit: RateLimit):
        self.rate_limit = rate_limit
        self.requests = []
        self.lock = threading.Lock()

    def can_make_request(self) -> bool:
        """Check if a request can be made within rate limits"""
        with self.lock:
            now = datetime.now()

            # Clean old requests
            cutoff_minute = now - timedelta(minutes=1)
            cutoff_hour = now - timedelta(hours=1)

            self.requests = [
                req_time for req_time in self.requests
                if req_time > cutoff_hour
            ]

            # Check limits
            requests_last_minute = sum(1 for req in self.requests if req > cutoff_minute)
            requests_last_hour = len(self.requests)

            return (requests_last_minute < self.rate_limit.requests_per_minute and
                   requests_last_hour < self.rate_limit.requests_per_hour)

    def record_request(self):
        """Record a request timestamp"""
        with self.lock:
            self.requests.append(datetime.now())

class BaseAPIConnector(ABC):
    """
    Abstract base class for API connectors
    """

    def __init__(self, config: APIConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.rate_limiter = RateLimiter(config.rate_limit)
        self.session = requests.Session()
        self._setup_session()

    def _setup_session(self):
        """Setup HTTP session with default headers"""
        self.session.headers.update({
            'User-Agent': self.config.user_agent,
            **self.config.headers
        })

    @abstractmethod
    def authenticate_request(self, request: requests.Request) -> requests.Request:
        """
        Apply authentication to a request

        Args:
            request: The request to authenticate

        Returns:
            Authenticated request
        """
        pass

    def make_request(self,
                    method: RequestMethod,
                    endpoint: str,
                    params: Dict[str, Any] = None,
                    data: Any = None,
                    json_data: Dict[str, Any] = None,
                    headers: Dict[str, str] = None) -> APIResponse:
        """
        Make an authenticated API request with rate limiting and retries

        Args:
            method: HTTP method
            endpoint: API endpoint (relative to base URL)
            params: Query parameters
            data: Request body data
            json_data: JSON request body
            headers: Additional headers

        Returns:
            APIResponse object
        """
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        # Prepare request
        request_kwargs = {
            'method': method.value,
            'url': url,
            'timeout': self.config.timeout
        }

        if params:
            request_kwargs['params'] = params
        if data:
            request_kwargs['data'] = data
        if json_data:
            request_kwargs['json'] = json_data
        if headers:
            request_kwargs['headers'] = {**self.session.headers, **headers}

        # Apply authentication
        req = requests.Request(**request_kwargs)
        authenticated_req = self.authenticate_request(req)

        return self._execute_with_retry(authenticated_req)

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, requests.exceptions.Timeout),
        max_tries=lambda: RetryConfig().max_attempts,
        giveup=lambda e: isinstance(e, requests.exceptions.HTTPError) and
                        e.response.status_code not in RetryConfig().retry_on_status_codes
    )
    def _execute_with_retry(self, request: requests.Request) -> APIResponse:
        """Execute request with retry logic"""
        start_time = time.time()

        # Check rate limit
        if not self.rate_limiter.can_make_request():
            return APIResponse(
                success=False,
                status_code=429,
                error_message="Rate limit exceeded",
                response_time=time.time() - start_time
            )

        try:
            # Prepare request
            prepared_request = self.session.prepare_request(request)

            # Make request
            response = self.session.send(prepared_request, timeout=self.config.timeout)

            # Record request
            self.rate_limiter.record_request()

            # Parse response
            response_time = time.time() - start_time

            # Check for rate limit headers
            rate_limit_remaining = self._extract_rate_limit_header(response, 'X-RateLimit-Remaining')
            rate_limit_reset = self._extract_rate_limit_header(response, 'X-RateLimit-Reset')

            if response.status_code >= 400:
                error_message = self._extract_error_message(response)
                return APIResponse(
                    success=False,
                    status_code=response.status_code,
                    error_message=error_message,
                    response_time=response_time,
                    rate_limit_remaining=rate_limit_remaining,
                    rate_limit_reset=rate_limit_reset
                )

            # Try to parse JSON response
            try:
                data = response.json()
            except ValueError:
                data = response.text

            return APIResponse(
                success=True,
                status_code=response.status_code,
                data=data,
                response_time=response_time,
                rate_limit_remaining=rate_limit_remaining,
                rate_limit_reset=rate_limit_reset
            )

        except requests.exceptions.RequestException as e:
            response_time = time.time() - start_time
            return APIResponse(
                success=False,
                status_code=0,
                error_message=str(e),
                response_time=response_time
            )

    def _extract_rate_limit_header(self, response: requests.Response, header_name: str) -> int:
        """Extract rate limit information from response headers"""
        try:
            return int(response.headers.get(header_name, -1))
        except (ValueError, TypeError):
            return -1

    def _extract_error_message(self, response: requests.Response) -> str:
        """Extract error message from response"""
        try:
            error_data = response.json()
            if isinstance(error_data, dict):
                return error_data.get('message', error_data.get('error', response.text))
            return str(error_data)
        except (ValueError, AttributeError):
            return response.text

    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limiting status"""
        return {
            'can_make_request': self.rate_limiter.can_make_request(),
            'config': self.config.rate_limit.__dict__,
            'recent_requests': len([
                req for req in self.rate_limiter.requests
                if req > datetime.now() - timedelta(minutes=1)
            ])
        }

class APIKeyConnector(BaseAPIConnector):
    """API connector with API key authentication"""

    def authenticate_request(self, request: requests.Request) -> requests.Request:
        if self.config.auth_type == AuthType.API_KEY:
            api_key = self.config.auth_credentials.get('api_key')
            key_name = self.config.auth_credentials.get('key_name', 'X-API-Key')

            if api_key:
                request.headers[key_name] = api_key

        return request

class BearerTokenConnector(BaseAPIConnector):
    """API connector with Bearer token authentication"""

    def authenticate_request(self, request: requests.Request) -> requests.Request:
        if self.config.auth_type == AuthType.BEARER_TOKEN:
            token = self.config.auth_credentials.get('token')

            if token:
                request.headers['Authorization'] = f"Bearer {token}"

        return request

class BasicAuthConnector(BaseAPIConnector):
    """API connector with Basic authentication"""

    def authenticate_request(self, request: requests.Request) -> requests.Request:
        if self.config.auth_type == AuthType.BASIC_AUTH:
            username = self.config.auth_credentials.get('username')
            password = self.config.auth_credentials.get('password')

            if username and password:
                import base64
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                request.headers['Authorization'] = f"Basic {credentials}"

        return request

class OAuth2Connector(BaseAPIConnector):
    """API connector with OAuth2 authentication"""

    def __init__(self, config: APIConfig):
        super().__init__(config)
        self.access_token = None
        self.token_expires_at = None

    def authenticate_request(self, request: requests.Request) -> requests.Request:
        if self.config.auth_type == AuthType.OAUTH2:
            token = self._get_valid_token()
            if token:
                request.headers['Authorization'] = f"Bearer {token}"

        return request

    def _get_valid_token(self) -> Optional[str]:
        """Get a valid access token, refreshing if necessary"""
        if self.access_token and self.token_expires_at:
            if datetime.now() < self.token_expires_at:
                return self.access_token

        # Token expired or missing, get new one
        return self._refresh_token()

    def _refresh_token(self) -> Optional[str]:
        """Refresh OAuth2 access token"""
        try:
            token_url = self.config.auth_credentials.get('token_url')
            client_id = self.config.auth_credentials.get('client_id')
            client_secret = self.config.auth_credentials.get('client_secret')
            grant_type = self.config.auth_credentials.get('grant_type', 'client_credentials')

            if not all([token_url, client_id, client_secret]):
                self.logger.error("Missing OAuth2 credentials")
                return None

            data = {
                'grant_type': grant_type,
                'client_id': client_id,
                'client_secret': client_secret
            }

            response = requests.post(token_url, data=data, timeout=30)

            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get('access_token')
                expires_in = token_data.get('expires_in', 3600)
                self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)

                return self.access_token
            else:
                self.logger.error(f"OAuth2 token refresh failed: {response.status_code}")
                return None

        except Exception as e:
            self.logger.error(f"OAuth2 token refresh error: {str(e)}")
            return None

class HMACConnector(BaseAPIConnector):
    """API connector with HMAC authentication"""

    def authenticate_request(self, request: requests.Request) -> requests.Request:
        if self.config.auth_type == AuthType.HMAC:
            secret_key = self.config.auth_credentials.get('secret_key')
            api_key = self.config.auth_credentials.get('api_key')
            algorithm = self.config.auth_credentials.get('algorithm', 'sha256')

            if secret_key and api_key:
                timestamp = str(int(time.time()))
                method = request.method
                path = request.url.replace(self.config.base_url, '')

                # Create signature
                message = f"{timestamp}{method}{path}"
                if request.body:
                    message += request.body.decode('utf-8') if isinstance(request.body, bytes) else str(request.body)

                signature = hmac.new(
                    secret_key.encode(),
                    message.encode(),
                    getattr(hashlib, algorithm)
                ).hexdigest()

                # Add headers
                request.headers['X-API-Key'] = api_key
                request.headers['X-Timestamp'] = timestamp
                request.headers['X-Signature'] = signature

        return request

class AsyncAPIConnector:
    """
    Asynchronous API connector for high-throughput data collection
    """

    def __init__(self, config: APIConfig, max_concurrent: int = 10):
        self.config = config
        self.max_concurrent = max_concurrent
        self.logger = logging.getLogger(__name__)
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def make_async_request(self,
                                method: RequestMethod,
                                endpoint: str,
                                params: Dict[str, Any] = None,
                                json_data: Dict[str, Any] = None,
                                headers: Dict[str, str] = None) -> APIResponse:
        """
        Make asynchronous API request
        """
        async with self.semaphore:
            url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

            request_data = {
                'method': method.value,
                'url': url,
                'timeout': aiohttp.ClientTimeout(total=self.config.timeout)
            }

            if params:
                request_data['params'] = params
            if json_data:
                request_data['json'] = json_data
            if headers:
                request_data['headers'] = headers

            start_time = time.time()

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(**request_data) as response:
                        response_time = time.time() - start_time

                        if response.status >= 400:
                            error_text = await response.text()
                            return APIResponse(
                                success=False,
                                status_code=response.status,
                                error_message=error_text,
                                response_time=response_time
                            )

                        try:
                            data = await response.json()
                        except aiohttp.ContentTypeError:
                            data = await response.text()

                        return APIResponse(
                            success=True,
                            status_code=response.status,
                            data=data,
                            response_time=response_time
                        )

            except Exception as e:
                response_time = time.time() - start_time
                return APIResponse(
                    success=False,
                    status_code=0,
                    error_message=str(e),
                    response_time=response_time
                )

class APIConnectorFactory:
    """Factory for creating API connectors"""

    @staticmethod
    def create_connector(config: APIConfig) -> BaseAPIConnector:
        """Create appropriate connector based on auth type"""
        connector_map = {
            AuthType.NONE: BaseAPIConnector,
            AuthType.API_KEY: APIKeyConnector,
            AuthType.BEARER_TOKEN: BearerTokenConnector,
            AuthType.BASIC_AUTH: BasicAuthConnector,
            AuthType.OAUTH2: OAuth2Connector,
            AuthType.HMAC: HMACConnector,
        }

        connector_class = connector_map.get(config.auth_type, BaseAPIConnector)
        return connector_class(config)

# Convenience functions
def create_api_connector(base_url: str, auth_type: AuthType = AuthType.NONE, **kwargs) -> BaseAPIConnector:
    """Create an API connector with minimal configuration"""
    config = APIConfig(base_url=base_url, auth_type=auth_type, **kwargs)
    return APIConnectorFactory.create_connector(config)

def test_api_connection(connector: BaseAPIConnector, test_endpoint: str = "") -> bool:
    """Test API connection"""
    try:
        response = connector.make_request(RequestMethod.GET, test_endpoint or "")
        return response.success or response.status_code in [200, 201, 202]
    except Exception:
        return False

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create API key connector
    config = APIConfig(
        base_url="https://api.example.com",
        auth_type=AuthType.API_KEY,
        auth_credentials={"api_key": "your-api-key"}
    )

    connector = APIConnectorFactory.create_connector(config)

    # Make a test request
    response = connector.make_request(RequestMethod.GET, "/test-endpoint")

    print(f"Request successful: {response.success}")
    print(f"Status code: {response.status_code}")
    print(f"Response time: {response.response_time:.2f}s")

    if response.success:
        print(f"Data: {response.data}")
    else:
        print(f"Error: {response.error_message}")
