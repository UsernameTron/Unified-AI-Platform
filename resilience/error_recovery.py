"""
Enhanced Error Recovery and Circuit Breaker System for AI Gatekeeper
Provides resilient error handling, graceful degradation, and automatic recovery
"""

import time
import asyncio
import random
from typing import Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import logging
from functools import wraps

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered

@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    success_threshold: int = 3  # for half-open state
    timeout: int = 30  # request timeout

class CircuitBreaker:
    """Circuit breaker for external service calls."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.next_attempt_time = None
        
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        
        # Check if circuit is open and should remain closed
        if self.state == CircuitState.OPEN:
            if time.time() < self.next_attempt_time:
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
            else:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} moving to HALF_OPEN")
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else asyncio.create_task(asyncio.to_thread(func, *args, **kwargs)),
                timeout=self.config.timeout
            )
            
            # Handle success
            await self._on_success()
            return result
            
        except Exception as e:
            await self._on_failure()
            raise
    
    async def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker {self.name} moving to CLOSED")
        else:
            self.failure_count = 0
    
    async def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self.next_attempt_time = time.time() + self.config.recovery_timeout
            logger.warning(f"Circuit breaker {self.name} moving to OPEN (half-open failure)")
        
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.next_attempt_time = time.time() + self.config.recovery_timeout
            logger.warning(f"Circuit breaker {self.name} moving to OPEN (threshold reached)")

class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass

class RetryConfig:
    """Configuration for retry logic."""
    
    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

class RetryableError(Exception):
    """Base class for retryable errors."""
    pass

class NonRetryableError(Exception):
    """Base class for non-retryable errors."""
    pass

async def retry_with_backoff(func: Callable, config: RetryConfig, 
                           retryable_exceptions: tuple = (Exception,)) -> Any:
    """Retry function with exponential backoff."""
    
    last_exception = None
    
    for attempt in range(config.max_attempts):
        try:
            if asyncio.iscoroutinefunction(func):
                return await func()
            else:
                return func()
                
        except Exception as e:
            last_exception = e
            
            # Check if error is retryable
            if not isinstance(e, retryable_exceptions):
                raise NonRetryableError(f"Non-retryable error: {e}") from e
            
            # Don't wait after the last attempt
            if attempt == config.max_attempts - 1:
                break
            
            # Calculate delay
            delay = min(
                config.base_delay * (config.exponential_base ** attempt),
                config.max_delay
            )
            
            # Add jitter to prevent thundering herd
            if config.jitter:
                delay *= (0.5 + random.random() * 0.5)
            
            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
            await asyncio.sleep(delay)
    
    raise last_exception

class FallbackHandler:
    """Handles fallback responses when services fail."""
    
    def __init__(self):
        self.fallback_responses = {}
        self.cached_responses = {}
        
    def register_fallback(self, service_name: str, fallback_func: Callable):
        """Register a fallback function for a service."""
        self.fallback_responses[service_name] = fallback_func
    
    def cache_response(self, service_name: str, key: str, response: Any, ttl: int = 300):
        """Cache a successful response."""
        expiry = time.time() + ttl
        if service_name not in self.cached_responses:
            self.cached_responses[service_name] = {}
        self.cached_responses[service_name][key] = {
            'response': response,
            'expiry': expiry
        }
    
    def get_cached_response(self, service_name: str, key: str) -> Optional[Any]:
        """Get cached response if available and not expired."""
        if service_name not in self.cached_responses:
            return None
        
        cached = self.cached_responses[service_name].get(key)
        if not cached:
            return None
        
        if time.time() > cached['expiry']:
            del self.cached_responses[service_name][key]
            return None
        
        return cached['response']
    
    async def execute_fallback(self, service_name: str, *args, **kwargs) -> Any:
        """Execute fallback for a service."""
        if service_name not in self.fallback_responses:
            raise ValueError(f"No fallback registered for service: {service_name}")
        
        fallback_func = self.fallback_responses[service_name]
        if asyncio.iscoroutinefunction(fallback_func):
            return await fallback_func(*args, **kwargs)
        else:
            return fallback_func(*args, **kwargs)

class ResilientService:
    """Wrapper for external services with resilience patterns."""
    
    def __init__(self, name: str, circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
                 retry_config: Optional[RetryConfig] = None):
        self.name = name
        self.circuit_breaker = CircuitBreaker(name, circuit_breaker_config or CircuitBreakerConfig())
        self.retry_config = retry_config or RetryConfig()
        self.fallback_handler = FallbackHandler()
        
    def register_fallback(self, fallback_func: Callable):
        """Register fallback function."""
        self.fallback_handler.register_fallback(self.name, fallback_func)
    
    async def call(self, func: Callable, *args, use_cache: bool = True, 
                  cache_key: str = None, **kwargs) -> Any:
        """Call service with resilience patterns."""
        
        # Try to get cached response first
        if use_cache and cache_key:
            cached = self.fallback_handler.get_cached_response(self.name, cache_key)
            if cached:
                logger.info(f"Returning cached response for {self.name}:{cache_key}")
                return cached
        
        try:
            # Retry with circuit breaker
            async def wrapped_call():
                return await self.circuit_breaker.call(func, *args, **kwargs)
            
            result = await retry_with_backoff(
                wrapped_call, 
                self.retry_config,
                retryable_exceptions=(asyncio.TimeoutError, ConnectionError, RetryableError)
            )
            
            # Cache successful response
            if use_cache and cache_key:
                self.fallback_handler.cache_response(self.name, cache_key, result)
            
            return result
            
        except (CircuitBreakerOpenError, NonRetryableError) as e:
            logger.warning(f"Service {self.name} failed, trying fallback: {e}")
            
            # Try fallback
            try:
                return await self.fallback_handler.execute_fallback(self.name, *args, **kwargs)
            except Exception as fallback_error:
                logger.error(f"Fallback for {self.name} also failed: {fallback_error}")
                raise ServiceUnavailableError(f"Service {self.name} and fallback both failed") from e

class ServiceUnavailableError(Exception):
    """Raised when service and fallback both fail."""
    pass

class ErrorRecoveryManager:
    """Manages error recovery for the entire AI Gatekeeper system."""
    
    def __init__(self):
        self.services: Dict[str, ResilientService] = {}
        self.error_handlers: Dict[type, Callable] = {}
        
    def register_service(self, name: str, circuit_config: Optional[CircuitBreakerConfig] = None,
                        retry_config: Optional[RetryConfig] = None) -> ResilientService:
        """Register a resilient service."""
        service = ResilientService(name, circuit_config, retry_config)
        self.services[name] = service
        return service
    
    def register_error_handler(self, error_type: type, handler: Callable):
        """Register error handler for specific error types."""
        self.error_handlers[error_type] = handler
    
    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle error with registered handlers."""
        error_type = type(error)
        
        # Look for specific handler
        if error_type in self.error_handlers:
            return await self._execute_handler(self.error_handlers[error_type], error, context)
        
        # Look for parent class handlers
        for registered_type, handler in self.error_handlers.items():
            if issubclass(error_type, registered_type):
                return await self._execute_handler(handler, error, context)
        
        # Default error handling
        return await self._default_error_handler(error, context)
    
    async def _execute_handler(self, handler: Callable, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute error handler."""
        try:
            if asyncio.iscoroutinefunction(handler):
                return await handler(error, context)
            else:
                return handler(error, context)
        except Exception as handler_error:
            logger.error(f"Error handler failed: {handler_error}")
            return await self._default_error_handler(error, context)
    
    async def _default_error_handler(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Default error handler."""
        logger.error(f"Unhandled error in {context.get('operation', 'unknown')}: {error}")
        
        return {
            'success': False,
            'error': 'Service temporarily unavailable',
            'error_type': 'service_error',
            'retry_after': 60,
            'fallback_message': 'We apologize for the inconvenience. Please try again later or contact support.',
            'timestamp': datetime.now().isoformat()
        }

# Decorator for resilient function calls
def resilient(service_name: str, fallback_func: Optional[Callable] = None,
             circuit_config: Optional[CircuitBreakerConfig] = None,
             retry_config: Optional[RetryConfig] = None):
    """Decorator to make functions resilient."""
    
    def decorator(func):
        service = recovery_manager.register_service(service_name, circuit_config, retry_config)
        
        if fallback_func:
            service.register_fallback(fallback_func)
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = kwargs.pop('_cache_key', None)
            use_cache = kwargs.pop('_use_cache', True)
            
            return await service.call(func, *args, use_cache=use_cache, 
                                    cache_key=cache_key, **kwargs)
        
        return wrapper
    return decorator

# Global recovery manager
recovery_manager = ErrorRecoveryManager()

# Default fallback functions for AI Gatekeeper services
async def triage_fallback(*args, **kwargs):
    """Fallback for triage agent."""
    return {
        'confidence_score': 0.5,
        'risk_score': 0.5,
        'escalation_reason': 'AI service unavailable - using safe defaults',
        'recommended_agent': 'human_expert'
    }

async def research_fallback(*args, **kwargs):
    """Fallback for research agent."""
    issue = kwargs.get('query', args[0] if args else 'Unknown issue')
    
    return {
        'title': 'Basic Support Guidance',
        'summary': 'AI assistance is currently unavailable. Please follow these general steps or contact human support.',
        'steps': [
            {
                'title': 'Document the Issue',
                'description': 'Take note of any error messages, steps that led to the problem, and your system information.',
                'troubleshooting': 'Screenshots can be helpful for visual issues.'
            },
            {
                'title': 'Try Basic Troubleshooting',
                'description': 'Restart the application, clear cache, or try again after a few minutes.',
                'troubleshooting': 'Many issues resolve themselves with a simple restart.'
            },
            {
                'title': 'Contact Human Support',
                'description': 'If the issue persists, please contact our support team with the details you documented.',
                'troubleshooting': 'Provide as much detail as possible for faster resolution.'
            }
        ],
        'estimated_time': '5-10 minutes',
        'confidence': 0.6,
        'fallback': True
    }

# Initialize default services
def initialize_error_recovery():
    """Initialize error recovery with default configurations."""
    
    # Configure AI service with circuit breaker
    ai_service = recovery_manager.register_service(
        'ai_service',
        CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30),
        RetryConfig(max_attempts=2, base_delay=1.0)
    )
    ai_service.register_fallback(research_fallback)
    
    # Configure triage service
    triage_service = recovery_manager.register_service(
        'triage_service',
        CircuitBreakerConfig(failure_threshold=3, recovery_timeout=30),
        RetryConfig(max_attempts=2)
    )
    triage_service.register_fallback(triage_fallback)
    
    # Configure search service
    search_service = recovery_manager.register_service(
        'search_service',
        CircuitBreakerConfig(failure_threshold=5, recovery_timeout=60),
        RetryConfig(max_attempts=3, base_delay=0.5)
    )
    
    logger.info("Error recovery services initialized")

# Export for use in main application
__all__ = [
    'recovery_manager',
    'resilient',
    'CircuitBreakerConfig',
    'RetryConfig',
    'ResilientService',
    'initialize_error_recovery',
    'RetryableError',
    'NonRetryableError',
    'ServiceUnavailableError'
]