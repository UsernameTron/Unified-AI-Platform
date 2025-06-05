"""
Enhanced configuration loader for multiple AI providers
Supports OpenAI, Anthropic, Gemini, Ollama, and additional services
"""

import os
import json
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class ProviderConfig:
    """Configuration for a single AI provider"""
    api_key: Optional[str] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    base_url: Optional[str] = None
    enabled: bool = True
    additional_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MultiProviderConfig:
    """Configuration for all AI providers"""
    openai: ProviderConfig = field(default_factory=ProviderConfig)
    anthropic: ProviderConfig = field(default_factory=ProviderConfig)
    gemini: ProviderConfig = field(default_factory=ProviderConfig)
    ollama: ProviderConfig = field(default_factory=ProviderConfig)
    tts_services: Dict[str, ProviderConfig] = field(default_factory=dict)
    vector_db: Dict[str, ProviderConfig] = field(default_factory=dict)
    additional_services: Dict[str, ProviderConfig] = field(default_factory=dict)

class AIProviderManager:
    """Manages configuration and initialization of multiple AI providers"""
    
    def __init__(self):
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self) -> MultiProviderConfig:
        """Load configuration from environment variables"""
        config = MultiProviderConfig()
        
        # OpenAI Configuration
        config.openai = ProviderConfig(
            api_key=os.getenv('OPENAI_API_KEY'),
            model=os.getenv('OPENAI_MODEL', 'gpt-4'),
            max_tokens=int(os.getenv('OPENAI_MAX_TOKENS', '2048')),
            enabled=bool(os.getenv('OPENAI_API_KEY'))
        )
        
        # Anthropic Configuration
        config.anthropic = ProviderConfig(
            api_key=os.getenv('ANTHROPIC_API_KEY'),
            model=os.getenv('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20241022'),
            max_tokens=int(os.getenv('ANTHROPIC_MAX_TOKENS', '4096')),
            enabled=bool(os.getenv('ANTHROPIC_API_KEY'))
        )
        
        # Google Gemini Configuration
        config.gemini = ProviderConfig(
            api_key=os.getenv('GOOGLE_API_KEY'),
            model=os.getenv('GEMINI_MODEL', 'gemini-1.5-pro'),
            max_tokens=int(os.getenv('GEMINI_MAX_TOKENS', '2048')),
            enabled=bool(os.getenv('GOOGLE_API_KEY'))
        )
        
        # Ollama Configuration
        config.ollama = ProviderConfig(
            base_url=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),
            model=os.getenv('OLLAMA_MODEL', 'phi3.5'),
            enabled=True  # Always enabled for local use
        )
        
        # TTS Services Configuration
        config.tts_services = {
            'elevenlabs': ProviderConfig(
                api_key=os.getenv('ELEVENLABS_API_KEY'),
                enabled=bool(os.getenv('ELEVENLABS_API_KEY'))
            ),
            'azure_speech': ProviderConfig(
                api_key=os.getenv('AZURE_SPEECH_KEY'),
                additional_params={
                    'region': os.getenv('AZURE_SPEECH_REGION')
                },
                enabled=bool(os.getenv('AZURE_SPEECH_KEY'))
            )
        }
        
        # Vector Database Configuration
        config.vector_db = {
            'pinecone': ProviderConfig(
                api_key=os.getenv('PINECONE_API_KEY'),
                additional_params={
                    'environment': os.getenv('PINECONE_ENVIRONMENT')
                },
                enabled=bool(os.getenv('PINECONE_API_KEY'))
            )
        }
        
        # Additional AI Services
        config.additional_services = {
            'cohere': ProviderConfig(
                api_key=os.getenv('COHERE_API_KEY'),
                enabled=bool(os.getenv('COHERE_API_KEY'))
            ),
            'huggingface': ProviderConfig(
                api_key=os.getenv('HUGGINGFACE_API_KEY'),
                enabled=bool(os.getenv('HUGGINGFACE_API_KEY'))
            )
        }
        
        return config
    
    def get_provider_config(self, provider: str) -> Optional[ProviderConfig]:
        """Get configuration for a specific provider"""
        provider_map = {
            'openai': self.config.openai,
            'anthropic': self.config.anthropic,
            'gemini': self.config.gemini,
            'ollama': self.config.ollama
        }
        
        if provider in provider_map:
            return provider_map[provider]
        
        # Check TTS services
        if provider in self.config.tts_services:
            return self.config.tts_services[provider]
        
        # Check vector databases
        if provider in self.config.vector_db:
            return self.config.vector_db[provider]
        
        # Check additional services
        if provider in self.config.additional_services:
            return self.config.additional_services[provider]
        
        return None
    
    def is_provider_enabled(self, provider: str) -> bool:
        """Check if a provider is enabled and properly configured"""
        config = self.get_provider_config(provider)
        return config is not None and config.enabled
    
    def get_enabled_providers(self) -> list:
        """Get list of all enabled providers"""
        enabled = []
        
        # Check main AI providers
        main_providers = ['openai', 'anthropic', 'gemini', 'ollama']
        for provider in main_providers:
            if self.is_provider_enabled(provider):
                enabled.append(provider)
        
        # Check TTS services
        for service_name in self.config.tts_services:
            if self.is_provider_enabled(service_name):
                enabled.append(f'tts_{service_name}')
        
        # Check vector databases
        for db_name in self.config.vector_db:
            if self.is_provider_enabled(db_name):
                enabled.append(f'vector_{db_name}')
        
        # Check additional services
        for service_name in self.config.additional_services:
            if self.is_provider_enabled(service_name):
                enabled.append(f'service_{service_name}')
        
        return enabled
    
    def get_provider_credentials(self, provider: str) -> Dict[str, Any]:
        """Get authentication credentials for a provider"""
        config = self.get_provider_config(provider)
        if not config:
            return {}
        
        credentials = {}
        if config.api_key:
            credentials['api_key'] = config.api_key
        if config.base_url:
            credentials['base_url'] = config.base_url
        
        # Add additional parameters
        credentials.update(config.additional_params)
        
        return credentials
    
    def get_model_config(self, provider: str) -> Dict[str, Any]:
        """Get model configuration for a provider"""
        config = self.get_provider_config(provider)
        if not config:
            return {}
        
        model_config = {}
        if config.model:
            model_config['model'] = config.model
        if config.max_tokens:
            model_config['max_tokens'] = config.max_tokens
        
        return model_config
    
    def validate_configuration(self) -> Dict[str, bool]:
        """Validate all provider configurations"""
        validation_results = {}
        
        # Validate main AI providers
        providers = ['openai', 'anthropic', 'gemini', 'ollama']
        for provider in providers:
            config = self.get_provider_config(provider)
            if config and config.enabled:
                # Check if required fields are present
                if provider == 'ollama':
                    # Ollama only needs base_url
                    validation_results[provider] = bool(config.base_url)
                else:
                    # Other providers need API key
                    validation_results[provider] = bool(config.api_key)
            else:
                validation_results[provider] = False
        
        return validation_results
    
    def get_provider_for_task(self, task_type: str) -> str:
        """Get the best provider for a specific task type"""
        task_preferences = {
            'code': 'anthropic',  # Claude is excellent for code
            'logic': 'anthropic',  # Claude is great for reasoning
            'content': 'gemini',   # Gemini is good for content creation
            'creative': 'gemini',  # Gemini for creative tasks
            'general': 'openai',   # OpenAI for general purpose
            'local': 'ollama',     # Ollama for local/offline use
            'fast': 'ollama',      # Ollama for quick responses
            'tts_high_quality': 'elevenlabs',  # ElevenLabs for high-quality TTS
            'tts_general': 'openai',           # OpenAI for general TTS
            'voice_high_quality': 'elevenlabs', # ElevenLabs for high-quality voice
            'voice_general': 'openai'           # OpenAI for general voice
        }
        
        preferred_provider = task_preferences.get(task_type, 'openai')
        
        # Check if preferred provider is available
        if self.is_provider_enabled(preferred_provider):
            return preferred_provider
        
        # Fallback to any available provider
        if task_type.startswith('tts_') or task_type.startswith('voice_'):
            # TTS fallback chain
            tts_providers = ['elevenlabs', 'openai']
            for provider in tts_providers:
                if self.is_provider_enabled(provider):
                    return provider
        
        # General fallback for text generation
        enabled_providers = [p for p in ['openai', 'anthropic', 'gemini', 'ollama'] 
                           if self.is_provider_enabled(p)]
        
        return enabled_providers[0] if enabled_providers else 'ollama'
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration"""
        return {
            'enabled_providers': self.get_enabled_providers(),
            'validation_results': self.validate_configuration(),
            'provider_configs': {
                'openai': {
                    'model': self.config.openai.model,
                    'enabled': self.config.openai.enabled
                },
                'anthropic': {
                    'model': self.config.anthropic.model,
                    'enabled': self.config.anthropic.enabled
                },
                'gemini': {
                    'model': self.config.gemini.model,
                    'enabled': self.config.gemini.enabled
                },
                'ollama': {
                    'model': self.config.ollama.model,
                    'base_url': self.config.ollama.base_url,
                    'enabled': self.config.ollama.enabled
                }
            }
        }

# Global instance
provider_manager = AIProviderManager()

def get_provider_manager() -> AIProviderManager:
    """Get the global provider manager instance"""
    return provider_manager

# Convenience functions for backward compatibility
def get_openai_config():
    """Get OpenAI configuration"""
    return provider_manager.get_provider_config('openai')

def get_anthropic_config():
    """Get Anthropic configuration"""
    return provider_manager.get_provider_config('anthropic')

def get_gemini_config():
    """Get Gemini configuration"""
    return provider_manager.get_provider_config('gemini')

def get_ollama_config():
    """Get Ollama configuration"""
    return provider_manager.get_provider_config('ollama')

if __name__ == "__main__":
    # Test the configuration
    manager = AIProviderManager()
    summary = manager.get_config_summary()
    print("AI Provider Configuration Summary:")
    print(json.dumps(summary, indent=2))
