#!/usr/bin/env python3
"""
Quick configuration status test for all AI providers
Tests API key validity and basic connectivity
"""

import os
import sys
import asyncio
import json
from datetime import datetime

# Add project root to path
sys.path.append('/Users/cpconnor/projects/Unified-AI-Platform/Unified-AI-Platform')

from shared_agents.config.multi_provider_config import AIProviderManager

async def test_provider_config():
    """Test all provider configurations"""
    print("🔍 Testing AI Provider Configuration Status")
    print("=" * 60)
    
    # Initialize provider manager
    manager = AIProviderManager()
    
    # Get configuration summary
    summary = manager.get_config_summary()
    
    print(f"📊 Configuration Summary:")
    print(f"Enabled Providers: {summary['enabled_providers']}")
    print()
    
    # Test each provider
    providers = ['openai', 'anthropic', 'gemini', 'ollama']
    test_results = {}
    
    for provider in providers:
        print(f"🧪 Testing {provider.upper()}...")
        
        config = manager.get_provider_config(provider)
        if not config:
            print(f"   ❌ No configuration found")
            test_results[provider] = {'status': 'not_configured'}
            continue
        
        if not config.enabled:
            print(f"   ⚠️  Provider disabled")
            test_results[provider] = {'status': 'disabled'}
            continue
        
        # Check credentials
        if provider == 'ollama':
            # For Ollama, just check base URL
            print(f"   ✅ Base URL: {config.base_url}")
            print(f"   ✅ Model: {config.model}")
            test_results[provider] = {
                'status': 'configured',
                'base_url': config.base_url,
                'model': config.model
            }
        else:
            # For API-based providers, check API key
            if config.api_key:
                key_preview = config.api_key[:10] + "..." + config.api_key[-4:] if len(config.api_key) > 14 else "***"
                print(f"   ✅ API Key: {key_preview}")
                print(f"   ✅ Model: {config.model}")
                test_results[provider] = {
                    'status': 'configured',
                    'has_api_key': True,
                    'model': config.model
                }
            else:
                print(f"   ❌ Missing API key")
                test_results[provider] = {
                    'status': 'missing_key',
                    'has_api_key': False
                }
        
        print()
    
    # Test task assignments
    print("🎯 Task-Specific Provider Assignments:")
    task_types = ['code', 'logic', 'content', 'creative', 'general', 'local']
    for task in task_types:
        assigned = manager.get_provider_for_task(task)
        print(f"   {task.capitalize()}: {assigned}")
    
    print()
    print("📋 Validation Results:")
    validation = manager.validate_configuration()
    for provider, is_valid in validation.items():
        status = "✅" if is_valid else "❌"
        print(f"   {provider}: {status}")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'summary': summary,
        'test_results': test_results,
        'validation': validation,
        'task_assignments': {task: manager.get_provider_for_task(task) for task in task_types}
    }
    
    with open('config_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: config_test_results.json")
    
    # Return overall status
    configured_count = sum(1 for r in test_results.values() if r.get('status') == 'configured')
    print(f"\n🎉 Summary: {configured_count}/{len(providers)} providers configured")
    
    return configured_count >= 2  # Need at least 2 providers for good coverage

if __name__ == "__main__":
    success = asyncio.run(test_provider_config())
    sys.exit(0 if success else 1)
