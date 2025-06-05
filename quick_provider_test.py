#!/usr/bin/env python3
"""
Simplified AI Provider Test - Tests all 4 configured providers
"""

import asyncio
import json
import time
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import openai
    import anthropic
    import google.generativeai as genai
    import requests  # for Ollama
    from shared_agents.config.multi_provider_config import AIProviderManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all required packages are installed")
    sys.exit(1)

class QuickProviderTest:
    """Quick test of all AI providers"""
    
    def __init__(self):
        self.manager = AIProviderManager()
        self.results = {}
    
    async def test_all_providers(self):
        """Test all configured providers"""
        print("🚀 Quick AI Provider Test")
        print("=" * 50)
        
        providers = ['openai', 'anthropic', 'gemini', 'ollama']
        
        for provider in providers:
            print(f"\n🧪 Testing {provider.upper()}...")
            try:
                result = await self.test_provider(provider)
                self.results[provider] = result
                if result.get('success'):
                    print(f"   ✅ {provider.upper()} - Working!")
                    print(f"   📝 Response: {result.get('response', '')[:50]}...")
                else:
                    print(f"   ❌ {provider.upper()} - Failed: {result.get('error')}")
            except Exception as e:
                print(f"   ❌ {provider.upper()} - Exception: {str(e)}")
                self.results[provider] = {'success': False, 'error': str(e)}
        
        # Summary
        print(f"\n📊 Test Results Summary:")
        working = sum(1 for r in self.results.values() if r.get('success'))
        total = len(self.results)
        print(f"   Working providers: {working}/{total}")
        
        # Save results
        with open('quick_provider_test_results.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': self.results,
                'summary': {'working': working, 'total': total}
            }, f, indent=2)
        
        return working >= 2
    
    async def test_provider(self, provider: str) -> dict:
        """Test a specific provider"""
        test_prompt = "Hello! Please respond with 'Test successful' to confirm you're working."
        
        try:
            if provider == 'openai':
                return await self.test_openai(test_prompt)
            elif provider == 'anthropic':
                return await self.test_anthropic(test_prompt)
            elif provider == 'gemini':
                return await self.test_gemini(test_prompt)
            elif provider == 'ollama':
                return await self.test_ollama(test_prompt)
            else:
                return {'success': False, 'error': f'Unknown provider: {provider}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_openai(self, prompt: str) -> dict:
        """Test OpenAI provider"""
        config = self.manager.get_provider_config('openai')
        if not config or not config.api_key:
            return {'success': False, 'error': 'No API key configured'}
        
        client = openai.OpenAI(api_key=config.api_key)
        
        response = client.chat.completions.create(
            model=config.model or "gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
        
        return {
            'success': True,
            'response': response.choices[0].message.content,
            'model': config.model
        }
    
    async def test_anthropic(self, prompt: str) -> dict:
        """Test Anthropic provider"""
        config = self.manager.get_provider_config('anthropic')
        if not config or not config.api_key:
            return {'success': False, 'error': 'No API key configured'}
        
        client = anthropic.Anthropic(api_key=config.api_key)
        
        response = client.messages.create(
            model=config.model or "claude-3-sonnet-20240229",
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            'success': True,
            'response': response.content[0].text,
            'model': config.model
        }
    
    async def test_gemini(self, prompt: str) -> dict:
        """Test Google Gemini provider"""
        config = self.manager.get_provider_config('gemini')
        if not config or not config.api_key:
            return {'success': False, 'error': 'No API key configured'}
        
        genai.configure(api_key=config.api_key)
        model = genai.GenerativeModel(config.model or 'gemini-pro')
        
        response = model.generate_content(prompt)
        
        return {
            'success': True,
            'response': response.text,
            'model': config.model
        }
    
    async def test_ollama(self, prompt: str) -> dict:
        """Test Ollama provider"""
        config = self.manager.get_provider_config('ollama')
        if not config or not config.base_url:
            return {'success': False, 'error': 'No base URL configured'}
        
        url = f"{config.base_url}/api/generate"
        data = {
            "model": config.model or "phi3.5",
            "prompt": prompt,
            "stream": False
        }
        
        response = requests.post(url, json=data, timeout=30)
        if response.status_code != 200:
            return {'success': False, 'error': f'HTTP {response.status_code}'}
        
        result = response.json()
        return {
            'success': True,
            'response': result.get('response', ''),
            'model': config.model
        }

async def main():
    """Main test function"""
    tester = QuickProviderTest()
    success = await tester.test_all_providers()
    return success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        print(f"\n🎉 Test {'PASSED' if success else 'FAILED'}")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)
