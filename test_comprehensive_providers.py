#!/usr/bin/env python3
"""
Test all AI providers including TTS services
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
    import requests  # for Ollama and ElevenLabs
    from shared_agents.config.multi_provider_config import AIProviderManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class ComprehensiveProviderTest:
    """Test all 6 AI providers"""
    
    def __init__(self):
        self.manager = AIProviderManager()
        self.results = {}
    
    async def test_all_providers(self):
        """Test all configured providers"""
        print("🚀 Comprehensive AI Provider Test")
        print("=" * 60)
        
        # Test text generation providers
        text_providers = ['openai', 'anthropic', 'gemini', 'ollama']
        for provider in text_providers:
            await self.test_text_provider(provider)
        
        # Test TTS providers
        await self.test_elevenlabs_tts()
        
        # Test Hugging Face
        await self.test_huggingface()
        
        # Summary
        self.print_summary()
        
        # Save results
        with open('comprehensive_provider_test_results.json', 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': self.results
            }, f, indent=2)
        
        working = sum(1 for r in self.results.values() if r.get('success'))
        return working >= 4  # Need at least 4 working for success
    
    async def test_text_provider(self, provider: str):
        """Test text generation providers"""
        print(f"\n🧪 Testing {provider.upper()} (Text Generation)...")
        
        test_prompt = f"Please respond with: '{provider.upper()} is working correctly!'"
        
        try:
            if provider == 'openai':
                result = await self.test_openai(test_prompt)
            elif provider == 'anthropic':
                result = await self.test_anthropic(test_prompt)
            elif provider == 'gemini':
                result = await self.test_gemini(test_prompt)
            elif provider == 'ollama':
                result = await self.test_ollama(test_prompt)
            
            self.results[provider] = result
            
            if result.get('success'):
                print(f"   ✅ {provider.upper()} - Working!")
                print(f"   📝 Response: {result.get('response', '')[:80]}...")
            else:
                print(f"   ❌ {provider.upper()} - Failed: {result.get('error')}")
                
        except Exception as e:
            print(f"   ❌ {provider.upper()} - Exception: {str(e)}")
            self.results[provider] = {'success': False, 'error': str(e)}
    
    async def test_elevenlabs_tts(self):
        """Test ElevenLabs TTS"""
        print(f"\n🎤 Testing ELEVENLABS (High-Quality TTS)...")
        
        config = self.manager.get_provider_config('elevenlabs')
        if not config or not config.api_key:
            print("   ❌ No API key configured")
            self.results['elevenlabs'] = {'success': False, 'error': 'No API key'}
            return
        
        try:
            # Test ElevenLabs API by getting available voices
            headers = {
                'xi-api-key': config.api_key,
                'Content-Type': 'application/json'
            }
            
            response = requests.get(
                'https://api.elevenlabs.io/v1/voices',
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                voices = response.json()
                voice_count = len(voices.get('voices', []))
                print(f"   ✅ ELEVENLABS - Working!")
                print(f"   🎯 Available voices: {voice_count}")
                self.results['elevenlabs'] = {
                    'success': True,
                    'voice_count': voice_count,
                    'service_type': 'TTS (High-Quality)'
                }
            else:
                print(f"   ❌ ELEVENLABS - HTTP {response.status_code}")
                self.results['elevenlabs'] = {
                    'success': False,
                    'error': f'HTTP {response.status_code}'
                }
        
        except Exception as e:
            print(f"   ❌ ELEVENLABS - Exception: {str(e)}")
            self.results['elevenlabs'] = {'success': False, 'error': str(e)}
    
    async def test_huggingface(self):
        """Test Hugging Face API"""
        print(f"\n🤗 Testing HUGGING FACE (Open Source Models)...")
        
        config = self.manager.get_provider_config('huggingface')
        if not config or not config.api_key:
            print("   ❌ No API key configured")
            self.results['huggingface'] = {'success': False, 'error': 'No API key'}
            return
        
        try:
            # Test Hugging Face API by getting user info
            headers = {
                'Authorization': f'Bearer {config.api_key}'
            }
            
            response = requests.get(
                'https://huggingface.co/api/whoami',
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                user_info = response.json()
                username = user_info.get('name', 'Unknown')
                print(f"   ✅ HUGGING FACE - Working!")
                print(f"   👤 User: {username}")
                self.results['huggingface'] = {
                    'success': True,
                    'username': username,
                    'service_type': 'Open Source Models'
                }
            else:
                print(f"   ❌ HUGGING FACE - HTTP {response.status_code}")
                self.results['huggingface'] = {
                    'success': False,
                    'error': f'HTTP {response.status_code}'
                }
        
        except Exception as e:
            print(f"   ❌ HUGGING FACE - Exception: {str(e)}")
            self.results['huggingface'] = {'success': False, 'error': str(e)}
    
    async def test_openai(self, prompt: str) -> dict:
        """Test OpenAI provider"""
        config = self.manager.get_provider_config('openai')
        client = openai.OpenAI(api_key=config.api_key)
        
        response = client.chat.completions.create(
            model=config.model or "gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50
        )
        
        return {
            'success': True,
            'response': response.choices[0].message.content,
            'model': config.model,
            'service_type': 'General Purpose AI'
        }
    
    async def test_anthropic(self, prompt: str) -> dict:
        """Test Anthropic provider"""
        config = self.manager.get_provider_config('anthropic')
        client = anthropic.Anthropic(api_key=config.api_key)
        
        response = client.messages.create(
            model=config.model or "claude-3-sonnet-20240229",
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            'success': True,
            'response': response.content[0].text,
            'model': config.model,
            'service_type': 'Code & Logic AI'
        }
    
    async def test_gemini(self, prompt: str) -> dict:
        """Test Google Gemini provider"""
        config = self.manager.get_provider_config('gemini')
        genai.configure(api_key=config.api_key)
        model = genai.GenerativeModel(config.model or 'gemini-pro')
        
        response = model.generate_content(prompt)
        
        return {
            'success': True,
            'response': response.text,
            'model': config.model,
            'service_type': 'Content Creation AI'
        }
    
    async def test_ollama(self, prompt: str) -> dict:
        """Test Ollama provider"""
        config = self.manager.get_provider_config('ollama')
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
            'model': config.model,
            'service_type': 'Local AI'
        }
    
    def print_summary(self):
        """Print test summary"""
        print(f"\n📊 Comprehensive Test Results")
        print("=" * 60)
        
        working = 0
        total = len(self.results)
        
        for provider, result in self.results.items():
            status = "✅" if result.get('success') else "❌"
            service_type = result.get('service_type', 'Unknown')
            print(f"   {status} {provider.upper().ljust(12)} - {service_type}")
            if result.get('success'):
                working += 1
        
        print(f"\n🎯 Summary: {working}/{total} providers working")
        
        # TTS preference note
        print(f"\n🎵 TTS Provider Preferences:")
        print(f"   🎤 High-Quality Voice: ElevenLabs")
        print(f"   🔊 General TTS: OpenAI TTS")

async def main():
    """Main test function"""
    tester = ComprehensiveProviderTest()
    success = await tester.test_all_providers()
    return success

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        print(f"\n🎉 Comprehensive Test {'PASSED' if success else 'FAILED'}")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)
