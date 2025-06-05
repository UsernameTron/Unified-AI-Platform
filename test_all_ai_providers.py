#!/usr/bin/env python3
"""
Comprehensive AI Provider Testing Script
Tests all configured AI providers with specialized use cases
"""

import asyncio
import json
import logging
import time
import sys
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shared_agents.config.multi_provider_config import AIProviderManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AIProviderTester:
    """Comprehensive tester for all AI providers"""
    
    def __init__(self):
        self.provider_manager = AIProviderManager()
        self.test_results = {}
        self.start_time = datetime.now()
        
        # Import AI libraries dynamically
        self.providers = {}
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize all available AI provider clients"""
        
        # OpenAI
        if self.provider_manager.is_provider_enabled('openai'):
            try:
                import openai
                config = self.provider_manager.get_provider_config('openai')
                self.providers['openai'] = openai.OpenAI(api_key=config.api_key)
                logger.info("✓ OpenAI client initialized")
            except ImportError:
                logger.warning("OpenAI library not installed")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
        
        # Anthropic
        if self.provider_manager.is_provider_enabled('anthropic'):
            try:
                import anthropic
                config = self.provider_manager.get_provider_config('anthropic')
                self.providers['anthropic'] = anthropic.Anthropic(api_key=config.api_key)
                logger.info("✓ Anthropic client initialized")
            except ImportError:
                logger.warning("Anthropic library not installed")
            except Exception as e:
                logger.error(f"Failed to initialize Anthropic: {e}")
        
        # Google Gemini
        if self.provider_manager.is_provider_enabled('gemini'):
            try:
                import google.generativeai as genai
                config = self.provider_manager.get_provider_config('gemini')
                genai.configure(api_key=config.api_key)
                self.providers['gemini'] = genai.GenerativeModel(config.model)
                logger.info("✓ Gemini client initialized")
            except ImportError:
                logger.warning("Google Generative AI library not installed")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
        
        # Ollama
        if self.provider_manager.is_provider_enabled('ollama'):
            try:
                import ollama
                config = self.provider_manager.get_provider_config('ollama')
                # Test connection to Ollama server
                response = ollama.list()
                self.providers['ollama'] = ollama
                logger.info("✓ Ollama client initialized")
            except ImportError:
                logger.warning("Ollama library not installed")
            except Exception as e:
                logger.error(f"Failed to initialize Ollama: {e}")
    
    async def test_openai(self) -> Dict[str, Any]:
        """Test OpenAI GPT models"""
        if 'openai' not in self.providers:
            return {'status': 'skipped', 'reason': 'OpenAI not configured'}
        
        try:
            client = self.providers['openai']
            config = self.provider_manager.get_provider_config('openai')
            
            # Test 1: General reasoning
            response1 = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "Explain quantum computing in simple terms."}
                ],
                max_tokens=200
            )
            
            # Test 2: Code generation
            response2 = client.chat.completions.create(
                model=config.model,
                messages=[
                    {"role": "system", "content": "You are a Python programming expert."},
                    {"role": "user", "content": "Write a Python function to calculate Fibonacci numbers."}
                ],
                max_tokens=300
            )
            
            return {
                'status': 'success',
                'model': config.model,
                'tests': {
                    'general_reasoning': {
                        'prompt': 'Explain quantum computing in simple terms.',
                        'response_length': len(response1.choices[0].message.content),
                        'tokens_used': response1.usage.total_tokens if response1.usage else 0
                    },
                    'code_generation': {
                        'prompt': 'Write a Python function to calculate Fibonacci numbers.',
                        'response_length': len(response2.choices[0].message.content),
                        'tokens_used': response2.usage.total_tokens if response2.usage else 0
                    }
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def test_anthropic(self) -> Dict[str, Any]:
        """Test Anthropic Claude models"""
        if 'anthropic' not in self.providers:
            return {'status': 'skipped', 'reason': 'Anthropic not configured'}
        
        try:
            client = self.providers['anthropic']
            config = self.provider_manager.get_provider_config('anthropic')
            
            # Test 1: Complex reasoning
            response1 = client.messages.create(
                model=config.model,
                max_tokens=400,
                messages=[
                    {"role": "user", "content": "Analyze the pros and cons of renewable energy adoption from economic, environmental, and social perspectives."}
                ]
            )
            
            # Test 2: Code analysis and improvement
            response2 = client.messages.create(
                model=config.model,
                max_tokens=500,
                messages=[
                    {"role": "user", "content": """Review this Python code and suggest improvements:

def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] > 0:
            result.append(data[i] * 2)
    return result"""}
                ]
            )
            
            return {
                'status': 'success',
                'model': config.model,
                'tests': {
                    'complex_reasoning': {
                        'prompt': 'Analyze renewable energy adoption',
                        'response_length': len(response1.content[0].text),
                        'tokens_used': response1.usage.input_tokens + response1.usage.output_tokens
                    },
                    'code_analysis': {
                        'prompt': 'Review and improve Python code',
                        'response_length': len(response2.content[0].text),
                        'tokens_used': response2.usage.input_tokens + response2.usage.output_tokens
                    }
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def test_gemini(self) -> Dict[str, Any]:
        """Test Google Gemini models"""
        if 'gemini' not in self.providers:
            return {'status': 'skipped', 'reason': 'Gemini not configured'}
        
        try:
            model = self.providers['gemini']
            config = self.provider_manager.get_provider_config('gemini')
            
            # Test 1: Creative content generation
            response1 = model.generate_content(
                "Write a creative short story about a robot who discovers emotions.",
                generation_config={'max_output_tokens': 400}
            )
            
            # Test 2: Data analysis and visualization suggestions
            response2 = model.generate_content(
                "Given a dataset of customer purchases, suggest 5 different ways to visualize the data and explain what insights each visualization would provide.",
                generation_config={'max_output_tokens': 500}
            )
            
            return {
                'status': 'success',
                'model': config.model,
                'tests': {
                    'creative_content': {
                        'prompt': 'Write a creative short story about a robot',
                        'response_length': len(response1.text),
                        'tokens_estimated': len(response1.text) // 4  # Rough estimate
                    },
                    'data_analysis': {
                        'prompt': 'Suggest data visualization methods',
                        'response_length': len(response2.text),
                        'tokens_estimated': len(response2.text) // 4
                    }
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def test_ollama(self) -> Dict[str, Any]:
        """Test Ollama local models"""
        if 'ollama' not in self.providers:
            return {'status': 'skipped', 'reason': 'Ollama not configured'}
        
        try:
            client = self.providers['ollama']
            config = self.provider_manager.get_provider_config('ollama')
            
            # Test 1: Quick response
            response1 = client.chat(
                model=config.model,
                messages=[
                    {'role': 'user', 'content': 'What is machine learning in one paragraph?'}
                ]
            )
            
            # Test 2: Local code assistance
            response2 = client.chat(
                model=config.model,
                messages=[
                    {'role': 'user', 'content': 'Write a simple Python function to reverse a string.'}
                ]
            )
            
            return {
                'status': 'success',
                'model': config.model,
                'base_url': config.base_url,
                'tests': {
                    'quick_response': {
                        'prompt': 'What is machine learning?',
                        'response_length': len(response1['message']['content']),
                        'local_processing': True
                    },
                    'code_assistance': {
                        'prompt': 'Write a function to reverse a string',
                        'response_length': len(response2['message']['content']),
                        'local_processing': True
                    }
                }
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def test_all_providers(self) -> Dict[str, Any]:
        """Run comprehensive tests on all providers"""
        logger.info("Starting comprehensive AI provider tests...")
        
        # Run tests for each provider
        test_tasks = []
        
        if 'openai' in self.providers:
            test_tasks.append(('openai', self.test_openai()))
        
        if 'anthropic' in self.providers:
            test_tasks.append(('anthropic', self.test_anthropic()))
        
        if 'gemini' in self.providers:
            test_tasks.append(('gemini', self.test_gemini()))
        
        if 'ollama' in self.providers:
            test_tasks.append(('ollama', self.test_ollama()))
        
        # Execute tests
        results = {}
        for provider_name, task in test_tasks:
            logger.info(f"Testing {provider_name}...")
            try:
                start_time = time.time()
                result = await task
                end_time = time.time()
                
                result['execution_time'] = end_time - start_time
                results[provider_name] = result
                
                if result['status'] == 'success':
                    logger.info(f"✓ {provider_name} tests completed successfully")
                else:
                    logger.warning(f"⚠ {provider_name} tests failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"✗ {provider_name} test failed with exception: {e}")
                results[provider_name] = {'status': 'error', 'error': str(e)}
        
        return results
    
    def test_provider_selection(self) -> Dict[str, str]:
        """Test the provider selection for different task types"""
        task_types = ['code', 'logic', 'content', 'creative', 'general', 'local', 'fast']
        selections = {}
        
        for task_type in task_types:
            selected = self.provider_manager.get_provider_for_task(task_type)
            selections[task_type] = selected
            logger.info(f"Task '{task_type}' → Provider '{selected}'")
        
        return selections
    
    def generate_test_report(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive test report"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Count successes and failures
        successful_providers = [p for p, r in test_results.items() if r.get('status') == 'success']
        failed_providers = [p for p, r in test_results.items() if r.get('status') == 'error']
        skipped_providers = [p for p, r in test_results.items() if r.get('status') == 'skipped']
        
        # Generate summary
        summary = {
            'test_overview': {
                'total_duration_seconds': duration,
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'providers_tested': len(test_results),
                'successful_tests': len(successful_providers),
                'failed_tests': len(failed_providers),
                'skipped_tests': len(skipped_providers)
            },
            'provider_status': {
                'successful': successful_providers,
                'failed': failed_providers,
                'skipped': skipped_providers
            },
            'provider_selection_strategy': self.test_provider_selection(),
            'configuration_summary': self.provider_manager.get_config_summary(),
            'detailed_results': test_results,
            'recommendations': self._generate_recommendations(test_results)
        }
        
        return summary
    
    def _generate_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check for missing API keys
        for provider, result in test_results.items():
            if result.get('status') == 'skipped' and 'not configured' in result.get('reason', ''):
                recommendations.append(f"Configure {provider} API key to enable additional AI capabilities")
        
        # Check for failed tests
        failed_providers = [p for p, r in test_results.items() if r.get('status') == 'error']
        if failed_providers:
            recommendations.append(f"Investigate issues with: {', '.join(failed_providers)}")
        
        # Performance recommendations
        ollama_working = test_results.get('ollama', {}).get('status') == 'success'
        if ollama_working:
            recommendations.append("Ollama is working well for local/private AI processing")
        
        # Provider-specific recommendations
        if test_results.get('anthropic', {}).get('status') == 'success':
            recommendations.append("Use Anthropic Claude for complex reasoning and code analysis tasks")
        
        if test_results.get('gemini', {}).get('status') == 'success':
            recommendations.append("Use Google Gemini for creative content generation and data analysis")
        
        if test_results.get('openai', {}).get('status') == 'success':
            recommendations.append("Use OpenAI for general-purpose AI tasks and proven reliability")
        
        if not recommendations:
            recommendations.append("All configured providers are working correctly!")
        
        return recommendations

async def main():
    """Main test execution function"""
    print("🚀 Starting Comprehensive AI Provider Testing...")
    print("=" * 60)
    
    # Initialize tester
    tester = AIProviderTester()
    
    # Display configuration summary
    config_summary = tester.provider_manager.get_config_summary()
    print(f"\n📊 Configuration Summary:")
    print(f"Enabled Providers: {', '.join(config_summary['enabled_providers'])}")
    print(f"Validation Results: {config_summary['validation_results']}")
    
    # Run comprehensive tests
    print(f"\n🧪 Running Tests...")
    test_results = await tester.test_all_providers()
    
    # Generate and display report
    report = tester.generate_test_report(test_results)
    
    print(f"\n📋 Test Results Summary:")
    print(f"Total Duration: {report['test_overview']['total_duration_seconds']:.2f} seconds")
    print(f"Successful: {report['test_overview']['successful_tests']}")
    print(f"Failed: {report['test_overview']['failed_tests']}")
    print(f"Skipped: {report['test_overview']['skipped_tests']}")
    
    print(f"\n✅ Successful Providers: {', '.join(report['provider_status']['successful'])}")
    if report['provider_status']['failed']:
        print(f"❌ Failed Providers: {', '.join(report['provider_status']['failed'])}")
    if report['provider_status']['skipped']:
        print(f"⏭️  Skipped Providers: {', '.join(report['provider_status']['skipped'])}")
    
    print(f"\n💡 Recommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    
    # Save detailed report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"ai_provider_test_report_{timestamp}.json"
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    print("=" * 60)
    print("🎉 AI Provider Testing Completed!")
    
    return report

if __name__ == "__main__":
    # Install required packages if not present
    required_packages = [
        "openai",
        "anthropic", 
        "google-generativeai",
        "ollama",
        "python-dotenv"
    ]
    
    print("Checking required packages...")
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        sys.exit(1)
    
    # Run the tests
    asyncio.run(main())
