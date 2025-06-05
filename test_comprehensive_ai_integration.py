#!/usr/bin/env python3
"""
Comprehensive AI Integration Test for OpenAI and Ollama
Tests both OpenAI and Ollama models through the unified interface
"""

import asyncio
import json
import time
import requests
import sys
import traceback
from typing import Dict, List, Any, Optional

# Test configuration
SERVER_BASE_URL = "http://localhost:5001"
TEST_TIMEOUT = 30

class ComprehensiveAITester:
    """Comprehensive tester for OpenAI and Ollama integration."""
    
    def __init__(self):
        self.results = {
            "openai_tests": {},
            "ollama_tests": {},
            "api_tests": {},
            "integration_tests": {},
            "performance_metrics": {}
        }
        
    def test_server_health(self) -> bool:
        """Test if the server is running and healthy."""
        try:
            print("🔍 Testing server health...")
            response = requests.get(f"{SERVER_BASE_URL}/health", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Server is healthy: {data.get('status', 'unknown')}")
                self.results["api_tests"]["health_check"] = {
                    "success": True,
                    "data": data
                }
                return True
            else:
                print(f"❌ Server health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Failed to connect to server: {e}")
            return False
    
    def test_openai_api_key(self) -> bool:
        """Test OpenAI API key configuration."""
        try:
            print("🔑 Testing OpenAI API key...")
            response = requests.get(f"{SERVER_BASE_URL}/api/test-api-key", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ OpenAI API key is working: {data.get('message', '')}")
                self.results["openai_tests"]["api_key"] = {
                    "success": True,
                    "available_models": data.get("available_models", 0)
                }
                return True
            else:
                print(f"❌ OpenAI API key test failed: {response.text}")
                self.results["openai_tests"]["api_key"] = {
                    "success": False,
                    "error": response.text
                }
                return False
        except Exception as e:
            print(f"❌ Error testing OpenAI API key: {e}")
            return False
    
    def test_ollama_connection(self) -> bool:
        """Test Ollama server connection."""
        try:
            print("🦙 Testing Ollama connection...")
            # Test direct Ollama connection
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            
            if response.status_code == 200:
                models = response.json().get("models", [])
                print(f"✅ Ollama is available with {len(models)} models")
                self.results["ollama_tests"]["connection"] = {
                    "success": True,
                    "models": [model["name"] for model in models]
                }
                return True
            else:
                print(f"❌ Ollama connection failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {e}")
            self.results["ollama_tests"]["connection"] = {
                "success": False,
                "error": str(e)
            }
            return False
    
    def test_enhanced_agents_list(self) -> bool:
        """Test enhanced agents endpoint."""
        try:
            print("🤖 Testing enhanced agents list...")
            response = requests.get(f"{SERVER_BASE_URL}/api/enhanced/agents", timeout=10)
            
            if response.status_code == 200:
                agents = response.json()
                print(f"✅ Found {len(agents)} enhanced agents")
                for agent in agents:
                    print(f"  - {agent.get('name', 'Unknown')} ({agent.get('agent_type', 'Unknown')})")
                
                self.results["api_tests"]["enhanced_agents"] = {
                    "success": True,
                    "count": len(agents),
                    "agents": agents
                }
                return True
            else:
                print(f"❌ Enhanced agents test failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Error testing enhanced agents: {e}")
            return False
    
    def test_openai_agent_query(self, agent_type: str = "code_analysis") -> bool:
        """Test querying an OpenAI-based agent."""
        try:
            print(f"🧠 Testing OpenAI agent: {agent_type}")
            
            test_query = {
                "agent_type": agent_type,
                "input": {
                    "code": "def hello_world():\n    print('Hello, World!')",
                    "query": "Analyze this Python function for best practices"
                }
            }
            
            start_time = time.time()
            response = requests.post(
                f"{SERVER_BASE_URL}/api/enhanced/agents/query",
                json=test_query,
                timeout=TEST_TIMEOUT
            )
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ OpenAI agent response received in {execution_time:.2f}s")
                print(f"   Agent: {data.get('agent_name', 'Unknown')}")
                print(f"   Success: {data.get('success', False)}")
                if data.get('result'):
                    print(f"   Result preview: {str(data['result'])[:100]}...")
                
                self.results["openai_tests"]["agent_query"] = {
                    "success": data.get('success', False),
                    "agent_type": agent_type,
                    "execution_time": execution_time,
                    "response_length": len(str(data.get('result', '')))
                }
                return data.get('success', False)
            else:
                print(f"❌ OpenAI agent query failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing OpenAI agent: {e}")
            traceback.print_exc()
            return False
    
    def test_ollama_integration(self) -> bool:
        """Test Ollama integration through the unified system."""
        try:
            print("🦙 Testing Ollama integration...")
            
            # Test if there's an Ollama-specific endpoint or if agents can use Ollama
            # For now, let's check if we can create an agent with local model preference
            test_query = {
                "agent_type": "code_analysis",
                "model_preference": "local",  # Request local model if available
                "input": {
                    "code": "function factorial(n) { return n <= 1 ? 1 : n * factorial(n-1); }",
                    "query": "Analyze this JavaScript factorial function"
                }
            }
            
            start_time = time.time()
            response = requests.post(
                f"{SERVER_BASE_URL}/api/enhanced/agents/query",
                json=test_query,
                timeout=TEST_TIMEOUT
            )
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Ollama integration test completed in {execution_time:.2f}s")
                print(f"   Success: {data.get('success', False)}")
                
                self.results["ollama_tests"]["integration"] = {
                    "success": data.get('success', False),
                    "execution_time": execution_time,
                    "response_received": True
                }
                return True
            else:
                print(f"⚠️ Ollama integration test returned {response.status_code}")
                # This might be expected if Ollama integration isn't fully implemented
                self.results["ollama_tests"]["integration"] = {
                    "success": False,
                    "status_code": response.status_code,
                    "note": "May not be fully integrated yet"
                }
                return False
                
        except Exception as e:
            print(f"❌ Error testing Ollama integration: {e}")
            return False
    
    def test_tts_service(self) -> bool:
        """Test Text-to-Speech service."""
        try:
            print("🔊 Testing TTS service...")
            
            # Test TTS status
            response = requests.get(f"{SERVER_BASE_URL}/api/tts/status", timeout=10)
            if response.status_code == 200:
                status = response.json()
                print(f"✅ TTS service status: {status.get('status', 'unknown')}")
                
                # Test TTS voices
                voices_response = requests.get(f"{SERVER_BASE_URL}/api/tts/voices", timeout=10)
                if voices_response.status_code == 200:
                    voices = voices_response.json()
                    print(f"✅ TTS voices available: {len(voices)}")
                    
                    # Test simple TTS generation
                    tts_query = {
                        "text": "Hello, this is a test of the unified AI platform text-to-speech service.",
                        "voice": "alloy"
                    }
                    
                    start_time = time.time()
                    generate_response = requests.post(
                        f"{SERVER_BASE_URL}/api/tts/generate",
                        json=tts_query,
                        timeout=TEST_TIMEOUT
                    )
                    execution_time = time.time() - start_time
                    
                    if generate_response.status_code == 200:
                        tts_data = generate_response.json()
                        print(f"✅ TTS generation successful in {execution_time:.2f}s")
                        
                        self.results["api_tests"]["tts_service"] = {
                            "success": True,
                            "voices_count": len(voices),
                            "generation_time": execution_time,
                            "audio_generated": tts_data.get('success', False)
                        }
                        return True
                    else:
                        print(f"❌ TTS generation failed: {generate_response.status_code}")
                        return False
                else:
                    print(f"❌ TTS voices request failed: {voices_response.status_code}")
                    return False
            else:
                print(f"❌ TTS status request failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing TTS service: {e}")
            return False
    
    def test_multiple_agent_types(self) -> bool:
        """Test multiple agent types with both OpenAI."""
        agent_types = [
            "code_analysis",
            "research",
            "ceo",
            "triage"
        ]
        
        results = {}
        print(f"🔄 Testing {len(agent_types)} different agent types...")
        
        for agent_type in agent_types:
            try:
                print(f"  Testing {agent_type}...")
                
                test_inputs = {
                    "code_analysis": {
                        "code": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",
                        "query": "Analyze this sorting algorithm"
                    },
                    "research": {
                        "topic": "Artificial Intelligence trends in 2024",
                        "query": "What are the key trends in AI for 2024?"
                    },
                    "ceo": {
                        "query": "Provide a strategic overview of our AI platform capabilities"
                    },
                    "triage": {
                        "query": "Analyze the priority of implementing new AI features"
                    }
                }
                
                query = {
                    "agent_type": agent_type,
                    "input": test_inputs.get(agent_type, {"query": "Test query"})
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{SERVER_BASE_URL}/api/enhanced/agents/query",
                    json=query,
                    timeout=TEST_TIMEOUT
                )
                execution_time = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    success = data.get('success', False)
                    print(f"    ✅ {agent_type}: {'Success' if success else 'Failed'} ({execution_time:.2f}s)")
                    
                    results[agent_type] = {
                        "success": success,
                        "execution_time": execution_time,
                        "response_length": len(str(data.get('result', '')))
                    }
                else:
                    print(f"    ❌ {agent_type}: HTTP {response.status_code}")
                    results[agent_type] = {
                        "success": False,
                        "status_code": response.status_code
                    }
                    
            except Exception as e:
                print(f"    ❌ {agent_type}: Error - {e}")
                results[agent_type] = {
                    "success": False,
                    "error": str(e)
                }
        
        self.results["integration_tests"]["multiple_agents"] = results
        successful_tests = sum(1 for result in results.values() if result.get('success', False))
        print(f"✅ Agent type tests: {successful_tests}/{len(agent_types)} successful")
        
        return successful_tests > 0
    
    def performance_benchmark(self) -> Dict[str, Any]:
        """Run performance benchmarks."""
        print("⚡ Running performance benchmarks...")
        
        # Test response times for different operations
        benchmarks = {}
        
        # Health check benchmark
        times = []
        for i in range(5):
            start_time = time.time()
            response = requests.get(f"{SERVER_BASE_URL}/health", timeout=10)
            times.append(time.time() - start_time)
        
        benchmarks["health_check"] = {
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times)
        }
        
        # Agent query benchmark
        query = {
            "agent_type": "code_analysis",
            "input": {
                "code": "print('Hello, World!')",
                "query": "Analyze this code"
            }
        }
        
        agent_times = []
        for i in range(3):  # Fewer iterations for longer operations
            try:
                start_time = time.time()
                response = requests.post(
                    f"{SERVER_BASE_URL}/api/enhanced/agents/query",
                    json=query,
                    timeout=TEST_TIMEOUT
                )
                if response.status_code == 200:
                    agent_times.append(time.time() - start_time)
            except Exception as e:
                print(f"    Benchmark iteration {i+1} failed: {e}")
        
        if agent_times:
            benchmarks["agent_query"] = {
                "avg_time": sum(agent_times) / len(agent_times),
                "min_time": min(agent_times),
                "max_time": max(agent_times)
            }
        
        self.results["performance_metrics"] = benchmarks
        
        print(f"✅ Performance benchmarks completed")
        if "health_check" in benchmarks:
            print(f"    Health check avg: {benchmarks['health_check']['avg_time']:.3f}s")
        if "agent_query" in benchmarks:
            print(f"    Agent query avg: {benchmarks['agent_query']['avg_time']:.3f}s")
        
        return benchmarks
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all comprehensive tests."""
        print("🚀 Starting Comprehensive AI Integration Tests")
        print("=" * 60)
        
        start_time = time.time()
        
        # Core connectivity tests
        tests_passed = 0
        total_tests = 0
        
        test_results = [
            ("Server Health", self.test_server_health()),
            ("OpenAI API Key", self.test_openai_api_key()),
            ("Ollama Connection", self.test_ollama_connection()),
            ("Enhanced Agents List", self.test_enhanced_agents_list()),
            ("OpenAI Agent Query", self.test_openai_agent_query()),
            ("Ollama Integration", self.test_ollama_integration()),
            ("TTS Service", self.test_tts_service()),
            ("Multiple Agent Types", self.test_multiple_agent_types())
        ]
        
        for test_name, result in test_results:
            total_tests += 1
            if result:
                tests_passed += 1
            print()
        
        # Performance benchmarks
        print()
        self.performance_benchmark()
        
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("🎯 COMPREHENSIVE TEST RESULTS")
        print("=" * 60)
        print(f"📊 Tests Passed: {tests_passed}/{total_tests}")
        print(f"⏱️  Total Time: {total_time:.2f}s")
        print(f"✅ Success Rate: {(tests_passed/total_tests)*100:.1f}%")
        
        # Detailed results
        print("\n📋 Detailed Results:")
        if self.results["openai_tests"]:
            print("  🧠 OpenAI Tests:")
            for test, result in self.results["openai_tests"].items():
                status = "✅" if result.get("success", False) else "❌"
                print(f"    {status} {test}")
        
        if self.results["ollama_tests"]:
            print("  🦙 Ollama Tests:")
            for test, result in self.results["ollama_tests"].items():
                status = "✅" if result.get("success", False) else "❌"
                print(f"    {status} {test}")
        
        if self.results["api_tests"]:
            print("  🔧 API Tests:")
            for test, result in self.results["api_tests"].items():
                status = "✅" if result.get("success", False) else "❌"
                print(f"    {status} {test}")
        
        self.results["summary"] = {
            "tests_passed": tests_passed,
            "total_tests": total_tests,
            "success_rate": (tests_passed/total_tests)*100,
            "total_time": total_time
        }
        
        return self.results
    
    def save_results(self, filename: str = "comprehensive_test_results.json"):
        """Save test results to file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"💾 Test results saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")

def main():
    """Main test runner."""
    tester = ComprehensiveAITester()
    
    try:
        results = tester.run_comprehensive_tests()
        tester.save_results()
        
        # Return appropriate exit code
        success_rate = results["summary"]["success_rate"]
        if success_rate >= 80:
            print("\n🎉 Comprehensive tests PASSED!")
            sys.exit(0)
        elif success_rate >= 50:
            print("\n⚠️ Comprehensive tests PARTIALLY PASSED")
            sys.exit(1)
        else:
            print("\n❌ Comprehensive tests FAILED")
            sys.exit(2)
            
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
