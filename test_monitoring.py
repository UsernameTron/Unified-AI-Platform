#!/usr/bin/env python3
"""
Test script for AI Gatekeeper monitoring system
Validates that all monitoring endpoints work correctly
"""

import sys
import requests
import json
import time
from datetime import datetime

def test_monitoring_endpoints(base_url="http://localhost:5000"):
    """Test all monitoring endpoints."""
    
    print("🧪 Testing AI Gatekeeper Monitoring System")
    print("=" * 50)
    
    endpoints = [
        {
            'name': 'System Health',
            'url': f'{base_url}/api/monitoring/health',
            'method': 'GET'
        },
        {
            'name': 'Metrics (JSON)',
            'url': f'{base_url}/api/monitoring/metrics',
            'method': 'GET'
        },
        {
            'name': 'Metrics (Prometheus)',
            'url': f'{base_url}/api/monitoring/metrics?format=prometheus',
            'method': 'GET'
        },
        {
            'name': 'Performance Summary',
            'url': f'{base_url}/api/monitoring/performance?hours=1',
            'method': 'GET'
        },
        {
            'name': 'Dashboard Data',
            'url': f'{base_url}/api/monitoring/dashboard',
            'method': 'GET'
        }
    ]
    
    results = []
    
    for endpoint in endpoints:
        print(f"\n📡 Testing {endpoint['name']}...")
        print(f"   URL: {endpoint['url']}")
        
        try:
            start_time = time.time()
            response = requests.get(endpoint['url'], timeout=10)
            duration = time.time() - start_time
            
            if response.status_code == 200:
                print(f"   ✅ Status: {response.status_code}")
                print(f"   ⏱️  Response time: {duration:.3f}s")
                
                # Try to parse JSON (except for Prometheus format)
                if 'prometheus' not in endpoint['url']:
                    try:
                        data = response.json()
                        print(f"   📊 Data keys: {list(data.keys())}")
                        
                        # Show some sample data
                        if 'health' in endpoint['name'].lower():
                            print(f"   🏥 System status: {data.get('status', 'unknown')}")
                            print(f"   ⏰ Uptime: {data.get('uptime', 0):.1f}s")
                        
                        elif 'performance' in endpoint['name'].lower():
                            perf = data.get('performance', {})
                            if not perf.get('no_data'):
                                print(f"   📈 Total requests: {perf.get('total_requests', 0)}")
                                print(f"   📊 Success rate: {perf.get('success_rate', 0):.1%}")
                        
                    except json.JSONDecodeError:
                        print(f"   ⚠️  Response is not valid JSON")
                else:
                    print(f"   📏 Response length: {len(response.text)} characters")
                
                results.append({
                    'endpoint': endpoint['name'],
                    'status': 'PASS',
                    'duration': duration,
                    'status_code': response.status_code
                })
                
            else:
                print(f"   ❌ Status: {response.status_code}")
                print(f"   📄 Response: {response.text[:200]}...")
                
                results.append({
                    'endpoint': endpoint['name'],
                    'status': 'FAIL',
                    'duration': duration,
                    'status_code': response.status_code
                })
                
        except requests.exceptions.RequestException as e:
            print(f"   💥 Request failed: {e}")
            results.append({
                'endpoint': endpoint['name'],
                'status': 'ERROR',
                'error': str(e)
            })
    
    # Summary
    print(f"\n📋 Test Summary")
    print("=" * 30)
    
    passed = len([r for r in results if r['status'] == 'PASS'])
    failed = len([r for r in results if r['status'] in ['FAIL', 'ERROR']])
    total = len(results)
    
    print(f"Total endpoints tested: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Success rate: {passed/total*100:.1f}%")
    
    if failed > 0:
        print(f"\n❌ Failed endpoints:")
        for result in results:
            if result['status'] in ['FAIL', 'ERROR']:
                print(f"   • {result['endpoint']}: {result['status']}")
                if 'error' in result:
                    print(f"     Error: {result['error']}")
                elif 'status_code' in result:
                    print(f"     HTTP {result['status_code']}")
    
    return passed == total

def test_metrics_collection():
    """Test that metrics are being collected properly."""
    
    print(f"\n🔬 Testing Metrics Collection")
    print("=" * 35)
    
    base_url = "http://localhost:5000"
    
    # Make a test request to generate metrics
    print("📤 Making test support request to generate metrics...")
    
    test_payload = {
        "message": "Test monitoring request - how do I reset my password?",
        "context": {
            "user_level": "beginner",
            "source": "monitoring_test"
        }
    }
    
    try:
        response = requests.post(
            f"{base_url}/api/support/evaluate",
            json=test_payload,
            timeout=30
        )
        
        if response.status_code == 200:
            print("   ✅ Test request successful")
            result = response.json()
            print(f"   🎯 Action: {result.get('action', 'unknown')}")
        else:
            print(f"   ⚠️  Test request failed: HTTP {response.status_code}")
    
    except Exception as e:
        print(f"   💥 Test request error: {e}")
    
    # Wait a moment for metrics to be recorded
    time.sleep(1)
    
    # Check if metrics were recorded
    print("\n📊 Checking recorded metrics...")
    
    try:
        metrics_response = requests.get(f"{base_url}/api/monitoring/metrics")
        
        if metrics_response.status_code == 200:
            metrics_data = metrics_response.json()
            metrics = metrics_data.get('metrics', {})
            
            print(f"   📈 Total metric types: {len(metrics)}")
            
            # Look for expected metrics
            expected_metrics = [
                'requests_total',
                'request_duration', 
                'system_health',
                'system_uptime'
            ]
            
            found_metrics = []
            for metric in expected_metrics:
                if metric in metrics:
                    found_metrics.append(metric)
                    count = metrics[metric].get('count', 0)
                    print(f"   ✅ {metric}: {count} data points")
                else:
                    print(f"   ❌ {metric}: not found")
            
            return len(found_metrics) >= len(expected_metrics) // 2
            
        else:
            print(f"   ❌ Failed to get metrics: HTTP {metrics_response.status_code}")
            return False
            
    except Exception as e:
        print(f"   💥 Metrics check error: {e}")
        return False

def main():
    """Main test function."""
    
    print(f"🛡️  AI Gatekeeper Monitoring Test Suite")
    print(f"⏰ Started at: {datetime.now().isoformat()}")
    print()
    
    # Test if server is running
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code != 200:
            print("❌ AI Gatekeeper server is not running or not healthy")
            print("   Start the server with: python3 app.py")
            return False
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to AI Gatekeeper server at http://localhost:5000")
        print("   Start the server with: python3 app.py")
        return False
    
    print("✅ AI Gatekeeper server is running")
    
    # Run endpoint tests
    endpoints_ok = test_monitoring_endpoints()
    
    # Run metrics collection tests
    metrics_ok = test_metrics_collection()
    
    print(f"\n🎯 Overall Result")
    print("=" * 20)
    
    if endpoints_ok and metrics_ok:
        print("✅ All monitoring tests passed!")
        print("🚀 Monitoring system is working correctly")
        return True
    else:
        print("❌ Some monitoring tests failed")
        if not endpoints_ok:
            print("   • Endpoint tests failed")
        if not metrics_ok:
            print("   • Metrics collection tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)