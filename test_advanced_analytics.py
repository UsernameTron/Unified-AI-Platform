#!/usr/bin/env python3
"""
Advanced Analytics Testing Script for Unified AI Platform

This script tests all the advanced analytics capabilities including:
- Advanced sentiment analysis with emotion detection
- Personality trait inference
- Communication style analysis  
- Intent prediction
- Content performance forecasting
- Conversation flow analysis
- Trend analysis and pattern recognition
"""

import asyncio
import json
import time
import sys
import os
from datetime import datetime

# Add the current directory to the path to import shared_agents
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from shared_agents.analytics.advanced_sentiment_agent import AdvancedSentimentAgent
from shared_agents.analytics.predictive_analytics_agent import PredictiveAnalyticsAgent


class AdvancedAnalyticsTest:
    """Comprehensive testing suite for advanced analytics"""
    
    def __init__(self):
        self.sentiment_agent = None
        self.predictive_agent = None
        self.test_results = {}
        
    async def initialize_agents(self):
        """Initialize analytics agents"""
        print("🚀 Initializing Advanced Analytics Agents...")
        try:
            self.sentiment_agent = AdvancedSentimentAgent()
            self.predictive_agent = PredictiveAnalyticsAgent()
            print("✅ Agents initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize agents: {e}")
            return False
    
    async def test_sentiment_analysis(self):
        """Test advanced sentiment analysis capabilities"""
        print("\n📊 Testing Advanced Sentiment Analysis...")
        print("=" * 60)
        
        test_texts = [
            {
                'text': "I'm absolutely thrilled about this new opportunity! It's going to be amazing and I can't wait to get started.",
                'expected_sentiment': 'positive',
                'description': 'Highly positive with excitement'
            },
            {
                'text': "I'm feeling quite anxious about the upcoming presentation tomorrow. What if something goes wrong?",
                'expected_sentiment': 'negative',  
                'description': 'Anxiety and worry'
            },
            {
                'text': "The quarterly analysis shows a 15% increase in revenue with improved operational efficiency metrics.",
                'expected_sentiment': 'neutral',
                'description': 'Professional/analytical content'
            },
            {
                'text': "Hey there! Hope you're having a fantastic day! Let me know if you need anything! 😊",
                'expected_sentiment': 'positive',
                'description': 'Casual and enthusiastic'
            },
            {
                'text': "This is completely unacceptable. I've had enough of these constant delays and poor service.",
                'expected_sentiment': 'negative',
                'description': 'Anger and frustration'
            }
        ]
        
        results = []
        total_time = 0
        
        for i, test_case in enumerate(test_texts, 1):
            print(f"\nTest {i}: {test_case['description']}")
            print(f"Text: '{test_case['text'][:80]}{'...' if len(test_case['text']) > 80 else ''}'")
            print("-" * 50)
            
            start_time = time.time()
            try:
                result = await self.sentiment_agent.analyze_sentiment(
                    test_case['text'],
                    include_personality=True,
                    include_communication_style=True
                )
                
                processing_time = time.time() - start_time
                total_time += processing_time
                
                # Display results
                print(f"Sentiment: {result.sentiment} (confidence: {result.confidence:.2f})")
                print(f"Dominant emotion: {result.emotions.dominant_emotion}")
                
                # Show top 3 emotions
                emotions = result.emotions.to_dict()
                top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"Top emotions: {', '.join([f'{e}: {s:.2f}' for e, s in top_emotions])}")
                
                # Show personality insights
                personality = result.personality.to_dict()
                top_traits = sorted(personality.items(), key=lambda x: x[1], reverse=True)[:2]
                print(f"Key personality traits: {', '.join([f'{t}: {s:.2f}' for t, s in top_traits])}")
                
                # Show communication style
                comm_style = result.communication_style.to_dict()
                top_styles = sorted(comm_style.items(), key=lambda x: x[1], reverse=True)[:2]
                print(f"Communication style: {', '.join([f'{s}: {v:.2f}' for s, v in top_styles])}")
                
                print(f"Processing time: {processing_time:.3f}s")
                print(f"Models used: {', '.join(result.models_used)}")
                
                # Validate result
                is_correct = result.sentiment == test_case['expected_sentiment']
                status = "✅ PASS" if is_correct else "⚠️  CHECK"
                print(f"Expected: {test_case['expected_sentiment']} | Got: {result.sentiment} | {status}")
                
                results.append({
                    'test': i,
                    'description': test_case['description'],
                    'expected': test_case['expected_sentiment'],
                    'actual': result.sentiment,
                    'confidence': result.confidence,
                    'processing_time': processing_time,
                    'correct': is_correct
                })
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
                results.append({
                    'test': i,
                    'description': test_case['description'],
                    'error': str(e),
                    'correct': False
                })
        
        # Summary
        print(f"\n📈 Sentiment Analysis Test Summary:")
        print("-" * 40)
        correct_count = sum(1 for r in results if r.get('correct', False))
        print(f"Tests passed: {correct_count}/{len(test_texts)}")
        print(f"Average processing time: {total_time/len(test_texts):.3f}s")
        print(f"Total processing time: {total_time:.3f}s")
        
        self.test_results['sentiment_analysis'] = results
        return results
    
    async def test_predictive_analytics(self):
        """Test predictive analytics capabilities"""
        print("\n🔮 Testing Predictive Analytics...")
        print("=" * 60)
        
        test_scenarios = [
            {
                'text': "I need help with creating a comprehensive marketing strategy for my tech startup",
                'context': {'user_role': 'entrepreneur', 'urgency': 'high', 'domain': 'technology'},
                'expected_intent': 'task_completion',
                'description': 'Business strategy request'
            },
            {
                'text': "Can you explain how neural networks work and give me some examples?",
                'context': {'user_role': 'student', 'subject': 'machine_learning'},
                'expected_intent': 'learning_education',
                'description': 'Educational inquiry'
            },
            {
                'text': "I'm feeling overwhelmed with all these tasks and deadlines",
                'context': {'emotional_state': 'stressed', 'support_needed': True},
                'expected_intent': 'emotional_support',
                'description': 'Emotional support request'
            },
            {
                'text': "What's the weather like today? Also, do you know any good jokes?",
                'context': {'conversation_type': 'casual'},
                'expected_intent': 'casual_conversation',
                'description': 'Casual chat'
            }
        ]
        
        results = []
        total_time = 0
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\nTest {i}: {scenario['description']}")
            print(f"Text: '{scenario['text'][:80]}{'...' if len(scenario['text']) > 80 else ''}'")
            print("-" * 50)
            
            start_time = time.time()
            try:
                # Comprehensive analysis
                comprehensive_result = await self.predictive_agent.comprehensive_analysis(
                    scenario['text'],
                    context=scenario['context'],
                    analyze_content=True,
                    content_type='general'
                )
                
                processing_time = time.time() - start_time
                total_time += processing_time
                
                # Display results
                print(f"Processing time: {processing_time:.3f}s")
                print(f"Confidence level: {comprehensive_result.confidence_level}")
                
                if comprehensive_result.intent_prediction:
                    intent = comprehensive_result.intent_prediction
                    print(f"Predicted intent: {intent.predicted_intent} (confidence: {intent.confidence:.2f})")
                    print(f"Context factors: {', '.join(intent.context_factors[:3])}")
                    
                    # Validate intent prediction
                    is_correct = intent.predicted_intent == scenario['expected_intent']
                    status = "✅ PASS" if is_correct else "⚠️  CHECK"
                    print(f"Expected: {scenario['expected_intent']} | Got: {intent.predicted_intent} | {status}")
                
                if comprehensive_result.content_performance:
                    perf = comprehensive_result.content_performance
                    print(f"Engagement score: {perf.engagement_score:.2f}")
                    print(f"Top suggestion: {perf.optimization_suggestions[0] if perf.optimization_suggestions else 'None'}")
                
                if comprehensive_result.conversation_flow:
                    flow = comprehensive_result.conversation_flow
                    print(f"Engagement level: {flow.engagement_level:.2f}")
                    print(f"Sentiment trend: {flow.conversation_sentiment_trend}")
                    print(f"Next topics: {', '.join([t[0] for t in flow.next_likely_topics[:2]])}")
                
                results.append({
                    'test': i,
                    'description': scenario['description'],
                    'expected_intent': scenario['expected_intent'],
                    'actual_intent': comprehensive_result.intent_prediction.predicted_intent if comprehensive_result.intent_prediction else 'unknown',
                    'confidence': comprehensive_result.intent_prediction.confidence if comprehensive_result.intent_prediction else 0,
                    'processing_time': processing_time,
                    'confidence_level': comprehensive_result.confidence_level,
                    'correct': comprehensive_result.intent_prediction.predicted_intent == scenario['expected_intent'] if comprehensive_result.intent_prediction else False
                })
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
                results.append({
                    'test': i,
                    'description': scenario['description'],
                    'error': str(e),
                    'correct': False
                })
        
        # Summary
        print(f"\n📈 Predictive Analytics Test Summary:")
        print("-" * 40)
        correct_count = sum(1 for r in results if r.get('correct', False))
        print(f"Tests passed: {correct_count}/{len(test_scenarios)}")
        print(f"Average processing time: {total_time/len(test_scenarios):.3f}s")
        print(f"Total processing time: {total_time:.3f}s")
        
        self.test_results['predictive_analytics'] = results
        return results
    
    async def test_model_capabilities(self):
        """Test model information and capabilities"""
        print("\n🔧 Testing Model Capabilities...")
        print("=" * 60)
        
        try:
            # Test sentiment agent capabilities
            sentiment_info = self.sentiment_agent.get_model_info()
            print("Sentiment Analysis Capabilities:")
            for key, value in sentiment_info.items():
                print(f"  {key}: {value}")
            
            print("\nPredictive Analytics Capabilities:")
            predictive_info = self.predictive_agent.get_analytics_summary()
            for key, value in predictive_info.items():
                print(f"  {key}: {value}")
            
            self.test_results['model_capabilities'] = {
                'sentiment': sentiment_info,
                'predictive': predictive_info
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Capability test failed: {e}")
            return False
    
    async def run_all_tests(self):
        """Run the complete test suite"""
        print("🧪 Advanced Analytics Test Suite")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        start_time = time.time()
        
        # Initialize agents
        if not await self.initialize_agents():
            print("❌ Cannot proceed without agents")
            return
        
        # Run all tests
        await self.test_model_capabilities()
        await self.test_sentiment_analysis()
        await self.test_predictive_analytics()
        
        total_time = time.time() - start_time
        
        # Final summary
        print(f"\n🏁 Test Suite Complete!")
        print("=" * 80)
        print(f"Total execution time: {total_time:.3f}s")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Overall results
        sentiment_results = self.test_results.get('sentiment_analysis', [])
        predictive_results = self.test_results.get('predictive_analytics', [])
        
        sentiment_passed = sum(1 for r in sentiment_results if r.get('correct', False))
        predictive_passed = sum(1 for r in predictive_results if r.get('correct', False))
        
        print(f"\n📊 Overall Results:")
        print(f"  Sentiment Analysis: {sentiment_passed}/{len(sentiment_results)} tests passed")
        print(f"  Predictive Analytics: {predictive_passed}/{len(predictive_results)} tests passed")
        
        total_tests = len(sentiment_results) + len(predictive_results)
        total_passed = sentiment_passed + predictive_passed
        pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        print(f"  Overall Pass Rate: {pass_rate:.1f}% ({total_passed}/{total_tests})")
        
        # Save detailed results
        with open('analytics_test_results.json', 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        print(f"\n💾 Detailed results saved to 'analytics_test_results.json'")
        
        return pass_rate >= 70  # Consider 70%+ a success


async def main():
    """Main test execution"""
    test_suite = AdvancedAnalyticsTest()
    success = await test_suite.run_all_tests()
    
    if success:
        print("\n🎉 Advanced Analytics Test Suite: PASSED")
        sys.exit(0)
    else:
        print("\n❌ Advanced Analytics Test Suite: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        sys.exit(1)
