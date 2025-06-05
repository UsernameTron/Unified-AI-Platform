"""
Analytics API Routes for Unified AI Platform

This module provides REST API endpoints for advanced sentiment analysis
and predictive analytics capabilities.
"""

from flask import Blueprint, request, jsonify, current_app
import asyncio
import json
from typing import Dict, Any, Optional
import time
from datetime import datetime

from ..analytics.advanced_sentiment_agent import AdvancedSentimentAgent
from ..analytics.predictive_analytics_agent import PredictiveAnalyticsAgent

# Create Blueprint for analytics routes
analytics_bp = Blueprint('analytics', __name__, url_prefix='/api/analytics')

# Global agents (initialized lazily)
_sentiment_agent: Optional[AdvancedSentimentAgent] = None
_predictive_agent: Optional[PredictiveAnalyticsAgent] = None


def get_sentiment_agent() -> AdvancedSentimentAgent:
    """Get or create sentiment analysis agent"""
    global _sentiment_agent
    if _sentiment_agent is None:
        _sentiment_agent = AdvancedSentimentAgent()
    return _sentiment_agent


def get_predictive_agent() -> PredictiveAnalyticsAgent:
    """Get or create predictive analytics agent"""
    global _predictive_agent
    if _predictive_agent is None:
        _predictive_agent = PredictiveAnalyticsAgent()
    return _predictive_agent


@analytics_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for analytics service"""
    try:
        sentiment_agent = get_sentiment_agent()
        predictive_agent = get_predictive_agent()
        
        sentiment_info = sentiment_agent.get_model_info()
        predictive_info = predictive_agent.get_analytics_summary()
        
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'sentiment_analysis': {
                'available': True,
                'transformers_available': sentiment_info.get('transformers_available', False),
                'loaded_models': sentiment_info.get('loaded_models', []),
                'providers_available': sentiment_info.get('providers_available', [])
            },
            'predictive_analytics': {
                'available': True,
                'sklearn_available': predictive_info.get('sklearn_available', False),
                'conversation_history_size': predictive_info.get('conversation_history_size', 0),
                'capabilities': predictive_info.get('analytics_capabilities', [])
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@analytics_bp.route('/sentiment/analyze', methods=['POST'])
def analyze_sentiment():
    """
    Analyze sentiment of provided text
    
    Request body:
    {
        "text": "Text to analyze",
        "include_personality": true,
        "include_communication_style": true
    }
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        
        text = data['text']
        include_personality = data.get('include_personality', True)
        include_communication_style = data.get('include_communication_style', True)
        
        if len(text.strip()) == 0:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        if len(text) > 10000:  # Reasonable limit
            return jsonify({'error': 'Text too long (max 10,000 characters)'}), 400
        
        # Perform sentiment analysis
        sentiment_agent = get_sentiment_agent()
        
        # Run async function in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                sentiment_agent.analyze_sentiment(
                    text, 
                    include_personality=include_personality,
                    include_communication_style=include_communication_style
                )
            )
        finally:
            loop.close()
        
        return jsonify({
            'success': True,
            'result': result.to_dict()
        })
        
    except Exception as e:
        current_app.logger.error(f"Sentiment analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/sentiment/batch', methods=['POST'])
def batch_analyze_sentiment():
    """
    Analyze sentiment for multiple texts
    
    Request body:
    {
        "texts": ["Text 1", "Text 2", ...],
        "include_personality": true,
        "include_communication_style": true
    }
    """
    try:
        data = request.get_json()
        if not data or 'texts' not in data:
            return jsonify({'error': 'Texts array is required'}), 400
        
        texts = data['texts']
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({'error': 'Texts must be a non-empty array'}), 400
        
        if len(texts) > 50:  # Reasonable batch limit
            return jsonify({'error': 'Too many texts (max 50 per batch)'}), 400
        
        include_personality = data.get('include_personality', True)
        include_communication_style = data.get('include_communication_style', True)
        
        # Validate all texts
        for i, text in enumerate(texts):
            if not isinstance(text, str) or len(text.strip()) == 0:
                return jsonify({'error': f'Text at index {i} is invalid'}), 400
            if len(text) > 5000:  # Lower limit for batch processing
                return jsonify({'error': f'Text at index {i} is too long (max 5,000 characters for batch)'}), 400
        
        # Perform batch sentiment analysis
        sentiment_agent = get_sentiment_agent()
        
        # Run async function in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(
                sentiment_agent.batch_analyze(
                    texts,
                    include_personality=include_personality,
                    include_communication_style=include_communication_style
                )
            )
        finally:
            loop.close()
        
        return jsonify({
            'success': True,
            'results': [result.to_dict() for result in results]
        })
        
    except Exception as e:
        current_app.logger.error(f"Batch sentiment analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/predict/intent', methods=['POST'])
def predict_intent():
    """
    Predict user intent from text
    
    Request body:
    {
        "text": "Text to analyze",
        "context": {"additional": "context"},
        "user_history": ["previous", "interactions"]
    }
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        
        text = data['text']
        context = data.get('context', {})
        user_history = data.get('user_history', [])
        
        if len(text.strip()) == 0:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Perform intent prediction
        predictive_agent = get_predictive_agent()
        
        # Run async function in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                predictive_agent.predict_intent(text, context, user_history)
            )
        finally:
            loop.close()
        
        return jsonify({
            'success': True,
            'result': {
                'predicted_intent': result.predicted_intent,
                'confidence': result.confidence,
                'intent_categories': result.intent_categories,
                'context_factors': result.context_factors,
                'reasoning': result.reasoning
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Intent prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/predict/content-performance', methods=['POST'])
def predict_content_performance():
    """
    Predict content performance metrics
    
    Request body:
    {
        "content": "Content to analyze",
        "target_audience": "Target audience description",
        "content_type": "blog|social|email|general"
    }
    """
    try:
        data = request.get_json()
        if not data or 'content' not in data:
            return jsonify({'error': 'Content is required'}), 400
        
        content = data['content']
        target_audience = data.get('target_audience')
        content_type = data.get('content_type', 'general')
        
        if len(content.strip()) == 0:
            return jsonify({'error': 'Content cannot be empty'}), 400
        
        # Perform content performance prediction
        predictive_agent = get_predictive_agent()
        
        # Run async function in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                predictive_agent.predict_content_performance(content, target_audience, content_type)
            )
        finally:
            loop.close()
        
        return jsonify({
            'success': True,
            'result': {
                'engagement_score': result.engagement_score,
                'virality_potential': result.virality_potential,
                'retention_score': result.retention_score,
                'optimization_suggestions': result.optimization_suggestions,
                'target_demographics': result.target_demographics,
                'predicted_metrics': result.predicted_metrics
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Content performance prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/conversation/flow', methods=['POST'])
def analyze_conversation_flow():
    """
    Analyze conversation flow and predict next steps
    
    Request body:
    {
        "current_message": "Latest message in conversation"
    }
    """
    try:
        data = request.get_json()
        current_message = data.get('current_message') if data else None
        
        # Perform conversation flow analysis
        predictive_agent = get_predictive_agent()
        
        # Run async function in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                predictive_agent.analyze_conversation_flow(current_message)
            )
        finally:
            loop.close()
        
        return jsonify({
            'success': True,
            'result': {
                'next_likely_topics': result.next_likely_topics,
                'conversation_sentiment_trend': result.conversation_sentiment_trend,
                'engagement_level': result.engagement_level,
                'suggested_responses': result.suggested_responses,
                'flow_optimization': result.flow_optimization
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Conversation flow analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/trends/analyze', methods=['POST'])
def analyze_trends():
    """
    Analyze trends and patterns
    
    Request body:
    {
        "data_sources": ["source1", "source2"],
        "time_horizon": "30_days|7_days|24_hours"
    }
    """
    try:
        data = request.get_json()
        data_sources = data.get('data_sources', []) if data else []
        time_horizon = data.get('time_horizon', '30_days') if data else '30_days'
        
        # Perform trend analysis
        predictive_agent = get_predictive_agent()
        
        # Run async function in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                predictive_agent.analyze_trends(data_sources, time_horizon)
            )
        finally:
            loop.close()
        
        return jsonify({
            'success': True,
            'result': {
                'emerging_topics': result.emerging_topics,
                'seasonal_patterns': result.seasonal_patterns,
                'user_behavior_patterns': result.user_behavior_patterns,
                'market_insights': result.market_insights,
                'prediction_horizon': result.prediction_horizon
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Trend analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/comprehensive', methods=['POST'])
def comprehensive_analysis():
    """
    Perform comprehensive analytics including all components
    
    Request body:
    {
        "text": "Text to analyze",
        "context": {"additional": "context"},
        "analyze_content": true,
        "target_audience": "audience description",
        "content_type": "content type",
        "data_sources": ["source1", "source2"],
        "time_horizon": "30_days"
    }
    """
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Text is required'}), 400
        
        text = data['text']
        if len(text.strip()) == 0:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Extract all parameters
        kwargs = {
            'context': data.get('context', {}),
            'analyze_content': data.get('analyze_content', True),
            'target_audience': data.get('target_audience'),
            'content_type': data.get('content_type', 'general'),
            'data_sources': data.get('data_sources', []),
            'time_horizon': data.get('time_horizon', '30_days')
        }
        
        # Perform comprehensive analysis
        predictive_agent = get_predictive_agent()
        
        # Run async function in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(
                predictive_agent.comprehensive_analysis(text, **kwargs)
            )
        finally:
            loop.close()
        
        return jsonify({
            'success': True,
            'result': {
                'intent_prediction': result.intent_prediction.__dict__ if result.intent_prediction else None,
                'content_performance': result.content_performance.__dict__ if result.content_performance else None,
                'conversation_flow': result.conversation_flow.__dict__ if result.conversation_flow else None,
                'trend_analysis': result.trend_analysis.__dict__ if result.trend_analysis else None,
                'processing_time': result.processing_time,
                'timestamp': result.timestamp,
                'data_sources': result.data_sources,
                'confidence_level': result.confidence_level
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Comprehensive analysis error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@analytics_bp.route('/models/info', methods=['GET'])
def get_models_info():
    """Get information about available models and capabilities"""
    try:
        sentiment_agent = get_sentiment_agent()
        predictive_agent = get_predictive_agent()
        
        sentiment_info = sentiment_agent.get_model_info()
        predictive_info = predictive_agent.get_analytics_summary()
        
        return jsonify({
            'success': True,
            'sentiment_analysis': sentiment_info,
            'predictive_analytics': predictive_info,
            'api_endpoints': {
                'sentiment': [
                    '/api/analytics/sentiment/analyze',
                    '/api/analytics/sentiment/batch'
                ],
                'predictive': [
                    '/api/analytics/predict/intent',
                    '/api/analytics/predict/content-performance',
                    '/api/analytics/conversation/flow',
                    '/api/analytics/trends/analyze',
                    '/api/analytics/comprehensive'
                ],
                'utility': [
                    '/api/analytics/health',
                    '/api/analytics/models/info'
                ]
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Models info error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Error handlers
@analytics_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': [
            '/api/analytics/health',
            '/api/analytics/sentiment/analyze',
            '/api/analytics/sentiment/batch',
            '/api/analytics/predict/intent',
            '/api/analytics/predict/content-performance',
            '/api/analytics/conversation/flow',
            '/api/analytics/trends/analyze',
            '/api/analytics/comprehensive',
            '/api/analytics/models/info'
        ]
    }), 404


@analytics_bp.errorhandler(400)
def bad_request(error):
    return jsonify({
        'success': False,
        'error': 'Bad request - check your request format and parameters'
    }), 400


@analytics_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error - please try again later'
    }), 500
