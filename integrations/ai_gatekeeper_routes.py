"""
AI Gatekeeper Flask Routes
Integrates with existing Flask application to provide support workflow endpoints
"""

import json
import asyncio
import traceback
from typing import Dict, Any, Optional
from flask import Blueprint, request, jsonify, current_app
from datetime import datetime

# Import AI Gatekeeper components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.support_request_processor import SupportRequestProcessor, SupportRequest, SupportRequestStatus
from knowledge.solution_generator import KnowledgeBaseSolutionGenerator, SolutionType


# Create Blueprint for AI Gatekeeper routes
ai_gatekeeper_bp = Blueprint('ai_gatekeeper', __name__, url_prefix='/api/support')

# Global instances (will be initialized when routes are registered)
support_processor: Optional[SupportRequestProcessor] = None
solution_generator: Optional[KnowledgeBaseSolutionGenerator] = None


def register_ai_gatekeeper_routes(app):
    """
    Register AI Gatekeeper routes with the existing Flask application.
    
    Args:
        app: Flask application instance
    """
    global support_processor, solution_generator
    
    # Initialize AI Gatekeeper components with existing infrastructure
    try:
        from shared_agents.config.shared_config import SharedConfig
        
        # Create config (could also be loaded from app.config)
        config = SharedConfig()
        
        # Initialize support processor
        support_processor = SupportRequestProcessor(config)
        
        # Connect to existing agent manager and search system
        if hasattr(app, 'agent_manager'):
            support_processor.set_agent_manager(app.agent_manager)
        
        if hasattr(app, 'search_system'):
            support_processor.set_search_system(app.search_system)
        
        # Initialize solution generator
        solution_generator = KnowledgeBaseSolutionGenerator(
            agent_manager=getattr(app, 'agent_manager', None),
            search_system=getattr(app, 'search_system', None)
        )
        
        print("✅ AI Gatekeeper components initialized successfully")
        
    except Exception as e:
        print(f"⚠️  AI Gatekeeper initialization failed: {e}")
    
    # Register the blueprint
    app.register_blueprint(ai_gatekeeper_bp)
    
    return app


@ai_gatekeeper_bp.route('/evaluate', methods=['POST'])
def evaluate_support_request():
    """
    Main AI Gatekeeper endpoint for evaluating support requests.
    
    Expected JSON payload:
    {
        "message": "Support request description",
        "context": {
            "user_level": "beginner|intermediate|advanced",
            "system": "System information",
            "priority": "low|medium|high|critical"
        }
    }
    """
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Extract required fields
        message = data.get('message', '').strip()
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        user_context = data.get('context', {})
        
        # Ensure support processor is available
        if not support_processor:
            return jsonify({'error': 'AI Gatekeeper not properly initialized'}), 503
        
        # Process support request asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            support_request = loop.run_until_complete(
                support_processor.process_support_request(message, user_context)
            )
        finally:
            loop.close()
        
        # Format response based on resolution path
        if support_request.resolution_path == "automated_resolution":
            # Successful automated resolution
            solution_data = support_request.metadata.get('solution', {})
            
            response = {
                'action': 'automated_resolution',
                'request_id': support_request.id,
                'solution': solution_data,
                'confidence': support_request.confidence_score,
                'risk_score': support_request.risk_score,
                'estimated_time': solution_data.get('estimated_time', 'Unknown'),
                'status': support_request.status.value,
                'message': 'Solution generated successfully'
            }
        
        else:
            # Escalation to human expert
            enriched_context = support_request.metadata.get('enriched_context', {})
            
            response = {
                'action': 'escalate_to_human',
                'request_id': support_request.id,
                'analysis': {
                    'confidence_score': support_request.confidence_score,
                    'risk_score': support_request.risk_score,
                    'priority': support_request.priority.value,
                    'escalation_reason': support_request.metadata.get('escalation_reason')
                },
                'enriched_context': enriched_context,
                'status': support_request.status.value,
                'message': 'Request escalated to human expert'
            }
        
        return jsonify(response), 200
        
    except Exception as e:
        error_details = {
            'error': 'Internal server error',
            'details': str(e),
            'type': type(e).__name__
        }
        
        # Log the full traceback for debugging
        current_app.logger.error(f"AI Gatekeeper evaluation error: {traceback.format_exc()}")
        
        return jsonify(error_details), 500


@ai_gatekeeper_bp.route('/generate-solution', methods=['POST'])
def generate_solution():
    """
    Generate a detailed solution for a specific issue.
    
    Expected JSON payload:
    {
        "issue_description": "Detailed issue description",
        "context": {
            "user_level": "beginner|intermediate|advanced",
            "system": "System information"
        },
        "solution_type": "step_by_step|troubleshooting|configuration|documentation"
    }
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Extract required fields
        issue_description = data.get('issue_description', '').strip()
        if not issue_description:
            return jsonify({'error': 'Issue description is required'}), 400
        
        user_context = data.get('context', {})
        solution_type_str = data.get('solution_type')
        
        # Parse solution type
        solution_type = None
        if solution_type_str:
            try:
                solution_type = SolutionType(solution_type_str)
            except ValueError:
                return jsonify({'error': f'Invalid solution type: {solution_type_str}'}), 400
        
        # Ensure solution generator is available
        if not solution_generator:
            return jsonify({'error': 'Solution generator not properly initialized'}), 503
        
        # Generate solution asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            generated_solution = loop.run_until_complete(
                solution_generator.generate_solution(
                    issue_description, 
                    user_context, 
                    solution_type
                )
            )
        finally:
            loop.close()
        
        # Format solution for response
        solution_response = {
            'solution_id': generated_solution.id,
            'title': generated_solution.title,
            'summary': generated_solution.summary,
            'solution_type': generated_solution.solution_type.value,
            'complexity': generated_solution.complexity.value,
            'estimated_time': generated_solution.estimated_time,
            'confidence_score': generated_solution.confidence_score,
            'success_rate': generated_solution.success_rate,
            'steps': [
                {
                    'step_number': step.step_number,
                    'title': step.title,
                    'description': step.description,
                    'commands': step.commands,
                    'expected_result': step.expected_result,
                    'troubleshooting': step.troubleshooting,
                    'risk_level': step.risk_level
                }
                for step in generated_solution.steps
            ],
            'prerequisites': generated_solution.prerequisites,
            'warnings': generated_solution.warnings,
            'related_docs': generated_solution.related_docs,
            'generated_at': generated_solution.generated_at.isoformat(),
            'metadata': generated_solution.metadata
        }
        
        return jsonify(solution_response), 200
        
    except Exception as e:
        error_details = {
            'error': 'Solution generation failed',
            'details': str(e),
            'type': type(e).__name__
        }
        
        current_app.logger.error(f"Solution generation error: {traceback.format_exc()}")
        
        return jsonify(error_details), 500


@ai_gatekeeper_bp.route('/status/<request_id>', methods=['GET'])
def get_request_status(request_id: str):
    """
    Get the current status of a support request.
    
    Args:
        request_id: The ID of the support request
    """
    try:
        if not support_processor:
            return jsonify({'error': 'AI Gatekeeper not properly initialized'}), 503
        
        support_request = support_processor.get_request_status(request_id)
        
        if not support_request:
            return jsonify({'error': 'Request not found'}), 404
        
        status_response = {
            'request_id': support_request.id,
            'status': support_request.status.value,
            'priority': support_request.priority.value,
            'message': support_request.message[:100] + '...' if len(support_request.message) > 100 else support_request.message,
            'confidence_score': support_request.confidence_score,
            'risk_score': support_request.risk_score,
            'resolution_path': support_request.resolution_path,
            'assigned_agent': support_request.assigned_agent,
            'created_at': support_request.created_at.isoformat(),
            'updated_at': support_request.updated_at.isoformat(),
            'metadata': support_request.metadata
        }
        
        return jsonify(status_response), 200
        
    except Exception as e:
        error_details = {
            'error': 'Status retrieval failed',
            'details': str(e)
        }
        
        current_app.logger.error(f"Status retrieval error: {traceback.format_exc()}")
        
        return jsonify(error_details), 500


@ai_gatekeeper_bp.route('/active-requests', methods=['GET'])
def get_active_requests():
    """
    Get all active support requests.
    """
    try:
        if not support_processor:
            return jsonify({'error': 'AI Gatekeeper not properly initialized'}), 503
        
        active_requests = support_processor.get_active_requests()
        
        requests_response = {
            'total_requests': len(active_requests),
            'requests': [
                {
                    'request_id': req.id,
                    'status': req.status.value,
                    'priority': req.priority.value,
                    'message_preview': req.message[:100] + '...' if len(req.message) > 100 else req.message,
                    'confidence_score': req.confidence_score,
                    'risk_score': req.risk_score,
                    'resolution_path': req.resolution_path,
                    'created_at': req.created_at.isoformat(),
                    'updated_at': req.updated_at.isoformat()
                }
                for req in active_requests
            ]
        }
        
        return jsonify(requests_response), 200
        
    except Exception as e:
        error_details = {
            'error': 'Active requests retrieval failed',
            'details': str(e)
        }
        
        current_app.logger.error(f"Active requests retrieval error: {traceback.format_exc()}")
        
        return jsonify(error_details), 500


@ai_gatekeeper_bp.route('/slack-integration', methods=['POST'])
def slack_integration():
    """
    Handle Slack integration for AI Gatekeeper.
    
    Expected JSON payload:
    {
        "channel": "Channel ID",
        "user": "User ID", 
        "message": "Support request message",
        "context": {
            "user_level": "beginner|intermediate|advanced"
        }
    }
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Extract Slack-specific fields
        channel = data.get('channel')
        user = data.get('user')
        message = data.get('message', '').strip()
        context = data.get('context', {})
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Add Slack context
        context.update({
            'source': 'slack',
            'channel': channel,
            'user': user,
            'timestamp': datetime.now().isoformat()
        })
        
        # Process through main evaluation endpoint
        if not support_processor:
            return jsonify({'error': 'AI Gatekeeper not properly initialized'}), 503
        
        # Process support request
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            support_request = loop.run_until_complete(
                support_processor.process_support_request(message, context)
            )
        finally:
            loop.close()
        
        # Format response for Slack
        if support_request.resolution_path == "automated_resolution":
            solution_data = support_request.metadata.get('solution', {})
            
            # Format solution for Slack message
            slack_message = format_solution_for_slack(solution_data, support_request.confidence_score)
            
            response = {
                'action': 'send_slack_message',
                'channel': channel,
                'message': slack_message,
                'request_id': support_request.id,
                'message_type': 'automated_solution'
            }
        
        else:
            # Format escalation message for Slack
            escalation_message = format_escalation_for_slack(support_request)
            
            response = {
                'action': 'send_slack_message',
                'channel': channel,
                'message': escalation_message,
                'request_id': support_request.id,
                'message_type': 'escalation',
                'escalation_data': {
                    'priority': support_request.priority.value,
                    'confidence_score': support_request.confidence_score,
                    'risk_score': support_request.risk_score
                }
            }
        
        return jsonify(response), 200
        
    except Exception as e:
        error_details = {
            'error': 'Slack integration failed',
            'details': str(e)
        }
        
        current_app.logger.error(f"Slack integration error: {traceback.format_exc()}")
        
        return jsonify(error_details), 500


@ai_gatekeeper_bp.route('/feedback', methods=['POST'])
def submit_feedback():
    """
    Submit feedback on AI Gatekeeper solutions.
    
    Expected JSON payload:
    {
        "request_id": "Request ID",
        "solution_id": "Solution ID (optional)",
        "rating": 1-5,
        "feedback": "Text feedback",
        "outcome": "resolved|not_resolved|partially_resolved"
    }
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Extract feedback fields
        request_id = data.get('request_id')
        solution_id = data.get('solution_id')
        rating = data.get('rating')
        feedback_text = data.get('feedback', '')
        outcome = data.get('outcome')
        
        if not request_id:
            return jsonify({'error': 'Request ID is required'}), 400
        
        if rating is not None and (not isinstance(rating, int) or rating < 1 or rating > 5):
            return jsonify({'error': 'Rating must be an integer between 1 and 5'}), 400
        
        # Store feedback (implement based on your feedback storage system)
        feedback_data = {
            'request_id': request_id,
            'solution_id': solution_id,
            'rating': rating,
            'feedback': feedback_text,
            'outcome': outcome,
            'submitted_at': datetime.now().isoformat()
        }
        
        # Process feedback for learning (implement based on your learning system)
        if solution_generator:
            solution_generator.get_solution_feedback(solution_id or request_id, feedback_data)
        
        response = {
            'status': 'feedback_received',
            'message': 'Thank you for your feedback',
            'feedback_id': str(hash(f"{request_id}_{datetime.now().timestamp()}"))
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        error_details = {
            'error': 'Feedback submission failed',
            'details': str(e)
        }
        
        current_app.logger.error(f"Feedback submission error: {traceback.format_exc()}")
        
        return jsonify(error_details), 500


def format_solution_for_slack(solution_data: Dict[str, Any], confidence_score: float) -> str:
    """Format solution data for Slack message."""
    title = solution_data.get('title', 'Solution')
    summary = solution_data.get('summary', 'Generated solution')
    steps = solution_data.get('steps', [])
    
    message = f"🤖 *AI Support Resolution*\n\n"
    message += f"**{title}**\n\n"
    message += f"{summary}\n\n"
    
    if steps:
        message += "*Steps to resolve:*\n"
        for i, step in enumerate(steps[:5], 1):  # Limit to 5 steps for Slack
            if isinstance(step, dict):
                step_title = step.get('title', f'Step {i}')
                step_desc = step.get('description', '')
            else:
                step_title = f'Step {i}'
                step_desc = str(step)
            
            message += f"{i}. {step_title}: {step_desc}\n"
        
        if len(steps) > 5:
            message += f"... and {len(steps) - 5} more steps\n"
    
    message += f"\n**Confidence**: {confidence_score:.0%}\n"
    message += f"**Estimated Time**: {solution_data.get('estimated_time', 'Unknown')}\n\n"
    message += "Need more help? Reply to this message or escalate to human support."
    
    return message


def format_escalation_for_slack(support_request: SupportRequest) -> str:
    """Format escalation data for Slack message."""
    message = f"👨‍💻 *Escalated to Human Expert*\n\n"
    message += f"**Issue**: {support_request.message[:200]}...\n\n" if len(support_request.message) > 200 else f"**Issue**: {support_request.message}\n\n"
    message += f"**Priority**: {support_request.priority.value.upper()}\n"
    message += f"**Request ID**: {support_request.id}\n\n"
    
    escalation_reason = support_request.metadata.get('escalation_reason', 'Requires human expertise')
    message += f"**Escalation Reason**: {escalation_reason}\n\n"
    
    message += "You'll be contacted shortly. Context has been shared with the support team."
    
    return message


@ai_gatekeeper_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for AI Gatekeeper system."""
    try:
        health_status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'support_processor': support_processor is not None,
                'solution_generator': solution_generator is not None,
                'agent_manager': support_processor.agent_manager is not None if support_processor else False,
                'search_system': support_processor.search_system is not None if support_processor else False
            }
        }
        
        # Check if all components are healthy
        all_healthy = all(health_status['components'].values())
        
        if not all_healthy:
            health_status['status'] = 'degraded'
            return jsonify(health_status), 503
        
        return jsonify(health_status), 200
        
    except Exception as e:
        health_status = {
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        }
        
        return jsonify(health_status), 503