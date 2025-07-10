"""
Slack Integration for AI Gatekeeper System
Leverages existing infrastructure for seamless Slack communication
"""

import os
import json
import asyncio
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

# Import existing TTS service and ExecutiveAssistant capabilities
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


@dataclass
class SlackMessage:
    """Slack message data structure."""
    channel: str
    text: str
    user: Optional[str] = None
    timestamp: Optional[str] = None
    thread_ts: Optional[str] = None
    attachments: Optional[List[Dict[str, Any]]] = None
    blocks: Optional[List[Dict[str, Any]]] = None


@dataclass
class SlackUser:
    """Slack user information."""
    id: str
    name: str
    email: Optional[str] = None
    is_admin: bool = False
    timezone: Optional[str] = None


class SlackConnector:
    """
    Slack integration connector that leverages existing infrastructure
    for AI Gatekeeper support workflow automation.
    """
    
    def __init__(self, bot_token: str, agent_manager=None, tts_service=None):
        """
        Initialize Slack connector.
        
        Args:
            bot_token: Slack bot token
            agent_manager: Existing agent manager instance
            tts_service: Existing TTS service instance
        """
        self.bot_token = bot_token
        self.agent_manager = agent_manager
        self.tts_service = tts_service
        
        # Slack API endpoints
        self.base_url = "https://slack.com/api"
        self.headers = {
            "Authorization": f"Bearer {bot_token}",
            "Content-Type": "application/json"
        }
        
        # Message templates
        self.message_templates = self._load_message_templates()
        
        # Support workflow settings
        self.support_channel_patterns = ['support', 'help', 'it-support']
        self.escalation_channels = {'high': '@support-team', 'critical': '@on-call'}
        
    def _load_message_templates(self) -> Dict[str, str]:
        """Load message templates for different scenarios."""
        return {
            'automated_solution': '''
🤖 *AI Support Resolution*

**Issue**: {issue}

**Solution**:
{solution_steps}

**Confidence**: {confidence:.0%}
**Estimated Time**: {estimated_time}

*Need more help? Reply to this message or react with ❓ to escalate to human support.*
            ''',
            
            'escalation_notification': '''
👨‍💻 *Escalated to Human Expert*

**Issue**: {issue}
**Priority**: {priority}
**Request ID**: {request_id}

**AI Analysis**:
• Confidence Score: {confidence:.0%}
• Risk Level: {risk_level}
• Escalation Reason: {escalation_reason}

You'll be contacted shortly by our support team.
            ''',
            
            'expert_handoff': '''
🔄 *Support Request Handoff*

**From**: AI Gatekeeper
**To**: {expert_name}
**Priority**: {priority}

**Issue Summary**: {issue}

**AI Analysis**:
• Confidence: {confidence:.0%}
• Risk: {risk_level}
• Similar Cases: {similar_cases_count}

**Context**: {enriched_context}

Please take over this support request.
            ''',
            
            'solution_feedback': '''
📊 *Solution Feedback Request*

How did our AI solution work for you?

{solution_summary}

Please rate your experience:
⭐ = Poor
⭐⭐ = Fair  
⭐⭐⭐ = Good
⭐⭐⭐⭐ = Very Good
⭐⭐⭐⭐⭐ = Excellent

Or provide detailed feedback by replying to this message.
            '''
        }
    
    async def send_automated_solution(self, 
                                    channel: str, 
                                    user: str,
                                    issue: str,
                                    solution_data: Dict[str, Any],
                                    confidence_score: float) -> Dict[str, Any]:
        """
        Send an automated solution via Slack.
        
        Args:
            channel: Slack channel ID
            user: User ID who requested support
            issue: Original issue description
            solution_data: Generated solution data
            confidence_score: AI confidence score
            
        Returns:
            Slack API response
        """
        try:
            # Format solution steps
            steps = solution_data.get('steps', [])
            solution_steps = ""
            
            for i, step in enumerate(steps[:5], 1):  # Limit to 5 steps for Slack
                if isinstance(step, dict):
                    title = step.get('title', f'Step {i}')
                    description = step.get('description', '')
                    solution_steps += f"{i}. **{title}**: {description}\n"
                else:
                    solution_steps += f"{i}. {str(step)}\n"
            
            if len(steps) > 5:
                solution_steps += f"... and {len(steps) - 5} more steps (view full solution online)\n"
            
            # Create message using template
            message_text = self.message_templates['automated_solution'].format(
                issue=issue[:200] + "..." if len(issue) > 200 else issue,
                solution_steps=solution_steps,
                confidence=confidence_score,
                estimated_time=solution_data.get('estimated_time', 'Unknown')
            )
            
            # Add interactive elements
            blocks = self._create_solution_blocks(solution_data, confidence_score)
            
            # Send message
            response = await self._send_message(
                channel=channel,
                text=message_text,
                blocks=blocks,
                user=user
            )
            
            # Generate audio version if TTS service available
            if self.tts_service:
                await self._send_audio_solution(channel, solution_data)
            
            return response
            
        except Exception as e:
            return {'ok': False, 'error': str(e)}
    
    async def send_escalation_notification(self,
                                         channel: str,
                                         user: str,
                                         support_request: Any) -> Dict[str, Any]:
        """
        Send escalation notification to user.
        
        Args:
            channel: Slack channel ID
            user: User ID
            support_request: Support request object
            
        Returns:
            Slack API response
        """
        try:
            # Create escalation message
            message_text = self.message_templates['escalation_notification'].format(
                issue=support_request.message[:150] + "..." if len(support_request.message) > 150 else support_request.message,
                priority=support_request.priority.value.upper(),
                request_id=support_request.id,
                confidence=support_request.confidence_score or 0.0,
                risk_level=self._format_risk_level(support_request.risk_score or 0.5),
                escalation_reason=support_request.metadata.get('escalation_reason', 'Requires human expertise')
            )
            
            # Create escalation blocks
            blocks = self._create_escalation_blocks(support_request)
            
            # Send message
            response = await self._send_message(
                channel=channel,
                text=message_text,
                blocks=blocks,
                user=user
            )
            
            # Notify support team
            await self._notify_support_team(support_request)
            
            return response
            
        except Exception as e:
            return {'ok': False, 'error': str(e)}
    
    async def handle_support_request_from_slack(self, 
                                              slack_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming support request from Slack.
        
        Args:
            slack_payload: Slack event payload
            
        Returns:
            Processing result
        """
        try:
            # Extract information from Slack payload
            event = slack_payload.get('event', {})
            channel = event.get('channel')
            user = event.get('user')
            text = event.get('text', '')
            
            # Check if this is a support request
            if not self._is_support_request(text, channel):
                return {'handled': False, 'reason': 'Not a support request'}
            
            # Get user context
            user_context = await self._get_user_context(user)
            
            # Add Slack-specific context
            user_context.update({
                'source': 'slack',
                'channel': channel,
                'user': user,
                'timestamp': event.get('ts')
            })
            
            # Process through AI Gatekeeper (assuming support processor is available)
            from core.support_request_processor import SupportRequestProcessor
            
            # This would be injected in real implementation
            support_processor = SupportRequestProcessor(None)  # Would need proper config
            
            if self.agent_manager:
                support_processor.set_agent_manager(self.agent_manager)
            
            # Process the support request
            support_request = await support_processor.process_support_request(text, user_context)
            
            # Send appropriate response based on resolution path
            if support_request.resolution_path == "automated_resolution":
                solution_data = support_request.metadata.get('solution', {})
                await self.send_automated_solution(
                    channel, user, text, solution_data, support_request.confidence_score
                )
            else:
                await self.send_escalation_notification(channel, user, support_request)
            
            return {
                'handled': True,
                'request_id': support_request.id,
                'resolution_path': support_request.resolution_path
            }
            
        except Exception as e:
            return {'handled': False, 'error': str(e)}
    
    async def _send_message(self, 
                          channel: str, 
                          text: str, 
                          blocks: Optional[List[Dict[str, Any]]] = None,
                          user: Optional[str] = None,
                          thread_ts: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a message to Slack channel.
        
        Args:
            channel: Channel ID
            text: Message text
            blocks: Optional blocks for rich formatting
            user: Optional user to mention
            thread_ts: Optional thread timestamp for threaded reply
            
        Returns:
            Slack API response
        """
        try:
            payload = {
                'channel': channel,
                'text': text
            }
            
            if blocks:
                payload['blocks'] = blocks
            
            if thread_ts:
                payload['thread_ts'] = thread_ts
            
            # Make API call
            response = requests.post(
                f"{self.base_url}/chat.postMessage",
                headers=self.headers,
                json=payload
            )
            
            return response.json()
            
        except Exception as e:
            return {'ok': False, 'error': str(e)}
    
    async def _send_audio_solution(self, channel: str, solution_data: Dict[str, Any]) -> None:
        """
        Send audio version of solution using existing TTS service.
        
        Args:
            channel: Slack channel
            solution_data: Solution data to convert to audio
        """
        try:
            if not self.tts_service:
                return
            
            # Create audio-friendly text
            solution_text = self._format_solution_for_audio(solution_data)
            
            # Generate audio using existing TTS service
            audio_response = await self.tts_service.generate_speech(
                text=solution_text,
                voice="sage",  # Use educational voice for support
                response_format="mp3",
                speed=1.0
            )
            
            if audio_response.get('success'):
                # Upload audio file to Slack (implementation depends on your file upload system)
                await self._upload_audio_to_slack(channel, audio_response['audio_data'])
                
        except Exception as e:
            print(f"Audio solution generation failed: {e}")
    
    def _create_solution_blocks(self, solution_data: Dict[str, Any], confidence_score: float) -> List[Dict[str, Any]]:
        """Create Slack blocks for solution display."""
        blocks = []
        
        # Header block
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "🤖 AI Support Solution"
            }
        })
        
        # Solution summary
        if solution_data.get('summary'):
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Summary:* {solution_data['summary']}"
                }
            })
        
        # Confidence and time
        blocks.append({
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Confidence:* {confidence_score:.0%}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Est. Time:* {solution_data.get('estimated_time', 'Unknown')}"
                }
            ]
        })
        
        # Action buttons
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "👍 Helpful"
                    },
                    "value": "solution_helpful",
                    "action_id": "solution_feedback_positive"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "👎 Not Helpful"
                    },
                    "value": "solution_not_helpful",
                    "action_id": "solution_feedback_negative"
                },
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "❓ Escalate"
                    },
                    "value": "escalate_request",
                    "action_id": "escalate_to_human"
                }
            ]
        })
        
        return blocks
    
    def _create_escalation_blocks(self, support_request: Any) -> List[Dict[str, Any]]:
        """Create Slack blocks for escalation notification."""
        blocks = []
        
        # Header
        blocks.append({
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": "👨‍💻 Escalated to Human Expert"
            }
        })
        
        # Request details
        blocks.append({
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Priority:* {support_request.priority.value.upper()}"
                },
                {
                    "type": "mrkdwn",
                    "text": f"*Request ID:* {support_request.id}"
                }
            ]
        })
        
        # Escalation reason
        escalation_reason = support_request.metadata.get('escalation_reason', 'Requires human expertise')
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Reason:* {escalation_reason}"
            }
        })
        
        return blocks
    
    async def _get_user_context(self, user_id: str) -> Dict[str, Any]:
        """
        Get user context from Slack API.
        
        Args:
            user_id: Slack user ID
            
        Returns:
            User context dictionary
        """
        try:
            # Get user info from Slack API
            response = requests.get(
                f"{self.base_url}/users.info",
                headers=self.headers,
                params={'user': user_id}
            )
            
            user_data = response.json()
            
            if user_data.get('ok'):
                user = user_data['user']
                return {
                    'user_id': user_id,
                    'user_name': user.get('name', ''),
                    'user_email': user.get('profile', {}).get('email', ''),
                    'user_level': 'intermediate',  # Default, could be determined from user profile
                    'timezone': user.get('tz', ''),
                    'is_admin': user.get('is_admin', False)
                }
            else:
                return {'user_id': user_id, 'user_level': 'intermediate'}
                
        except Exception as e:
            print(f"Error getting user context: {e}")
            return {'user_id': user_id, 'user_level': 'intermediate'}
    
    def _is_support_request(self, text: str, channel: str) -> bool:
        """
        Determine if a message is a support request.
        
        Args:
            text: Message text
            channel: Channel ID
            
        Returns:
            True if this appears to be a support request
        """
        # Check if posted in support channels
        if any(pattern in channel.lower() for pattern in self.support_channel_patterns):
            return True
        
        # Check for support keywords in message
        support_keywords = [
            'help', 'issue', 'problem', 'error', 'bug', 'broken',
            'not working', 'can\'t', 'unable', 'how to', 'support'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in support_keywords)
    
    def _format_risk_level(self, risk_score: float) -> str:
        """Format risk score as human-readable level."""
        if risk_score < 0.3:
            return "Low"
        elif risk_score < 0.7:
            return "Medium"
        else:
            return "High"
    
    def _format_solution_for_audio(self, solution_data: Dict[str, Any]) -> str:
        """Format solution for audio/TTS conversion."""
        title = solution_data.get('title', 'Support Solution')
        summary = solution_data.get('summary', '')
        steps = solution_data.get('steps', [])
        
        audio_text = f"Here's your support solution: {title}. "
        
        if summary:
            audio_text += f"{summary}. "
        
        if steps:
            audio_text += "Here are the steps to resolve your issue: "
            for i, step in enumerate(steps[:3], 1):  # Limit to 3 steps for audio
                if isinstance(step, dict):
                    step_desc = step.get('description', step.get('title', ''))
                else:
                    step_desc = str(step)
                
                audio_text += f"Step {i}: {step_desc}. "
            
            if len(steps) > 3:
                audio_text += f"There are {len(steps) - 3} additional steps available in the text version. "
        
        audio_text += "Let me know if you need any clarification or additional help."
        
        return audio_text
    
    async def _upload_audio_to_slack(self, channel: str, audio_data: bytes) -> None:
        """Upload audio file to Slack channel."""
        try:
            # This would implement actual file upload to Slack
            # For now, just log that audio would be uploaded
            print(f"Would upload audio solution to channel {channel}")
            
        except Exception as e:
            print(f"Audio upload failed: {e}")
    
    async def _notify_support_team(self, support_request: Any) -> None:
        """Notify appropriate support team about escalation."""
        try:
            priority = support_request.priority.value
            notification_target = self.escalation_channels.get(priority, '@support-team')
            
            # Format expert handoff message
            expert_message = self.message_templates['expert_handoff'].format(
                expert_name=notification_target,
                priority=priority.upper(),
                issue=support_request.message[:200] + "..." if len(support_request.message) > 200 else support_request.message,
                confidence=support_request.confidence_score or 0.0,
                risk_level=self._format_risk_level(support_request.risk_score or 0.5),
                similar_cases_count=len(support_request.metadata.get('enriched_context', {}).get('similar_cases', [])),
                enriched_context=json.dumps(support_request.metadata.get('enriched_context', {}), indent=2)[:500] + "..."
            )
            
            # Send to support team channel (implementation would depend on your setup)
            print(f"Would notify {notification_target} about escalation: {support_request.id}")
            
        except Exception as e:
            print(f"Support team notification failed: {e}")


class SlackEventHandler:
    """
    Handler for Slack events and interactions.
    """
    
    def __init__(self, slack_connector: SlackConnector):
        """Initialize event handler."""
        self.slack_connector = slack_connector
    
    async def handle_message_event(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming message events."""
        return await self.slack_connector.handle_support_request_from_slack(event_data)
    
    async def handle_button_interaction(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle button click interactions."""
        try:
            action_id = interaction_data.get('actions', [{}])[0].get('action_id')
            
            if action_id == 'solution_feedback_positive':
                return await self._handle_positive_feedback(interaction_data)
            elif action_id == 'solution_feedback_negative':
                return await self._handle_negative_feedback(interaction_data)
            elif action_id == 'escalate_to_human':
                return await self._handle_escalation_request(interaction_data)
            
            return {'handled': False, 'reason': f'Unknown action: {action_id}'}
            
        except Exception as e:
            return {'handled': False, 'error': str(e)}
    
    async def _handle_positive_feedback(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle positive feedback on solution."""
        # Log positive feedback and update learning systems
        return {'handled': True, 'feedback': 'positive'}
    
    async def _handle_negative_feedback(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle negative feedback on solution."""
        # Log negative feedback and potentially escalate
        return {'handled': True, 'feedback': 'negative'}
    
    async def _handle_escalation_request(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle manual escalation request."""
        # Process escalation request
        return {'handled': True, 'action': 'escalated'}