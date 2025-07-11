"""
Configuration-Driven Workflows and Integration System for AI Gatekeeper
Enables dynamic workflow configuration, webhook integration, and plugin architecture
"""

import json
import asyncio
import hmac
import hashlib
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
import aiohttp
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Event types for the integration system."""
    REQUEST_RECEIVED = "request_received"
    REQUEST_PROCESSED = "request_processed"
    SOLUTION_GENERATED = "solution_generated"
    ESCALATION_TRIGGERED = "escalation_triggered"
    FEEDBACK_RECEIVED = "feedback_received"
    SYSTEM_ERROR = "system_error"

@dataclass
class WorkflowRule:
    """Workflow rule configuration."""
    id: str
    name: str
    condition: Dict[str, Any]
    action: Dict[str, Any]
    priority: int = 100
    enabled: bool = True

@dataclass
class WebhookConfig:
    """Webhook configuration."""
    url: str
    events: List[EventType]
    secret: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    retry_count: int = 3
    enabled: bool = True

@dataclass
class IntegrationEvent:
    """Event data for integrations."""
    event_type: EventType
    timestamp: datetime
    data: Dict[str, Any]
    source: str = "ai_gatekeeper"
    correlation_id: Optional[str] = None

class RuleEngine:
    """Rule engine for dynamic workflow configuration."""
    
    def __init__(self):
        self.rules: List[WorkflowRule] = []
        self.custom_conditions: Dict[str, Callable] = {}
        self.custom_actions: Dict[str, Callable] = {}
        
    def add_rule(self, rule: WorkflowRule):
        """Add a workflow rule."""
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority)
        
    def register_condition(self, name: str, func: Callable):
        """Register custom condition function."""
        self.custom_conditions[name] = func
        
    def register_action(self, name: str, func: Callable):
        """Register custom action function."""
        self.custom_actions[name] = func
        
    async def evaluate_rules(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate rules against context and return actions to execute."""
        actions_to_execute = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
                
            try:
                if await self._evaluate_condition(rule.condition, context):
                    action_result = await self._prepare_action(rule.action, context)
                    actions_to_execute.append({
                        'rule_id': rule.id,
                        'rule_name': rule.name,
                        'action': action_result
                    })
                    
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.id}: {e}")
                
        return actions_to_execute
        
    async def _evaluate_condition(self, condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate a single condition."""
        condition_type = condition.get('type')
        
        if condition_type == 'and':
            return all(await self._evaluate_condition(c, context) for c in condition.get('conditions', []))
            
        elif condition_type == 'or':
            return any(await self._evaluate_condition(c, context) for c in condition.get('conditions', []))
            
        elif condition_type == 'field_equals':
            field_path = condition.get('field')
            expected_value = condition.get('value')
            actual_value = self._get_nested_value(context, field_path)
            return actual_value == expected_value
            
        elif condition_type == 'field_contains':
            field_path = condition.get('field')
            search_value = condition.get('value')
            actual_value = self._get_nested_value(context, field_path)
            return search_value in str(actual_value) if actual_value else False
            
        elif condition_type == 'field_greater_than':
            field_path = condition.get('field')
            threshold = condition.get('value')
            actual_value = self._get_nested_value(context, field_path)
            return float(actual_value) > float(threshold) if actual_value is not None else False
            
        elif condition_type == 'custom':
            func_name = condition.get('function')
            if func_name in self.custom_conditions:
                return await self._call_function(self.custom_conditions[func_name], condition, context)
                
        return False
        
    async def _prepare_action(self, action: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare action for execution."""
        action_type = action.get('type')
        
        if action_type == 'set_field':
            return {
                'type': 'set_field',
                'field': action.get('field'),
                'value': self._interpolate_value(action.get('value'), context)
            }
            
        elif action_type == 'webhook':
            return {
                'type': 'webhook',
                'url': self._interpolate_value(action.get('url'), context),
                'payload': self._interpolate_dict(action.get('payload', {}), context)
            }
            
        elif action_type == 'escalate':
            return {
                'type': 'escalate',
                'reason': self._interpolate_value(action.get('reason'), context),
                'priority': action.get('priority', 'medium')
            }
            
        elif action_type == 'custom':
            func_name = action.get('function')
            if func_name in self.custom_actions:
                return await self._call_function(self.custom_actions[func_name], action, context)
                
        return action
        
    def _get_nested_value(self, data: Dict[str, Any], field_path: str) -> Any:
        """Get nested value using dot notation (e.g., 'user.level')."""
        parts = field_path.split('.')
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
                
        return current
        
    def _interpolate_value(self, template: str, context: Dict[str, Any]) -> str:
        """Interpolate template string with context values."""
        if not isinstance(template, str):
            return template
            
        # Simple template interpolation (could be enhanced with Jinja2)
        import re
        
        def replace_var(match):
            var_name = match.group(1)
            return str(self._get_nested_value(context, var_name) or '')
            
        return re.sub(r'\{\{(.+?)\}\}', replace_var, template)
        
    def _interpolate_dict(self, template_dict: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Interpolate dictionary values."""
        result = {}
        for key, value in template_dict.items():
            if isinstance(value, str):
                result[key] = self._interpolate_value(value, context)
            elif isinstance(value, dict):
                result[key] = self._interpolate_dict(value, context)
            else:
                result[key] = value
        return result
        
    async def _call_function(self, func: Callable, config: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Call custom function safely."""
        try:
            if asyncio.iscoroutinefunction(func):
                return await func(config, context)
            else:
                return func(config, context)
        except Exception as e:
            logger.error(f"Custom function call failed: {e}")
            return None

class WebhookManager:
    """Manages webhook integrations."""
    
    def __init__(self):
        self.webhooks: List[WebhookConfig] = []
        self.session: Optional[aiohttp.ClientSession] = None
        
    def add_webhook(self, webhook: WebhookConfig):
        """Add webhook configuration."""
        self.webhooks.append(webhook)
        
    async def trigger_event(self, event: IntegrationEvent):
        """Trigger webhooks for an event."""
        relevant_webhooks = [
            wh for wh in self.webhooks 
            if wh.enabled and event.event_type in wh.events
        ]
        
        if not relevant_webhooks:
            return
            
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        tasks = [
            self._send_webhook(webhook, event)
            for webhook in relevant_webhooks
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _send_webhook(self, webhook: WebhookConfig, event: IntegrationEvent):
        """Send webhook with retry logic."""
        payload = {
            'event_type': event.event_type.value,
            'timestamp': event.timestamp.isoformat(),
            'data': event.data,
            'source': event.source
        }
        
        if event.correlation_id:
            payload['correlation_id'] = event.correlation_id
            
        headers = webhook.headers.copy()
        headers['Content-Type'] = 'application/json'
        
        # Add signature if secret is configured
        if webhook.secret:
            signature = self._generate_signature(json.dumps(payload), webhook.secret)
            headers['X-Webhook-Signature'] = signature
            
        for attempt in range(webhook.retry_count):
            try:
                async with self.session.post(
                    webhook.url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=webhook.timeout)
                ) as response:
                    if response.status < 400:
                        logger.info(f"Webhook sent successfully to {webhook.url}")
                        return
                    else:
                        logger.warning(f"Webhook failed with status {response.status}: {await response.text()}")
                        
            except Exception as e:
                logger.error(f"Webhook attempt {attempt + 1} failed: {e}")
                if attempt < webhook.retry_count - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
        logger.error(f"All webhook attempts failed for {webhook.url}")
        
    def _generate_signature(self, payload: str, secret: str) -> str:
        """Generate HMAC signature for webhook security."""
        return hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

class PluginManager:
    """Plugin system for extending AI Gatekeeper functionality."""
    
    def __init__(self):
        self.plugins: Dict[str, Any] = {}
        self.hooks: Dict[str, List[Callable]] = {}
        
    def register_plugin(self, name: str, plugin: Any):
        """Register a plugin."""
        self.plugins[name] = plugin
        
        # Auto-register hooks if plugin has them
        if hasattr(plugin, 'hooks'):
            for hook_name, hook_func in plugin.hooks.items():
                self.register_hook(hook_name, hook_func)
                
    def register_hook(self, hook_name: str, hook_func: Callable):
        """Register a hook function."""
        if hook_name not in self.hooks:
            self.hooks[hook_name] = []
        self.hooks[hook_name].append(hook_func)
        
    async def execute_hooks(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Execute all hooks for a given event."""
        if hook_name not in self.hooks:
            return []
            
        results = []
        for hook_func in self.hooks[hook_name]:
            try:
                if asyncio.iscoroutinefunction(hook_func):
                    result = await hook_func(*args, **kwargs)
                else:
                    result = hook_func(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Hook {hook_name} failed: {e}")
                
        return results

class WorkflowOrchestrator:
    """Orchestrates the entire workflow system."""
    
    def __init__(self):
        self.rule_engine = RuleEngine()
        self.webhook_manager = WebhookManager()
        self.plugin_manager = PluginManager()
        self._register_default_conditions_and_actions()
        
    def _register_default_conditions_and_actions(self):
        """Register default conditions and actions."""
        
        # Default conditions
        self.rule_engine.register_condition('is_critical_priority', self._is_critical_priority)
        self.rule_engine.register_condition('low_confidence', self._low_confidence)
        self.rule_engine.register_condition('high_risk', self._high_risk)
        
        # Default actions
        self.rule_engine.register_action('notify_team', self._notify_team)
        self.rule_engine.register_action('escalate_immediately', self._escalate_immediately)
        
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process support request through workflow."""
        
        # Execute pre-processing hooks
        await self.plugin_manager.execute_hooks('pre_process', request_data)
        
        # Evaluate workflow rules
        actions = await self.rule_engine.evaluate_rules(request_data)
        
        # Execute actions
        for action_config in actions:
            await self._execute_action(action_config['action'], request_data)
            
        # Trigger events
        await self.webhook_manager.trigger_event(IntegrationEvent(
            event_type=EventType.REQUEST_PROCESSED,
            timestamp=datetime.now(),
            data=request_data
        ))
        
        # Execute post-processing hooks
        await self.plugin_manager.execute_hooks('post_process', request_data)
        
        return request_data
        
    async def _execute_action(self, action: Dict[str, Any], context: Dict[str, Any]):
        """Execute a workflow action."""
        action_type = action.get('type')
        
        if action_type == 'webhook':
            # Trigger webhook
            webhook_event = IntegrationEvent(
                event_type=EventType.REQUEST_PROCESSED,
                timestamp=datetime.now(),
                data=action.get('payload', context)
            )
            # Send to specific URL
            webhook_config = WebhookConfig(
                url=action['url'],
                events=[EventType.REQUEST_PROCESSED]
            )
            await self.webhook_manager._send_webhook(webhook_config, webhook_event)
            
        elif action_type == 'escalate':
            # Trigger escalation
            await self.webhook_manager.trigger_event(IntegrationEvent(
                event_type=EventType.ESCALATION_TRIGGERED,
                timestamp=datetime.now(),
                data={
                    'reason': action.get('reason'),
                    'priority': action.get('priority'),
                    'context': context
                }
            ))
            
    # Default condition functions
    async def _is_critical_priority(self, config: Dict[str, Any], context: Dict[str, Any]) -> bool:
        return context.get('priority') == 'critical'
        
    async def _low_confidence(self, config: Dict[str, Any], context: Dict[str, Any]) -> bool:
        threshold = config.get('threshold', 0.7)
        confidence = context.get('confidence_score', 1.0)
        return confidence < threshold
        
    async def _high_risk(self, config: Dict[str, Any], context: Dict[str, Any]) -> bool:
        threshold = config.get('threshold', 0.7)
        risk = context.get('risk_score', 0.0)
        return risk > threshold
        
    # Default action functions
    async def _notify_team(self, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'type': 'notification',
            'team': config.get('team', 'support'),
            'message': config.get('message', 'Attention required'),
            'context': context
        }
        
    async def _escalate_immediately(self, config: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'type': 'escalate',
            'priority': 'immediate',
            'reason': 'Automatic escalation triggered',
            'context': context
        }

# Flask Blueprint for configuration management
config_bp = Blueprint('config', __name__, url_prefix='/api/config')

# Global orchestrator instance
orchestrator = WorkflowOrchestrator()

@config_bp.route('/rules', methods=['GET'])
def get_rules():
    """Get current workflow rules."""
    return jsonify({
        'rules': [
            {
                'id': rule.id,
                'name': rule.name,
                'condition': rule.condition,
                'action': rule.action,
                'priority': rule.priority,
                'enabled': rule.enabled
            }
            for rule in orchestrator.rule_engine.rules
        ]
    })

@config_bp.route('/rules', methods=['POST'])
def add_rule():
    """Add new workflow rule."""
    rule_data = request.get_json()
    
    rule = WorkflowRule(
        id=rule_data['id'],
        name=rule_data['name'],
        condition=rule_data['condition'],
        action=rule_data['action'],
        priority=rule_data.get('priority', 100),
        enabled=rule_data.get('enabled', True)
    )
    
    orchestrator.rule_engine.add_rule(rule)
    
    return jsonify({'status': 'success', 'message': 'Rule added'})

@config_bp.route('/webhooks', methods=['GET'])
def get_webhooks():
    """Get webhook configurations."""
    return jsonify({
        'webhooks': [
            {
                'url': wh.url,
                'events': [e.value for e in wh.events],
                'timeout': wh.timeout,
                'retry_count': wh.retry_count,
                'enabled': wh.enabled
            }
            for wh in orchestrator.webhook_manager.webhooks
        ]
    })

@config_bp.route('/webhooks', methods=['POST'])
def add_webhook():
    """Add webhook configuration."""
    webhook_data = request.get_json()
    
    webhook = WebhookConfig(
        url=webhook_data['url'],
        events=[EventType(e) for e in webhook_data['events']],
        secret=webhook_data.get('secret'),
        headers=webhook_data.get('headers', {}),
        timeout=webhook_data.get('timeout', 30),
        retry_count=webhook_data.get('retry_count', 3),
        enabled=webhook_data.get('enabled', True)
    )
    
    orchestrator.webhook_manager.add_webhook(webhook)
    
    return jsonify({'status': 'success', 'message': 'Webhook added'})

@config_bp.route('/test-webhook', methods=['POST'])
def test_webhook():
    """Test webhook configuration."""
    webhook_data = request.get_json()
    
    # Create test event
    test_event = IntegrationEvent(
        event_type=EventType.REQUEST_RECEIVED,
        timestamp=datetime.now(),
        data={'test': True, 'message': 'Test webhook event'}
    )
    
    # Send test webhook
    webhook = WebhookConfig(
        url=webhook_data['url'],
        events=[EventType.REQUEST_RECEIVED],
        secret=webhook_data.get('secret')
    )
    
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(
            orchestrator.webhook_manager._send_webhook(webhook, test_event)
        )
        loop.close()
        
        return jsonify({'status': 'success', 'message': 'Test webhook sent'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

# Example configurations for common workflows
EXAMPLE_CONFIGURATIONS = {
    'critical_escalation': {
        'rule': WorkflowRule(
            id='critical_escalation',
            name='Escalate Critical Issues',
            condition={
                'type': 'or',
                'conditions': [
                    {'type': 'field_equals', 'field': 'priority', 'value': 'critical'},
                    {'type': 'field_contains', 'field': 'message', 'value': 'urgent'}
                ]
            },
            action={
                'type': 'webhook',
                'url': 'https://your-team-chat.com/webhook',
                'payload': {
                    'text': 'Critical support request: {{message}}',
                    'priority': 'high',
                    'user': '{{user_context.user_name}}'
                }
            },
            priority=1
        )
    },
    'low_confidence_review': {
        'rule': WorkflowRule(
            id='low_confidence_review',
            name='Review Low Confidence Solutions',
            condition={
                'type': 'custom',
                'function': 'low_confidence',
                'threshold': 0.6
            },
            action={
                'type': 'escalate',
                'reason': 'Low confidence solution requires human review',
                'priority': 'medium'
            },
            priority=50
        )
    }
}

# Export for use in main application
__all__ = [
    'orchestrator',
    'config_bp',
    'WorkflowRule',
    'WebhookConfig',
    'IntegrationEvent',
    'EventType',
    'EXAMPLE_CONFIGURATIONS'
]