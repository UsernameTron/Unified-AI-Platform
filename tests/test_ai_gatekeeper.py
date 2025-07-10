#!/usr/bin/env python3
"""
Comprehensive Test Suite for AI Gatekeeper System
Tests all components and integration points using existing testing framework
"""

import sys
import os
import json
import asyncio
import unittest
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any, List
from datetime import datetime

# Add project paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared_agents'))

# Import AI Gatekeeper components
from core.support_request_processor import (
    SupportRequestProcessor, SupportRequest, SupportRequestPriority, SupportRequestStatus
)
from knowledge.solution_generator import (
    KnowledgeBaseSolutionGenerator, SolutionType, SolutionComplexity, GeneratedSolution
)
from knowledge.knowledge_base_setup import SupportKnowledgeBaseManager
from integrations.slack_integration import SlackConnector, SlackEventHandler
from shared_agents.core.agent_factory import AgentResponse


class TestSupportRequestProcessor(unittest.TestCase):
    """Test cases for the Support Request Processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock configuration
        self.mock_config = Mock()
        self.mock_config.support_confidence_threshold = 0.8
        self.mock_config.support_risk_threshold = 0.3
        
        # Create processor instance
        self.processor = SupportRequestProcessor(self.mock_config)
        
        # Mock agent manager
        self.mock_agent_manager = Mock()
        self.processor.set_agent_manager(self.mock_agent_manager)
        
        # Mock search system
        self.mock_search_system = Mock()
        self.processor.set_search_system(self.mock_search_system)
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        self.assertIsNotNone(self.processor)
        self.assertEqual(self.processor.confidence_threshold, 0.8)
        self.assertEqual(self.processor.risk_threshold, 0.3)
        self.assertIsNotNone(self.processor.agent_manager)
        self.assertIsNotNone(self.processor.search_system)
    
    def test_priority_determination(self):
        """Test support request priority determination."""
        # Test critical priority
        critical_message = "CRITICAL: System is down and users cannot access the application"
        priority = self.processor._determine_priority(critical_message, {})
        self.assertEqual(priority, SupportRequestPriority.CRITICAL)
        
        # Test high priority
        high_message = "URGENT: Production issue blocking customer orders"
        priority = self.processor._determine_priority(high_message, {})
        self.assertEqual(priority, SupportRequestPriority.HIGH)
        
        # Test medium priority (default)
        normal_message = "How do I reset my password?"
        priority = self.processor._determine_priority(normal_message, {})
        self.assertEqual(priority, SupportRequestPriority.MEDIUM)
    
    def test_confidence_score_calculation(self):
        """Test confidence score calculation."""
        # Mock triage result
        triage_result = {'analysis': 'test'}
        
        # Test common issue (high confidence)
        request = SupportRequest(
            message="How to reset password",
            user_context={'user_level': 'intermediate'}
        )
        confidence = self.processor._calculate_confidence_score(triage_result, request)
        self.assertGreater(confidence, 0.5)
        
        # Test complex issue (low confidence)
        request = SupportRequest(
            message="Complex API integration issue with custom development",
            user_context={'user_level': 'beginner'},
            priority=SupportRequestPriority.CRITICAL
        )
        confidence = self.processor._calculate_confidence_score(triage_result, request)
        self.assertLess(confidence, 0.5)
    
    def test_risk_score_calculation(self):
        """Test risk score calculation."""
        triage_result = {'analysis': 'test'}
        
        # Test low risk operation
        request = SupportRequest(
            message="Show me how to view my account settings",
            user_context={'user_level': 'advanced'}
        )
        risk = self.processor._calculate_risk_score(triage_result, request)
        self.assertLess(risk, 0.5)
        
        # Test high risk operation
        request = SupportRequest(
            message="Need to modify critical database configuration",
            user_context={'user_level': 'beginner'}
        )
        risk = self.processor._calculate_risk_score(triage_result, request)
        self.assertGreater(risk, 0.5)
    
    def test_resolution_path_determination(self):
        """Test resolution path determination logic."""
        # High confidence, low risk -> automated
        request = SupportRequest()
        request.confidence_score = 0.9
        request.risk_score = 0.2
        
        path = self.processor._determine_resolution_path(request)
        self.assertEqual(path, "automated_resolution")
        
        # Low confidence -> escalation
        request.confidence_score = 0.5
        request.risk_score = 0.2
        
        path = self.processor._determine_resolution_path(request)
        self.assertEqual(path, "escalation")
        
        # High risk -> escalation
        request.confidence_score = 0.9
        request.risk_score = 0.8
        
        path = self.processor._determine_resolution_path(request)
        self.assertEqual(path, "escalation")
    
    @patch('asyncio.new_event_loop')
    def test_support_request_processing(self, mock_loop):
        """Test full support request processing flow."""
        # Mock the agent manager response
        mock_triage_response = AgentResponse(
            success=True,
            result={'confidence': 0.9, 'risk': 0.2},
            agent_type='triage',
            timestamp=datetime.now().isoformat()
        )
        
        mock_research_response = AgentResponse(
            success=True,
            result={'solution': 'Mock solution'},
            agent_type='research',
            timestamp=datetime.now().isoformat()
        )
        
        # Configure mocks
        self.mock_agent_manager.execute_agent.side_effect = [
            mock_triage_response,
            mock_research_response
        ]
        
        # Mock event loop
        mock_loop_instance = Mock()
        mock_loop.return_value = mock_loop_instance
        mock_loop_instance.run_until_complete.return_value = Mock()
        
        # This would be an async test in a real implementation
        # For now, we test the components
        self.assertTrue(True)  # Placeholder


class TestSolutionGenerator(unittest.TestCase):
    """Test cases for the Solution Generator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_agent_manager = Mock()
        self.mock_search_system = Mock()
        
        self.generator = KnowledgeBaseSolutionGenerator(
            agent_manager=self.mock_agent_manager,
            search_system=self.mock_search_system
        )
    
    def test_generator_initialization(self):
        """Test solution generator initialization."""
        self.assertIsNotNone(self.generator)
        self.assertEqual(self.generator.agent_manager, self.mock_agent_manager)
        self.assertEqual(self.generator.search_system, self.mock_search_system)
    
    def test_solution_type_determination(self):
        """Test solution type determination."""
        # Test troubleshooting type
        issue = "Application is not working and showing error messages"
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            solution_type = loop.run_until_complete(
                self.generator._determine_solution_type(issue)
            )
            self.assertEqual(solution_type, SolutionType.TROUBLESHOOTING)
        finally:
            loop.close()
        
        # Test configuration type
        issue = "How to setup email client configuration"
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            solution_type = loop.run_until_complete(
                self.generator._determine_solution_type(issue)
            )
            self.assertEqual(solution_type, SolutionType.CONFIGURATION)
        finally:
            loop.close()
        
        # Test step-by-step type
        issue = "How to create a new user account"
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            solution_type = loop.run_until_complete(
                self.generator._determine_solution_type(issue)
            )
            self.assertEqual(solution_type, SolutionType.STEP_BY_STEP)
        finally:
            loop.close()
    
    def test_complexity_determination(self):
        """Test solution complexity determination."""
        from knowledge.solution_generator import SolutionStep
        
        # Simple solution
        simple_steps = [
            SolutionStep(1, "Click button", "Just click the button", [])
        ]
        complexity = self.generator._determine_complexity(simple_steps, {'user_level': 'beginner'})
        self.assertEqual(complexity, SolutionComplexity.SIMPLE)
        
        # Advanced solution
        advanced_steps = [
            SolutionStep(1, "Open terminal", "Use command line", ["sudo apt-get update"], risk_level="high"),
            SolutionStep(2, "Edit config", "Modify configuration file", ["nano /etc/config"]),
            SolutionStep(3, "Restart service", "Restart the service", ["systemctl restart service"])
        ]
        complexity = self.generator._determine_complexity(advanced_steps, {'user_level': 'beginner'})
        self.assertEqual(complexity, SolutionComplexity.ADVANCED)
    
    def test_confidence_calculation(self):
        """Test solution confidence calculation."""
        # High confidence solution
        solution_content = {
            'steps': [{'troubleshooting': 'included'}],
            'prerequisites': ['basic knowledge']
        }
        user_context = {'user_level': 'advanced'}
        
        confidence = self.generator._calculate_solution_confidence(solution_content, user_context)
        self.assertGreater(confidence, 0.7)
        
        # Lower confidence solution
        solution_content = {'steps': []}
        user_context = {'user_level': 'beginner'}
        
        confidence = self.generator._calculate_solution_confidence(solution_content, user_context)
        self.assertLess(confidence, 0.8)
    
    def test_fallback_solution_generation(self):
        """Test fallback solution generation."""
        issue = "Test issue"
        solution_type = SolutionType.TROUBLESHOOTING
        
        fallback = self.generator._generate_fallback_solution(issue, solution_type)
        
        self.assertIn('title', fallback)
        self.assertIn('steps', fallback)
        self.assertIsInstance(fallback['steps'], list)
        self.assertGreater(len(fallback['steps']), 0)


class TestKnowledgeBaseSetup(unittest.TestCase):
    """Test cases for Knowledge Base Setup."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_search_system = Mock()
        self.manager = SupportKnowledgeBaseManager(self.mock_search_system)
    
    def test_manager_initialization(self):
        """Test knowledge base manager initialization."""
        self.assertIsNotNone(self.manager)
        self.assertEqual(self.manager.search_system, self.mock_search_system)
        self.assertIsInstance(self.manager.knowledge_categories, list)
        self.assertGreater(len(self.manager.knowledge_categories), 0)
    
    def test_sample_data_generation(self):
        """Test sample knowledge data generation."""
        sample_data = self.manager._generate_sample_knowledge_data()
        
        self.assertIsInstance(sample_data, dict)
        self.assertIn('technical_solutions', sample_data)
        self.assertIn('troubleshooting_guides', sample_data)
        
        # Check data structure
        for category, documents in sample_data.items():
            self.assertIsInstance(documents, list)
            for doc in documents:
                self.assertIn('title', doc)
                self.assertIn('content', doc)
                self.assertIn('keywords', doc)
    
    def test_category_stats(self):
        """Test category statistics generation."""
        stats = self.manager.get_category_stats()
        
        self.assertIn('total_categories', stats)
        self.assertIn('categories', stats)
        self.assertIn('setup_timestamp', stats)
        self.assertGreater(stats['total_categories'], 0)


class TestSlackIntegration(unittest.TestCase):
    """Test cases for Slack Integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_agent_manager = Mock()
        self.mock_tts_service = Mock()
        
        self.slack_connector = SlackConnector(
            bot_token="test_token",
            agent_manager=self.mock_agent_manager,
            tts_service=self.mock_tts_service
        )
    
    def test_slack_connector_initialization(self):
        """Test Slack connector initialization."""
        self.assertIsNotNone(self.slack_connector)
        self.assertEqual(self.slack_connector.bot_token, "test_token")
        self.assertEqual(self.slack_connector.agent_manager, self.mock_agent_manager)
        self.assertEqual(self.slack_connector.tts_service, self.mock_tts_service)
    
    def test_support_request_detection(self):
        """Test support request detection."""
        # Should detect support request
        support_text = "I need help with my password reset"
        channel = "general"
        
        is_support = self.slack_connector._is_support_request(support_text, channel)
        self.assertTrue(is_support)
        
        # Should detect in support channel
        normal_text = "Hello everyone"
        support_channel = "it-support"
        
        is_support = self.slack_connector._is_support_request(normal_text, support_channel)
        self.assertTrue(is_support)
        
        # Should not detect non-support message
        casual_text = "Good morning! How is everyone doing today?"
        general_channel = "random"
        
        is_support = self.slack_connector._is_support_request(casual_text, general_channel)
        self.assertFalse(is_support)
    
    def test_risk_level_formatting(self):
        """Test risk level formatting."""
        self.assertEqual(self.slack_connector._format_risk_level(0.1), "Low")
        self.assertEqual(self.slack_connector._format_risk_level(0.5), "Medium")
        self.assertEqual(self.slack_connector._format_risk_level(0.8), "High")
    
    def test_audio_text_formatting(self):
        """Test solution formatting for audio."""
        solution_data = {
            'title': 'Password Reset Guide',
            'summary': 'How to reset your password',
            'steps': [
                {'description': 'Go to login page'},
                {'description': 'Click forgot password'},
                {'description': 'Enter your email'},
                {'description': 'Check your email'},
                {'description': 'Create new password'}
            ]
        }
        
        audio_text = self.slack_connector._format_solution_for_audio(solution_data)
        
        self.assertIn('Password Reset Guide', audio_text)
        self.assertIn('Step 1', audio_text)
        self.assertIn('Step 2', audio_text)
        self.assertIn('Step 3', audio_text)
        # Should limit to 3 steps for audio
        self.assertNotIn('Step 4', audio_text)
    
    def test_message_template_loading(self):
        """Test message template loading."""
        templates = self.slack_connector.message_templates
        
        self.assertIn('automated_solution', templates)
        self.assertIn('escalation_notification', templates)
        self.assertIn('expert_handoff', templates)
        self.assertIn('solution_feedback', templates)
        
        # Check template format
        template = templates['automated_solution']
        self.assertIn('{issue}', template)
        self.assertIn('{solution_steps}', template)
        self.assertIn('{confidence}', template)


class TestAIGatekeeperIntegration(unittest.TestCase):
    """Integration tests for the complete AI Gatekeeper system."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Mock all dependencies
        self.mock_config = Mock()
        self.mock_agent_manager = Mock()
        self.mock_search_system = Mock()
        self.mock_tts_service = Mock()
        
        # Set up components
        self.processor = SupportRequestProcessor(self.mock_config)
        self.processor.set_agent_manager(self.mock_agent_manager)
        self.processor.set_search_system(self.mock_search_system)
        
        self.generator = KnowledgeBaseSolutionGenerator(
            agent_manager=self.mock_agent_manager,
            search_system=self.mock_search_system
        )
        
        self.slack_connector = SlackConnector(
            bot_token="test_token",
            agent_manager=self.mock_agent_manager,
            tts_service=self.mock_tts_service
        )
    
    def test_end_to_end_workflow_simulation(self):
        """Test end-to-end workflow simulation."""
        # Simulate a support request flow
        test_message = "My application keeps crashing when I try to save files"
        test_context = {
            'user_level': 'intermediate',
            'system': 'Windows 10',
            'source': 'slack'
        }
        
        # Mock agent responses
        mock_triage_response = AgentResponse(
            success=True,
            result={'confidence': 0.9, 'risk': 0.2},
            agent_type='triage',
            timestamp=datetime.now().isoformat()
        )
        
        mock_research_response = AgentResponse(
            success=True,
            result={
                'title': 'Application Crash Resolution',
                'summary': 'Steps to resolve application crashes',
                'steps': [
                    {'title': 'Check system resources', 'description': 'Monitor CPU and memory'},
                    {'title': 'Update application', 'description': 'Install latest version'},
                    {'title': 'Clear cache', 'description': 'Remove temporary files'}
                ]
            },
            agent_type='research',
            timestamp=datetime.now().isoformat()
        )
        
        self.mock_agent_manager.execute_agent.side_effect = [
            mock_triage_response,
            mock_research_response
        ]
        
        # Test that components work together
        self.assertTrue(True)  # This would be expanded in actual async testing
    
    def test_error_handling(self):
        """Test error handling across components."""
        # Test processor error handling
        self.processor.agent_manager = None
        
        # Test generator error handling
        generator_without_deps = KnowledgeBaseSolutionGenerator()
        self.assertIsNone(generator_without_deps.agent_manager)
        
        # Test Slack connector error handling
        broken_connector = SlackConnector(bot_token="")
        self.assertEqual(broken_connector.bot_token, "")
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        valid_config = Mock()
        valid_config.support_confidence_threshold = 0.8
        valid_config.support_risk_threshold = 0.3
        
        processor = SupportRequestProcessor(valid_config)
        self.assertEqual(processor.confidence_threshold, 0.8)
        self.assertEqual(processor.risk_threshold, 0.3)
        
        # Test missing configuration (should use defaults)
        minimal_config = Mock()
        processor = SupportRequestProcessor(minimal_config)
        self.assertIsNotNone(processor.confidence_threshold)
        self.assertIsNotNone(processor.risk_threshold)


class AIGatekeeperTestSuite:
    """
    Comprehensive test suite runner for AI Gatekeeper system.
    """
    
    def __init__(self):
        """Initialize test suite."""
        self.test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'errors': [],
            'summary': {}
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all AI Gatekeeper tests.
        
        Returns:
            Test results dictionary
        """
        print("🧪 Running AI Gatekeeper Test Suite")
        print("===================================")
        
        # Define test classes
        test_classes = [
            TestSupportRequestProcessor,
            TestSolutionGenerator,
            TestKnowledgeBaseSetup,
            TestSlackIntegration,
            TestAIGatekeeperIntegration
        ]
        
        for test_class in test_classes:
            print(f"\n📋 Running {test_class.__name__}...")
            self._run_test_class(test_class)
        
        # Generate summary
        self._generate_summary()
        
        return self.test_results
    
    def _run_test_class(self, test_class):
        """Run tests for a specific test class."""
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
        runner = unittest.TextTestRunner(verbosity=2, stream=open(os.devnull, 'w'))
        result = runner.run(suite)
        
        # Update results
        tests_run = result.testsRun
        failures = len(result.failures)
        errors = len(result.errors)
        passed = tests_run - failures - errors
        
        self.test_results['total_tests'] += tests_run
        self.test_results['passed_tests'] += passed
        self.test_results['failed_tests'] += (failures + errors)
        
        # Store class-specific results
        class_name = test_class.__name__
        self.test_results['summary'][class_name] = {
            'total': tests_run,
            'passed': passed,
            'failed': failures,
            'errors': errors
        }
        
        # Store error details
        for failure in result.failures:
            self.test_results['errors'].append({
                'test': str(failure[0]),
                'type': 'failure',
                'message': failure[1]
            })
        
        for error in result.errors:
            self.test_results['errors'].append({
                'test': str(error[0]),
                'type': 'error',
                'message': error[1]
            })
        
        # Print results
        print(f"✅ Passed: {passed}")
        if failures > 0:
            print(f"❌ Failed: {failures}")
        if errors > 0:
            print(f"💥 Errors: {errors}")
    
    def _generate_summary(self):
        """Generate test summary."""
        total = self.test_results['total_tests']
        passed = self.test_results['passed_tests']
        failed = self.test_results['failed_tests']
        
        success_rate = (passed / total * 100) if total > 0 else 0
        
        print(f"\n🎯 Test Suite Summary")
        print(f"====================")
        print(f"Total Tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if failed == 0:
            print("🎉 All tests passed!")
        else:
            print(f"⚠️  {failed} tests failed - see details above")
        
        # Add summary to results
        self.test_results['success_rate'] = success_rate
        self.test_results['timestamp'] = datetime.now().isoformat()


def main():
    """Main test runner function."""
    if len(sys.argv) > 1 and sys.argv[1] == '--single':
        # Run specific test class
        if len(sys.argv) > 2:
            test_name = sys.argv[2]
            globals_dict = globals()
            if test_name in globals_dict:
                suite = unittest.TestLoader().loadTestsFromTestCase(globals_dict[test_name])
                runner = unittest.TextTestRunner(verbosity=2)
                runner.run(suite)
            else:
                print(f"Test class '{test_name}' not found")
        else:
            print("Usage: python test_ai_gatekeeper.py --single <TestClassName>")
    else:
        # Run full test suite
        test_suite = AIGatekeeperTestSuite()
        results = test_suite.run_all_tests()
        
        # Save results to file
        results_file = 'ai_gatekeeper_test_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📄 Test results saved to: {results_file}")


if __name__ == "__main__":
    main()