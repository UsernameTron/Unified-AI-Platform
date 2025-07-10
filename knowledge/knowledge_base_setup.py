"""
Knowledge Base Setup for AI Gatekeeper System
Configures vector database for support knowledge storage and retrieval
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

# Support knowledge categories
SUPPORT_KNOWLEDGE_CATEGORIES = [
    'technical_solutions',
    'troubleshooting_guides', 
    'configuration_guides',
    'user_documentation',
    'escalation_procedures',
    'best_practices',
    'common_issues',
    'system_requirements',
    'installation_guides',
    'api_documentation'
]


class SupportKnowledgeBaseManager:
    """
    Manages the setup and configuration of the support knowledge base
    using the existing vector database infrastructure.
    """
    
    def __init__(self, search_system=None):
        """Initialize the knowledge base manager."""
        self.search_system = search_system
        self.knowledge_categories = SUPPORT_KNOWLEDGE_CATEGORIES
        
    async def setup_support_knowledge_base(self) -> Dict[str, Any]:
        """
        Set up the complete support knowledge base with all categories.
        
        Returns:
            Dictionary with setup results and status
        """
        if not self.search_system:
            raise ValueError("Search system not initialized")
        
        setup_results = {
            'status': 'success',
            'categories_created': [],
            'categories_failed': [],
            'sample_data_loaded': [],
            'total_documents': 0,
            'setup_timestamp': datetime.now().isoformat()
        }
        
        print("🔧 Setting up AI Gatekeeper knowledge base...")
        
        # Create vector stores for each knowledge category
        for category in self.knowledge_categories:
            try:
                await self._create_knowledge_category(category)
                setup_results['categories_created'].append(category)
                print(f"✅ Created knowledge category: {category}")
            except Exception as e:
                setup_results['categories_failed'].append({
                    'category': category,
                    'error': str(e)
                })
                print(f"❌ Failed to create category {category}: {e}")
        
        # Load sample knowledge data
        sample_data_results = await self._load_sample_knowledge_data()
        setup_results['sample_data_loaded'] = sample_data_results['loaded_categories']
        setup_results['total_documents'] = sample_data_results['total_documents']
        
        if setup_results['categories_failed']:
            setup_results['status'] = 'partial_success'
        
        print(f"🎯 Knowledge base setup completed: {len(setup_results['categories_created'])} categories created")
        
        return setup_results
    
    async def _create_knowledge_category(self, category_name: str) -> None:
        """
        Create a vector store for a specific knowledge category.
        
        Args:
            category_name: Name of the knowledge category
        """
        try:
            # Use existing search system to create vector store
            await self.search_system.create_vector_store(category_name)
            
            # Set category-specific metadata
            category_metadata = {
                'category': category_name,
                'created_for': 'ai_gatekeeper',
                'type': 'support_knowledge',
                'created_at': datetime.now().isoformat()
            }
            
            # Store category metadata (if supported by search system)
            if hasattr(self.search_system, 'set_vector_store_metadata'):
                await self.search_system.set_vector_store_metadata(category_name, category_metadata)
            
        except Exception as e:
            print(f"Error creating knowledge category {category_name}: {e}")
            raise
    
    async def _load_sample_knowledge_data(self) -> Dict[str, Any]:
        """
        Load sample knowledge data into the vector stores.
        
        Returns:
            Dictionary with loading results
        """
        loading_results = {
            'loaded_categories': [],
            'failed_categories': [],
            'total_documents': 0
        }
        
        sample_data = self._generate_sample_knowledge_data()
        
        for category, documents in sample_data.items():
            try:
                # Upload documents to the category vector store
                upload_result = await self._upload_documents_to_category(category, documents)
                
                if upload_result['success']:
                    loading_results['loaded_categories'].append({
                        'category': category,
                        'document_count': len(documents)
                    })
                    loading_results['total_documents'] += len(documents)
                    print(f"📚 Loaded {len(documents)} documents into {category}")
                else:
                    loading_results['failed_categories'].append({
                        'category': category,
                        'error': upload_result.get('error', 'Unknown error')
                    })
                    
            except Exception as e:
                loading_results['failed_categories'].append({
                    'category': category,
                    'error': str(e)
                })
                print(f"❌ Failed to load data into {category}: {e}")
        
        return loading_results
    
    async def _upload_documents_to_category(self, category: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Upload documents to a specific category vector store.
        
        Args:
            category: Knowledge category name
            documents: List of documents to upload
            
        Returns:
            Upload result dictionary
        """
        try:
            # Format documents for vector store upload
            formatted_docs = []
            
            for doc in documents:
                formatted_doc = {
                    'content': doc['content'],
                    'metadata': {
                        'title': doc['title'],
                        'category': category,
                        'type': doc.get('type', 'support_document'),
                        'keywords': doc.get('keywords', []),
                        'difficulty_level': doc.get('difficulty_level', 'intermediate'),
                        'created_at': datetime.now().isoformat(),
                        'source': 'ai_gatekeeper_setup'
                    }
                }
                formatted_docs.append(formatted_doc)
            
            # Use existing search system upload method
            # Note: This assumes the search system has an upload method
            # Adapt based on your actual search system implementation
            if hasattr(self.search_system, 'upload_documents'):
                upload_result = await self.search_system.upload_documents(category, formatted_docs)
            else:
                # Fallback method if direct upload not available
                upload_result = await self._fallback_document_upload(category, formatted_docs)
            
            return {'success': True, 'uploaded_count': len(formatted_docs)}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _fallback_document_upload(self, category: str, documents: List[Dict[str, Any]]) -> None:
        """
        Fallback method for uploading documents if direct upload not available.
        
        Args:
            category: Knowledge category name
            documents: Formatted documents to upload
        """
        # This is a fallback implementation
        # In a real implementation, you would use the actual search system's API
        print(f"Using fallback upload for {len(documents)} documents in {category}")
        
        # Store documents in a local format for now
        # In production, this would integrate with the actual vector database
        storage_path = f"/tmp/ai_gatekeeper_knowledge/{category}"
        os.makedirs(storage_path, exist_ok=True)
        
        for i, doc in enumerate(documents):
            doc_path = os.path.join(storage_path, f"doc_{i}.json")
            with open(doc_path, 'w') as f:
                json.dump(doc, f, indent=2)
    
    def _generate_sample_knowledge_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Generate sample knowledge data for initial setup.
        
        Returns:
            Dictionary mapping categories to document lists
        """
        sample_data = {
            'technical_solutions': [
                {
                    'title': 'Application Crash Troubleshooting',
                    'content': '''
                    When an application crashes unexpectedly, follow these steps:
                    1. Check system logs for error messages
                    2. Verify system requirements are met
                    3. Update application to latest version
                    4. Clear application cache and temporary files
                    5. Restart the application in safe mode
                    6. If issue persists, reinstall the application
                    ''',
                    'type': 'troubleshooting_guide',
                    'keywords': ['crash', 'application', 'troubleshooting', 'error'],
                    'difficulty_level': 'intermediate'
                },
                {
                    'title': 'Password Reset Procedure',
                    'content': '''
                    To reset a user password:
                    1. Navigate to the login page
                    2. Click "Forgot Password" link
                    3. Enter registered email address
                    4. Check email for reset instructions
                    5. Click the reset link within 24 hours
                    6. Create a new strong password
                    7. Confirm the password change
                    ''',
                    'type': 'procedure',
                    'keywords': ['password', 'reset', 'login', 'account'],
                    'difficulty_level': 'beginner'
                },
                {
                    'title': 'Network Connectivity Issues',
                    'content': '''
                    For network connectivity problems:
                    1. Check physical cable connections
                    2. Restart network equipment (router, modem)
                    3. Verify network adapter settings
                    4. Run network diagnostics
                    5. Check firewall and antivirus settings
                    6. Test with different device
                    7. Contact ISP if issue persists
                    ''',
                    'type': 'troubleshooting_guide',
                    'keywords': ['network', 'connectivity', 'internet', 'connection'],
                    'difficulty_level': 'intermediate'
                }
            ],
            
            'troubleshooting_guides': [
                {
                    'title': 'Slow Performance Diagnosis',
                    'content': '''
                    To diagnose slow system performance:
                    1. Check CPU and memory usage in Task Manager
                    2. Identify resource-intensive processes
                    3. Check available disk space (minimum 15% free)
                    4. Run disk cleanup and defragmentation
                    5. Update device drivers
                    6. Scan for malware
                    7. Consider hardware upgrade if needed
                    ''',
                    'type': 'diagnostic_guide',
                    'keywords': ['performance', 'slow', 'speed', 'optimization'],
                    'difficulty_level': 'intermediate'
                },
                {
                    'title': 'Software Installation Failures',
                    'content': '''
                    When software installation fails:
                    1. Run installer as administrator
                    2. Temporarily disable antivirus
                    3. Check system compatibility
                    4. Clear previous installation remnants
                    5. Download fresh installer copy
                    6. Check Windows Update status
                    7. Use installation troubleshooter
                    ''',
                    'type': 'troubleshooting_guide',
                    'keywords': ['installation', 'software', 'install', 'failure'],
                    'difficulty_level': 'intermediate'
                }
            ],
            
            'configuration_guides': [
                {
                    'title': 'Email Client Setup',
                    'content': '''
                    Configure email client settings:
                    1. Open email application
                    2. Go to Account Settings
                    3. Add new account
                    4. Enter email address and password
                    5. Configure incoming server (IMAP/POP3)
                    6. Configure outgoing server (SMTP)
                    7. Test send and receive functionality
                    8. Configure sync settings
                    ''',
                    'type': 'configuration_guide',
                    'keywords': ['email', 'setup', 'configuration', 'client'],
                    'difficulty_level': 'beginner'
                },
                {
                    'title': 'VPN Configuration',
                    'content': '''
                    Set up VPN connection:
                    1. Obtain VPN credentials from IT
                    2. Open Network & Internet settings
                    3. Select VPN from left menu
                    4. Click "Add a VPN connection"
                    5. Choose VPN provider
                    6. Enter connection details
                    7. Save and test connection
                    8. Configure auto-connect if needed
                    ''',
                    'type': 'configuration_guide',
                    'keywords': ['vpn', 'network', 'security', 'remote'],
                    'difficulty_level': 'advanced'
                }
            ],
            
            'common_issues': [
                {
                    'title': 'Printer Not Responding',
                    'content': '''
                    Common printer issues and solutions:
                    1. Check power and cable connections
                    2. Verify printer is set as default
                    3. Clear print queue
                    4. Restart print spooler service
                    5. Update or reinstall printer drivers
                    6. Check paper and ink/toner levels
                    7. Run printer troubleshooter
                    ''',
                    'type': 'common_issue',
                    'keywords': ['printer', 'printing', 'not responding'],
                    'difficulty_level': 'beginner'
                },
                {
                    'title': 'Browser Running Slowly',
                    'content': '''
                    Speed up slow web browser:
                    1. Close unnecessary tabs and windows
                    2. Clear browser cache and cookies
                    3. Disable unnecessary extensions
                    4. Update browser to latest version
                    5. Reset browser settings if needed
                    6. Check for malware
                    7. Consider using different browser
                    ''',
                    'type': 'common_issue',
                    'keywords': ['browser', 'slow', 'internet', 'web'],
                    'difficulty_level': 'beginner'
                }
            ],
            
            'escalation_procedures': [
                {
                    'title': 'When to Escalate to Level 2 Support',
                    'content': '''
                    Escalate to Level 2 support when:
                    1. Issue requires specialized knowledge
                    2. Hardware replacement needed
                    3. System administration access required
                    4. Security incident suspected
                    5. Customer requests supervisor
                    6. Issue affects multiple users
                    7. Resolution time exceeds SLA
                    ''',
                    'type': 'escalation_criteria',
                    'keywords': ['escalation', 'level 2', 'support', 'criteria'],
                    'difficulty_level': 'intermediate'
                },
                {
                    'title': 'Critical Issue Escalation',
                    'content': '''
                    For critical system issues:
                    1. Immediately notify supervisor
                    2. Document all symptoms and error messages
                    3. Identify affected systems and users
                    4. Estimate business impact
                    5. Implement temporary workarounds if possible
                    6. Create detailed escalation ticket
                    7. Follow up within 30 minutes
                    ''',
                    'type': 'escalation_procedure',
                    'keywords': ['critical', 'escalation', 'emergency', 'urgent'],
                    'difficulty_level': 'advanced'
                }
            ]
        }
        
        return sample_data
    
    async def add_knowledge_document(self, category: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a new document to the knowledge base.
        
        Args:
            category: Knowledge category
            document: Document data
            
        Returns:
            Result dictionary
        """
        try:
            if category not in self.knowledge_categories:
                return {'success': False, 'error': f'Invalid category: {category}'}
            
            # Format document
            formatted_doc = {
                'content': document['content'],
                'metadata': {
                    'title': document['title'],
                    'category': category,
                    'type': document.get('type', 'support_document'),
                    'keywords': document.get('keywords', []),
                    'difficulty_level': document.get('difficulty_level', 'intermediate'),
                    'created_at': datetime.now().isoformat(),
                    'source': document.get('source', 'manual_entry')
                }
            }
            
            # Upload to vector store
            upload_result = await self._upload_documents_to_category(category, [formatted_doc])
            
            return upload_result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def search_knowledge_base(self, query: str, categories: Optional[List[str]] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search across the knowledge base.
        
        Args:
            query: Search query
            categories: Optional list of categories to search
            limit: Maximum number of results
            
        Returns:
            List of search results
        """
        try:
            if not self.search_system:
                return []
            
            search_categories = categories or self.knowledge_categories
            all_results = []
            
            for category in search_categories:
                try:
                    # Use existing search system
                    results = await self.search_system.assisted_search(
                        vector_store_ids=[category],
                        query=query,
                        num_results=limit
                    )
                    
                    if results:
                        for result in results:
                            result['category'] = category
                            all_results.append(result)
                            
                except Exception as e:
                    print(f"Search error in category {category}: {e}")
            
            # Sort by relevance (implement scoring if needed)
            return all_results[:limit]
            
        except Exception as e:
            print(f"Knowledge base search error: {e}")
            return []
    
    def get_category_stats(self) -> Dict[str, Any]:
        """
        Get statistics about knowledge base categories.
        
        Returns:
            Dictionary with category statistics
        """
        return {
            'total_categories': len(self.knowledge_categories),
            'categories': self.knowledge_categories,
            'setup_timestamp': datetime.now().isoformat()
        }


async def setup_ai_gatekeeper_knowledge_base(search_system) -> Dict[str, Any]:
    """
    Convenience function to set up the AI Gatekeeper knowledge base.
    
    Args:
        search_system: Initialized search system instance
        
    Returns:
        Setup results dictionary
    """
    manager = SupportKnowledgeBaseManager(search_system)
    return await manager.setup_support_knowledge_base()


# Command-line setup script
if __name__ == "__main__":
    import sys
    
    print("🛡️ AI Gatekeeper Knowledge Base Setup")
    print("=====================================")
    
    # This would typically be run with a properly initialized search system
    print("Note: This script requires integration with your search system.")
    print("Please use the setup_ai_gatekeeper_knowledge_base() function")
    print("with your initialized search system instance.")