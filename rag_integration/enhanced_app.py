"""
Flask web application for the RAG File Search System with Analytics Integration.
"""
import os
import json
import traceback
import asyncio
import threading
from typing import Dict, Any
from flask import Flask, request, jsonify, render_template, current_app
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

from search_system import SearchSystem
from config import Config
from integrations.analytics_integration import AnalyticsIntegration
from integrations.report_ingestion import ReportIngestion


def create_app(config_name: str | None = None) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Load configuration
    if config_name is None:
        config_name = os.getenv('FLASK_ENV', 'development')
    
    config = Config(config_name)
    
    # Validate configuration before proceeding
    try:
        config.validate()
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("\n🔑 Setup Instructions:")
        print("1. Get your OpenAI API key from: https://platform.openai.com/api-keys")
        print("2. Add it to your .env file: OPENAI_API_KEY=your_actual_api_key")
        print("3. Or run: ./setup_api_key.sh for guided setup")
        print("\n📖 See README.md for more details\n")
        raise
    
    app.config.update(config.to_dict())
    
    # Enable CORS
    CORS(app)
    
    # Initialize search system
    search_system = SearchSystem(config)
    app.search_system = search_system
    
    # Initialize analytics integration
    try:
        analytics_integration = AnalyticsIntegration(search_system, "Daily_Reporting")
        app.analytics_integration = analytics_integration
        
        # Initialize report ingestion
        report_ingestion = ReportIngestion(search_system, "Daily_Reporting")
        app.report_ingestion = report_ingestion
        
        print("✅ Analytics integration initialized successfully")
    except Exception as e:
        print(f"⚠️  Analytics integration failed: {e}")
        app.analytics_integration = None
        app.report_ingestion = None
    
    # Initialize agent system
    try:
        from unified_agent_system import create_unified_system
        agent_manager = create_unified_system(search_system, analytics_integration)
        app.agent_manager = agent_manager
        
        # Register agent routes
        from agent_flask_integration import register_agent_routes
        register_agent_routes(app)
        
        print("✅ Unified agent system initialized with 12 specialized agents")
        
        # Initialize RAG-Agent integration for code improvement
        try:
            from unified_agent_system_rag_integration import integrate_rag_with_agents
            from app_code_improvement_routes import register_code_improvement_routes
            
            # Create RAG-enhanced code improvement orchestrator
            code_orchestrator = integrate_rag_with_agents(agent_manager, search_system)
            app.code_orchestrator = code_orchestrator
            
            # Register code improvement routes
            register_code_improvement_routes(app)
            
            print("🔗 RAG-Agent integration completed - Code improvement system ready")
        except Exception as e:
            print(f"⚠️  RAG-Agent integration failed: {e}")
            app.code_orchestrator = None
        
        # Initialize AI Gatekeeper system
        try:
            from integrations.ai_gatekeeper_routes import register_ai_gatekeeper_routes
            
            # Register AI Gatekeeper routes
            register_ai_gatekeeper_routes(app)
            
            print("🛡️ AI Gatekeeper system initialized - Support automation ready")
        except Exception as e:
            print(f"⚠️  AI Gatekeeper initialization failed: {e}")
            
    except Exception as e:
        print(f"⚠️  Agent system initialization failed: {e}")
        app.agent_manager = None
        app.code_orchestrator = None
    
    # Register error handlers
    register_error_handlers(app)
    
    # Register routes
    register_routes(app)
    
    return app


def register_error_handlers(app: Flask):
    """Register error handlers for the application."""
    
    @app.errorhandler(413)
    @app.errorhandler(RequestEntityTooLarge)
    def handle_file_too_large(error):
        return jsonify({
            'error': 'File too large',
            'message': 'The uploaded file exceeds the maximum size limit.',
            'max_size': app.config.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024)
        }), 413
    
    @app.errorhandler(404)
    def handle_not_found(error):
        return jsonify({
            'error': 'Not found',
            'message': 'The requested resource was not found.'
        }), 404
    
    @app.errorhandler(500)
    def handle_internal_error(error):
        return jsonify({
            'error': 'Internal server error',
            'message': 'An unexpected error occurred. Please try again later.'
        }), 500


def register_routes(app: Flask):
    """Register application routes."""
    
    @app.route('/')
    def index():
        """Render the main application page."""
        return render_template('index.html')
    
    @app.route('/dashboard')
    def dashboard():
        """Render the AI agent dashboard page."""
        return render_template('agent_dashboard.html')
    
    @app.route('/health')
    def health_check():
        """Health check endpoint."""
        try:
            # Test OpenAI API connection
            current_app.search_system.client.models.list()
            api_status = 'connected'
        except Exception as e:
            api_status = f'error: {str(e)}'
        
        return jsonify({
            'status': 'healthy' if api_status == 'connected' else 'degraded',
            'service': 'RAG File Search System',
            'version': '1.0.0',
            'openai_api': api_status
        })
    
    @app.route('/api/test-api-key')
    def test_api_key():
        """Test if the OpenAI API key is working."""
        try:
            # Simple test to verify API key works
            models = current_app.search_system.client.models.list()
            return jsonify({
                'status': 'success',
                'message': 'OpenAI API key is working correctly',
                'available_models': len(models.data)
            })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'OpenAI API key test failed: {str(e)}',
                'suggestion': 'Please check your OPENAI_API_KEY in the .env file'
            }), 401
    
    @app.route('/api/upload', methods=['POST'])
    def upload_file():
        """Upload a file to the system."""
        try:
            if 'file' not in request.files and 'url' not in request.form:
                return jsonify({
                    'error': 'No file or URL provided',
                    'message': 'Please provide either a file or URL to upload.'
                }), 400
            
            vector_store_id = request.form.get('vector_store_id')
            if not vector_store_id:
                return jsonify({
                    'error': 'Vector store ID required',
                    'message': 'Please provide a vector store ID.'
                }), 400
            
            # Handle file upload
            if 'file' in request.files:
                file = request.files['file']
                if file.filename == '':
                    return jsonify({
                        'error': 'No file selected',
                        'message': 'Please select a file to upload.'
                    }), 400
                
                # Save file temporarily
                filename = secure_filename(file.filename)
                temp_path = os.path.join('/tmp', filename)
                file.save(temp_path)
                
                # Upload to vector store
                result = app.search_system.upload_file(temp_path, vector_store_id)
                
                # Clean up temporary file
                os.unlink(temp_path)
                
            # Handle URL upload
            elif 'url' in request.form:
                url = request.form['url']
                result = app.search_system.upload_from_url(url, vector_store_id)
            
            return jsonify({
                'success': True,
                'message': 'File uploaded successfully',
                'file_id': result.get('id'),
                'filename': result.get('filename')
            })
            
        except Exception as e:
            current_app.logger.error(f"Upload error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({
                'error': 'Upload failed',
                'message': str(e)
            }), 500
    
    @app.route('/api/dashboard/upload', methods=['POST'])
    def dashboard_upload():
        """Simplified file upload for dashboard document analysis."""
        try:
            if 'file' not in request.files:
                return jsonify({
                    'error': 'No file provided',
                    'message': 'Please select a file to upload.'
                }), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({
                    'error': 'No file selected',
                    'message': 'Please select a file to upload.'
                }), 400
            
            # Get or create default vector store for dashboard uploads
            vector_stores = app.search_system.list_vector_stores()
            dashboard_store = None
            
            # Look for existing dashboard store
            for store in vector_stores:
                if store.get('name') == 'Dashboard_Documents':
                    dashboard_store = store
                    break
            
            # Create dashboard store if it doesn't exist
            if not dashboard_store:
                dashboard_store = app.search_system.create_vector_store('Dashboard_Documents')
            
            # Save file temporarily
            filename = secure_filename(file.filename)
            temp_path = os.path.join('/tmp', filename)
            file.save(temp_path)
            
            # Upload to vector store
            result = app.search_system.upload_file(temp_path, dashboard_store['id'])
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            return jsonify({
                'success': True,
                'message': 'Document uploaded successfully and ready for analysis',
                'file_id': result.get('id'),
                'filename': result.get('filename'),
                'vector_store_id': dashboard_store['id']
            })
            
        except Exception as e:
            current_app.logger.error(f"Dashboard upload error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({
                'error': 'Upload failed',
                'message': str(e)
            }), 500
    
    @app.route('/api/dashboard/analyze', methods=['POST'])
    def dashboard_analyze():
        """Analyze uploaded documents using appropriate AI agents."""
        try:
            data = request.get_json()
            query = data.get('query', '')
            agent_type = data.get('agent_type', 'research')
            file_context = data.get('file_context', False)
            
            if not query:
                return jsonify({
                    'error': 'Query required',
                    'message': 'Please provide a question or analysis request.'
                }), 400
            
            # Get dashboard vector store for context
            vector_stores = app.search_system.list_vector_stores()
            dashboard_store_id = None
            
            for store in vector_stores:
                if store.get('name') == 'Dashboard_Documents':
                    dashboard_store_id = store['id']
                    break
            
            # If we have documents and file_context is requested, include RAG search
            context = ""
            if file_context and dashboard_store_id:
                try:
                    # Perform semantic search on uploaded documents
                    search_results = app.search_system.semantic_search(
                        dashboard_store_id, query, max_results=5
                    )
                    if search_results and hasattr(search_results, 'data'):
                        context = "\n\nRelevant document context:\n"
                        for result in search_results.data[:3]:  # Top 3 results
                            context += f"- {result.content[:200]}...\n"
                except Exception as search_error:
                    current_app.logger.warning(f"Context search failed: {search_error}")
            
            # Use appropriate agent based on type
            if not app.agent_manager:
                return jsonify({
                    'error': 'Agent system not available',
                    'message': 'AI agents are not initialized.'
                }), 503
            
            # Prepare enhanced query with context
            enhanced_query = query + context
            
            # Route to appropriate agent
            agent_mapping = {
                'research': 'ResearchAgent',
                'ceo': 'CEOAgent', 
                'coaching': 'CoachingAgent',
                'performance': 'PerformanceAgent',
                'analytics': 'AnalyticsAgent',
                'triage': 'TriageAgent'
            }
            
            agent_name = agent_mapping.get(agent_type, 'ResearchAgent')
            
            # Execute agent
            result = app.agent_manager.process_request(enhanced_query)
            
            return jsonify({
                'success': True,
                'query': query,
                'agent_type': agent_type,
                'response': result.result if hasattr(result, 'result') else str(result),
                'agent_name': result.agent_name if hasattr(result, 'agent_name') else agent_name,
                'execution_time': result.execution_time if hasattr(result, 'execution_time') else 0,
                'used_file_context': bool(context)
            })
            
        except Exception as e:
            current_app.logger.error(f"Dashboard analyze error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({
                'error': 'Analysis failed',
                'message': str(e)
            }), 500
    
    @app.route('/api/vector-stores', methods=['GET'])
    def list_vector_stores():
        """List all vector stores."""
        try:
            stores = app.search_system.list_vector_stores()
            return jsonify({
                'success': True,
                'vector_stores': stores
            })
        except Exception as e:
            current_app.logger.error(f"List vector stores error: {str(e)}")
            return jsonify({
                'error': 'Failed to list vector stores',
                'message': str(e)
            }), 500
    
    @app.route('/api/vector-stores', methods=['POST'])
    def create_vector_store():
        """Create a new vector store."""
        try:
            data = request.get_json()
            name = data.get('name')
            if not name:
                return jsonify({
                    'error': 'Name required',
                    'message': 'Vector store name is required.'
                }), 400
            
            store = app.search_system.create_vector_store(name)
            return jsonify({
                'success': True,
                'message': 'Vector store created successfully',
                'vector_store': store
            })
            
        except Exception as e:
            current_app.logger.error(f"Create vector store error: {str(e)}")
            return jsonify({
                'error': 'Failed to create vector store',
                'message': str(e)
            }), 500
    
    @app.route('/api/vector-stores/<store_id>', methods=['DELETE'])
    def delete_vector_store(store_id):
        """Delete a vector store."""
        try:
            app.search_system.delete_vector_store(store_id)
            return jsonify({
                'success': True,
                'message': 'Vector store deleted successfully'
            })
            
        except Exception as e:
            current_app.logger.error(f"Delete vector store error: {str(e)}")
            return jsonify({
                'error': 'Failed to delete vector store',
                'message': str(e)
            }), 500
    
    @app.route('/api/search', methods=['POST'])
    def search():
        """Perform a search query."""
        try:
            data = request.get_json()
            query = data.get('query')
            vector_store_ids = data.get('vector_store_ids', [])
            search_type = data.get('search_type', 'assisted')
            max_results = data.get('max_results', 10)
            
            if not query:
                return jsonify({
                    'error': 'Query required',
                    'message': 'Search query is required.'
                }), 400
            
            if not vector_store_ids:
                return jsonify({
                    'error': 'Vector store IDs required',
                    'message': 'At least one vector store ID is required.'
                }), 400
            
            # Perform search based on type
            if search_type == 'semantic':
                results = app.search_system.semantic_search(
                    vector_store_ids[0], query, max_results
                )
            elif search_type == 'assisted':
                results = app.search_system.assisted_search(
                    vector_store_ids, query
                )
            else:
                return jsonify({
                    'error': 'Invalid search type',
                    'message': 'Search type must be "semantic" or "assisted".'
                }), 400
            
            # Handle serialization of OpenAI Response objects
            if hasattr(results, '__dict__'):
                try:
                    # Convert Response object to dictionary for JSON serialization
                    results = {
                        'content': str(results),
                        'type': type(results).__name__
                    }
                except Exception as serialize_error:
                    current_app.logger.warning(f"Serialization fallback: {serialize_error}")
                    results = {'content': str(results)}
            
            return jsonify({
                'success': True,
                'query': query,
                'search_type': search_type,
                'results': results
            })
            
        except Exception as e:
            current_app.logger.error(f"Search error: {str(e)}\n{traceback.format_exc()}")
            return jsonify({
                'error': 'Search failed',
                'message': str(e)
            }), 500
    
    @app.route('/api/vector-stores/<store_id>/status')
    def vector_store_status(store_id):
        """Get vector store status."""
        try:
            status = app.search_system.get_vector_store_status(store_id)
            return jsonify({
                'success': True,
                'status': status
            })
        except Exception as e:
            current_app.logger.error(f"Status check error: {str(e)}")
            return jsonify({
                'error': 'Failed to get status',
                'message': str(e)
            }), 500

    # Analytics Integration Routes
    @app.route('/analytics')
    def analytics_dashboard():
        """Render the analytics dashboard page."""
        return render_template('analytics.html')

    # Code Improvement Dashboard Route
    @app.route('/code-improvement')
    def code_improvement_dashboard():
        """Render the code improvement dashboard page."""
        return render_template('code_improvement.html')

    @app.route('/api/analytics/dashboard')
    def get_analytics_dashboard():
        """Get analytics dashboard data."""
        try:
            if not app.analytics_integration:
                return jsonify({
                    'error': 'Analytics not available',
                    'message': 'Analytics integration is not initialized.'
                }), 503

            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                dashboard_data = loop.run_until_complete(
                    app.analytics_integration.get_analytics_dashboard_data()
                )
            finally:
                loop.close()

            return jsonify({
                'success': True,
                'data': dashboard_data
            })

        except Exception as e:
            current_app.logger.error(f"Analytics dashboard error: {str(e)}")
            return jsonify({
                'error': 'Failed to get analytics data',
                'message': str(e)
            }), 500

    @app.route('/api/analytics/search', methods=['POST'])
    def analytics_search():
        """Perform business intelligence search."""
        try:
            if not app.analytics_integration:
                return jsonify({
                    'error': 'Analytics not available',
                    'message': 'Analytics integration is not initialized.'
                }), 503

            data = request.get_json()
            query = data.get('query')
            search_type = data.get('search_type', 'assisted')

            if not query:
                return jsonify({
                    'error': 'Query required',
                    'message': 'Search query is required.'
                }), 400

            # Run async function in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(
                    app.analytics_integration.search_business_intelligence(query, search_type)
                )
            finally:
                loop.close()

            return jsonify({
                'success': True,
                'results': results
            })

        except Exception as e:
            current_app.logger.error(f"Analytics search error: {str(e)}")
            return jsonify({
                'error': 'Analytics search failed',
                'message': str(e)
            }), 500

    @app.route('/api/analytics/ingestion/status')
    def get_ingestion_status():
        """Get report ingestion status."""
        try:
            if not app.report_ingestion:
                return jsonify({
                    'error': 'Report ingestion not available',
                    'message': 'Report ingestion is not initialized.'
                }), 503

            status = app.report_ingestion.get_ingestion_status()
            return jsonify({
                'success': True,
                'status': status
            })

        except Exception as e:
            current_app.logger.error(f"Ingestion status error: {str(e)}")
            return jsonify({
                'error': 'Failed to get ingestion status',
                'message': str(e)
            }), 500

    @app.route('/api/analytics/ingestion/bulk', methods=['POST'])
    def trigger_bulk_ingestion():
        """Trigger manual bulk ingestion of reports."""
        try:
            if not app.report_ingestion:
                return jsonify({
                    'error': 'Report ingestion not available',
                    'message': 'Report ingestion is not initialized.'
                }), 503

            results = app.report_ingestion.manual_bulk_ingestion()
            return jsonify({
                'success': True,
                'results': results
            })

        except Exception as e:
            current_app.logger.error(f"Bulk ingestion error: {str(e)}")
            return jsonify({
                'error': 'Bulk ingestion failed',
                'message': str(e)
            }), 500

    @app.route('/api/analytics/sample-reports', methods=['POST'])
    def create_sample_reports():
        """Create sample reports for testing."""
        try:
            if not app.report_ingestion:
                return jsonify({
                    'error': 'Report ingestion not available',
                    'message': 'Report ingestion is not initialized.'
                }), 503

            count = app.report_ingestion.create_sample_reports()
            return jsonify({
                'success': True,
                'message': f'Created {count} sample reports',
                'reports_created': count
            })

        except Exception as e:
            current_app.logger.error(f"Sample reports error: {str(e)}")
            return jsonify({
                'error': 'Failed to create sample reports',
                'message': str(e)
            }), 500

    @app.route('/api/integration/health')
    def integration_health():
        """Check integration health status."""
        try:
            health_data = {
                'rag_system': 'healthy',
                'analytics_integration': 'healthy' if app.analytics_integration else 'disabled',
                'report_ingestion': 'healthy' if app.report_ingestion else 'disabled',
                'timestamp': json.dumps(None)  # Will be set by datetime
            }

            if app.analytics_integration:
                analytics_health = app.analytics_integration.get_integration_health()
                health_data['analytics_details'] = analytics_health

            return jsonify({
                'success': True,
                'health': health_data
            })

        except Exception as e:
            current_app.logger.error(f"Integration health check error: {str(e)}")
            return jsonify({
                'error': 'Health check failed',
                'message': str(e)
            }), 500
    


if __name__ == '__main__':
    import sys
    
    # Allow port to be specified as command line argument
    port = 5002  # Default to 5002 for AI Agent System
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print("Invalid port number. Using default port 5002.")
    
    # Allow port from environment variable
    port = int(os.getenv('FLASK_RUN_PORT', port))
    
    app = create_app()
    print(f"\n🚀 RAG File Search System starting on port {port}")
    print(f"📱 Open your browser to: http://localhost:{port}")
    print("🔑 Make sure your OpenAI API key is configured in .env file\n")
    
    app.run(debug=True, host='0.0.0.0', port=port)
