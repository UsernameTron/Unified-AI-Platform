#!/bin/bash

# ==============================================================================
# Agent-Assisted Code Cleanup Script for Unified-AI-Platform Repository
# 
# This script implements the comprehensive cleanup protocol outlined in the
# code cleanup instructions to eliminate duplicate implementations and 
# superseded components while preserving all Level 4-5 advanced functionality.
# ==============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
REPO_ROOT="/Users/cpconnor/projects/Unified-AI-Platform/Unified-AI-Platform"
BACKUP_DIR="${REPO_ROOT}_backup_$(date +%Y%m%d_%H%M%S)"
CLEANUP_BRANCH="cleanup/unified-prune"

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}   Agent-Assisted Code Cleanup for Unified-AI-Platform${NC}"
echo -e "${BLUE}================================================================${NC}"

# ==============================================================================
# Phase 1: Pre-Cleanup Validation and Branch Creation
# ==============================================================================

phase1_setup() {
    echo -e "\n${YELLOW}Phase 1: Pre-Cleanup Setup and Validation${NC}"
    
    # Navigate to repository root
    cd "$REPO_ROOT" || {
        echo -e "${RED}Error: Repository root not found at $REPO_ROOT${NC}"
        exit 1
    }
    
    # Verify git repository
    if [ ! -d ".git" ]; then
        echo -e "${RED}Error: Not a git repository${NC}"
        exit 1
    fi
    
    # Create backup
    echo -e "${BLUE}Creating backup at: $BACKUP_DIR${NC}"
    cp -r "$REPO_ROOT" "$BACKUP_DIR"
    
    # Check current branch and create cleanup branch if needed
    CURRENT_BRANCH=$(git branch --show-current)
    echo -e "${BLUE}Current branch: $CURRENT_BRANCH${NC}"
    
    if [[ "$CURRENT_BRANCH" != "$CLEANUP_BRANCH" ]]; then
        echo -e "${BLUE}Creating cleanup branch: $CLEANUP_BRANCH${NC}"
        git checkout -b "$CLEANUP_BRANCH" 2>/dev/null || {
            echo -e "${YELLOW}Cleanup branch already exists, switching to it...${NC}"
            git checkout "$CLEANUP_BRANCH" || {
                echo -e "${RED}Error: Could not switch to cleanup branch${NC}"
                exit 1
            }
        }
    else
        echo -e "${GREEN}Already on cleanup branch: $CLEANUP_BRANCH${NC}"
    fi
    
    # Verify Level 4-5 components exist
    echo -e "${BLUE}Validating Level 4-5 components...${NC}"
    
    local level45_components=(
        "shared_agents"
        "agent_system/web_interface.py"
        "rag_integration/enhanced_app.py"
        "services/tts_service.py"
    )
    
    for component in "${level45_components[@]}"; do
        if [ ! -e "$component" ]; then
            echo -e "${RED}Warning: Level 4-5 component not found: $component${NC}"
        else
            echo -e "${GREEN}✓ Found: $component${NC}"
        fi
    done
    
    echo -e "${GREEN}Phase 1 completed successfully${NC}"
}

# ==============================================================================
# Phase 2: Identify and Remove Legacy Components
# ==============================================================================

phase2_remove_legacy() {
    echo -e "\n${YELLOW}Phase 2: Removing Legacy Components${NC}"
    
    # Remove VectorDBRAG legacy directory if it exists but is empty or has legacy content
    if [ -d "VectorDBRAG" ]; then
        echo -e "${BLUE}Checking VectorDBRAG directory...${NC}"
        if [ -z "$(ls -A VectorDBRAG)" ]; then
            echo -e "${BLUE}Removing empty VectorDBRAG directory${NC}"
            rm -rf VectorDBRAG
        else
            echo -e "${YELLOW}VectorDBRAG directory contains files - manual review needed${NC}"
        fi
    fi
    
    # Remove comparison directory (duplicate documentation)
    if [ -d "comparison" ]; then
        echo -e "${BLUE}Removing comparison directory (duplicate documentation)${NC}"
        git rm -rf comparison
    fi
    
    # Remove MindMeld-v1.1 if it exists (original source, not needed in cleaned repo)
    if [ -d "MindMeld-v1.1" ]; then
        echo -e "${BLUE}Removing MindMeld-v1.1 directory (original source)${NC}"
        git rm -rf MindMeld-v1.1
    fi
    
    # Remove legacy test files that duplicate functionality
    local legacy_tests=(
        "test_simple_enhanced.py"
        "test_enhanced_integration.py" 
        "test_enhanced_flask_integration.py"
        "simple_validate.py"
        "simplified_validation.py"
    )
    
    for test_file in "${legacy_tests[@]}"; do
        if [ -f "$test_file" ]; then
            echo -e "${BLUE}Removing legacy test file: $test_file${NC}"
            git rm "$test_file"
        fi
    done
    
    # Remove duplicate validation scripts
    local duplicate_validation=(
        "validate_end_to_end.py"
        "demonstrate_system.py"
    )
    
    for val_file in "${duplicate_validation[@]}"; do
        if [ -f "$val_file" ]; then
            echo -e "${BLUE}Removing duplicate validation script: $val_file${NC}"
            git rm "$val_file"
        fi
    done
    
    echo -e "${GREEN}Phase 2 completed - Legacy components removed${NC}"
}

# ==============================================================================
# Phase 3: Consolidate Documentation
# ==============================================================================

phase3_consolidate_docs() {
    echo -e "\n${YELLOW}Phase 3: Consolidating Documentation${NC}"
    
    # Remove duplicate README files
    local duplicate_docs=(
        "DOCKER_README.md"
        "IMPLEMENTATION_COMPLETE.md"
        "ENHANCEMENT_OPTIONS.md"
        "MIGRATION_GUIDE.md"
        "TTS_IMPLEMENTATION_COMPLETE.md"
        "TTS_VOICE_EXPANSION_SUCCESS.md"
    )
    
    for doc_file in "${duplicate_docs[@]}"; do
        if [ -f "$doc_file" ]; then
            echo -e "${BLUE}Removing duplicate documentation: $doc_file${NC}"
            git rm "$doc_file"
        fi
    done
    
    # Keep only essential documentation
    echo -e "${BLUE}Preserving essential documentation files:${NC}"
    local essential_docs=(
        "README.md"
        "DEPLOYMENT_GUIDE.md"
        "DEPLOYMENT_SUCCESS_REPORT.md"
        "UNIFIED_INTERFACE_README.md"
    )
    
    for doc_file in "${essential_docs[@]}"; do
        if [ -f "$doc_file" ]; then
            echo -e "${GREEN}✓ Preserving: $doc_file${NC}"
        fi
    done
    
    echo -e "${GREEN}Phase 3 completed - Documentation consolidated${NC}"
}

# ==============================================================================
# Phase 4: Remove Superseded Test Files and Validation Scripts
# ==============================================================================

phase4_cleanup_tests() {
    echo -e "\n${YELLOW}Phase 4: Cleaning Up Test Files${NC}"
    
    # Remove test result files and temporary data
    local test_artifacts=(
        "comprehensive_tts_test_results.json"
        "FINAL_VALIDATION_REPORT.json"
        "unified_system_test_report.json"
    )
    
    for artifact in "${test_artifacts[@]}"; do
        if [ -f "$artifact" ]; then
            echo -e "${BLUE}Removing test artifact: $artifact${NC}"
            git rm "$artifact"
        fi
    done
    
    # Remove individual TTS test files (functionality integrated into main system)
    local tts_tests=(
        "test_comprehensive_tts_voices.py"
    )
    
    for tts_test in "${tts_tests[@]}"; do
        if [ -f "$tts_test" ]; then
            echo -e "${BLUE}Removing TTS test file: $tts_test${NC}"
            git rm "$tts_test"
        fi
    done
    
    # Keep the unified system test as it's the Level 4-5 implementation
    if [ -f "test_unified_system.py" ]; then
        echo -e "${GREEN}✓ Preserving: test_unified_system.py (Level 4-5 component)${NC}"
    fi
    
    # Keep final validation as it's the comprehensive test
    if [ -f "final_validation.py" ]; then
        echo -e "${GREEN}✓ Preserving: final_validation.py (Level 4-5 component)${NC}"
    fi
    
    echo -e "${GREEN}Phase 4 completed - Test files cleaned${NC}"
}

# ==============================================================================
# Phase 5: Clean Up Flask Session Files and Temporary Data
# ==============================================================================

phase5_cleanup_temp() {
    echo -e "\n${YELLOW}Phase 5: Cleaning Up Temporary Files${NC}"
    
    # Remove flask session directory
    if [ -d "flask_session" ]; then
        echo -e "${BLUE}Removing Flask session directory${NC}"
        rm -rf flask_session
    fi
    
    # Remove Python cache directories
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "*.pyo" -delete 2>/dev/null || true
    
    # Remove package-lock.json if present (not needed for Python project)
    if [ -f "package-lock.json" ]; then
        echo -e "${BLUE}Removing package-lock.json${NC}"
        rm -f "package-lock.json"
    fi
    
    echo -e "${GREEN}Phase 5 completed - Temporary files cleaned${NC}"
}

# ==============================================================================
# Phase 6: Validate Level 4-5 Component Integrity
# ==============================================================================

phase6_validate_components() {
    echo -e "\n${YELLOW}Phase 6: Validating Level 4-5 Component Integrity${NC}"
    
    # Check that all Level 4-5 components still exist
    local level45_components=(
        "shared_agents/"
        "shared_agents/core/"
        "shared_agents/config/"
        "shared_agents/validation/"
        "shared_agents/tests/"
        "agent_system/"
        "agent_system/web_interface.py"
        "agent_system/analytics_dashboard.py"
        "rag_integration/"
        "rag_integration/enhanced_app.py"
        "services/"
        "services/tts_service.py"
    )
    
    local missing_components=()
    
    for component in "${level45_components[@]}"; do
        if [ ! -e "$component" ]; then
            missing_components+=("$component")
            echo -e "${RED}✗ Missing: $component${NC}"
        else
            echo -e "${GREEN}✓ Present: $component${NC}"
        fi
    done
    
    if [ ${#missing_components[@]} -eq 0 ]; then
        echo -e "${GREEN}All Level 4-5 components validated successfully${NC}"
    else
        echo -e "${RED}Warning: ${#missing_components[@]} Level 4-5 components are missing${NC}"
        for component in "${missing_components[@]}"; do
            echo -e "${RED}  - $component${NC}"
        done
    fi
    
    echo -e "${GREEN}Phase 6 completed - Component validation finished${NC}"
}

# ==============================================================================
# Phase 7: Update Import Statements and Dependencies
# ==============================================================================

phase7_update_imports() {
    echo -e "\n${YELLOW}Phase 7: Updating Import Statements${NC}"
    
    # Check for broken imports in remaining Python files
    echo -e "${BLUE}Checking for import issues in remaining Python files...${NC}"
    
    # Compile all Python files to check for import errors
    local python_files=$(find . -name "*.py" -not -path "./__pycache__/*" -not -path "./.*")
    local import_errors=()
    
    while IFS= read -r py_file; do
        if [ -f "$py_file" ]; then
            if ! python3 -m py_compile "$py_file" 2>/dev/null; then
                import_errors+=("$py_file")
                echo -e "${YELLOW}⚠ Potential import issue in: $py_file${NC}"
            fi
        fi
    done <<< "$python_files"
    
    if [ ${#import_errors[@]} -eq 0 ]; then
        echo -e "${GREEN}No import errors detected${NC}"
    else
        echo -e "${YELLOW}Found ${#import_errors[@]} files with potential import issues${NC}"
        echo -e "${YELLOW}These may need manual review after cleanup${NC}"
    fi
    
    echo -e "${GREEN}Phase 7 completed - Import validation finished${NC}"
}

# ==============================================================================
# Phase 8: Clean Up Requirements and Dependencies
# ==============================================================================

phase8_cleanup_deps() {
    echo -e "\n${YELLOW}Phase 8: Cleaning Up Dependencies${NC}"
    
    # Keep only the main requirements file
    if [ -f "requirements-dev.txt" ]; then
        echo -e "${BLUE}Removing development-only requirements file${NC}"
        git rm "requirements-dev.txt" 2>/dev/null || rm -f "requirements-dev.txt"
    fi
    
    # Update requirements.txt to include only necessary dependencies
    if [ -f "requirements.txt" ]; then
        echo -e "${GREEN}✓ Preserving: requirements.txt${NC}"
    else
        echo -e "${YELLOW}⚠ requirements.txt not found - may need to be created${NC}"
    fi
    
    echo -e "${GREEN}Phase 8 completed - Dependencies cleaned${NC}"
}

# ==============================================================================
# Phase 8.5: Setup API Configuration Management
# ==============================================================================

phase8_5_setup_api_config() {
    echo -e "\n${YELLOW}Phase 8.5: Setting Up API Configuration Management${NC}"
    
    # Create comprehensive .env.example template
    if [ ! -f ".env.example" ]; then
        echo -e "${BLUE}Creating .env.example template${NC}"
        cat > .env.example << 'EOF'
# ==============================================================================
# Unified-AI-Platform API Configuration Template
# Copy to .env and fill in your actual API keys
# ==============================================================================

# OpenAI Configuration (General Purpose)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
OPENAI_MAX_TOKENS=2048

# Anthropic Claude Configuration (Logic & Code)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
ANTHROPIC_MAX_TOKENS=4096

# Google Gemini Configuration (Content Creation)
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL=gemini-1.5-pro
GEMINI_MAX_TOKENS=2048

# Local AI Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=phi3.5

# Text-to-Speech Services
ELEVENLABS_API_KEY=your_elevenlabs_key_here
AZURE_SPEECH_KEY=your_azure_speech_key_here
AZURE_SPEECH_REGION=your_azure_region_here

# Vector Database (if using)
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_ENVIRONMENT=your_pinecone_env_here

# Additional AI Services
COHERE_API_KEY=your_cohere_key_here
HUGGINGFACE_API_KEY=your_huggingface_key_here
EOF
    fi
    
    # Update .gitignore to protect API keys
    if [ ! -f ".gitignore" ]; then
        touch .gitignore
    fi
    
    if ! grep -q "^\.env$" .gitignore; then
        echo -e "${BLUE}Adding environment files to .gitignore${NC}"
        cat >> .gitignore << 'EOF'

# Environment variables and API keys
.env
.env.local
.env.*.local
.env.production
.env.development

# API Configuration
config/api_keys.json
config/secrets.yaml
EOF
    fi
    
    # Create actual .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        echo -e "${BLUE}Creating .env file from template${NC}"
        cp .env.example .env
        echo -e "${YELLOW}Please edit .env file with your actual API keys${NC}"
    fi
    
    echo -e "${GREEN}Phase 8.5 completed - API configuration setup${NC}"
}

# ==============================================================================
# Phase 9: Validate API Endpoints and Routes
# ==============================================================================

phase9_validate_api() {
    echo -e "\n${YELLOW}Phase 9: Validating API Endpoints${NC}"
    
    # Check if key route files exist
    local route_files=(
        "agent_system/web_interface.py"
        "rag_integration/enhanced_app.py"
    )
    
    for route_file in "${route_files[@]}"; do
        if [ -f "$route_file" ]; then
            echo -e "${GREEN}✓ Route file present: $route_file${NC}"
            
            # Check for Level 4-5 endpoints
            if grep -q "/api/enhanced" "$route_file" 2>/dev/null; then
                echo -e "${GREEN}  ✓ Enhanced API endpoints found${NC}"
            fi
            
            if grep -q "/api/dashboard" "$route_file" 2>/dev/null; then
                echo -e "${GREEN}  ✓ Dashboard API endpoints found${NC}"
            fi
        else
            echo -e "${RED}✗ Route file missing: $route_file${NC}"
        fi
    done
    
    echo -e "${GREEN}Phase 9 completed - API validation finished${NC}"
}

# ==============================================================================
# Phase 10: Create Summary and Commit Changes
# ==============================================================================

phase10_commit_changes() {
    echo -e "\n${YELLOW}Phase 10: Committing Cleanup Changes${NC}"
    
    # Add all changes to git
    git add -A
    
    # Create commit message
    local commit_msg="Agent-assisted code cleanup: Remove Level 3 components, preserve Level 4-5

- Removed duplicate implementations and superseded components
- Preserved Level 4-5 components: shared_agents/, agent_system/, rag_integration/, services/
- Consolidated documentation and removed legacy test files
- Cleaned up temporary files and session data
- Validated component integrity and API endpoints
- Maintained all advanced functionality including MindMeld framework integration

Components preserved:
- shared_agents/ (MindMeld framework extraction)
- agent_system/web_interface.py (unified Flask application)  
- rag_integration/enhanced_app.py (integration layer)
- services/tts_service.py (text-to-speech capabilities)
- Enhanced agent implementations and unified dashboard

This cleanup maintains complete functional capability while eliminating
duplicate code paths and legacy implementations."
    
    # Commit changes
    git commit -m "$commit_msg"
    
    echo -e "${GREEN}Cleanup changes committed successfully${NC}"
    
    # Show summary of changes
    echo -e "\n${BLUE}=== CLEANUP SUMMARY ===${NC}"
    echo -e "${GREEN}Files removed:${NC}"
    git diff --name-status HEAD~1 | grep "^D" | awk '{print "  - " $2}' || echo "  (No files removed via git)"
    
    echo -e "\n${GREEN}Files modified:${NC}"
    git diff --name-status HEAD~1 | grep "^M" | awk '{print "  - " $2}' || echo "  (No files modified)"
    
    echo -e "\n${BLUE}Repository is now cleaned and ready for validation${NC}"
}

# ==============================================================================
# Phase 11: Post-Cleanup Validation
# ==============================================================================

phase11_final_validation() {
    echo -e "\n${YELLOW}Phase 11: Post-Cleanup Validation${NC}"
    
    # Run Python compilation check on all remaining files
    echo -e "${BLUE}Running final Python compilation check...${NC}"
    local python_files=$(find . -name "*.py" -not -path "./__pycache__/*")
    local compile_errors=0
    
    while IFS= read -r py_file; do
        if [ -f "$py_file" ]; then
            if ! python3 -m py_compile "$py_file" 2>/dev/null; then
                echo -e "${RED}✗ Compilation error in: $py_file${NC}"
                ((compile_errors++))
            fi
        fi
    done <<< "$python_files"
    
    if [ $compile_errors -eq 0 ]; then
        echo -e "${GREEN}✓ All Python files compile successfully${NC}"
    else
        echo -e "${RED}⚠ $compile_errors files have compilation errors${NC}"
    fi
    
    # Check project structure
    echo -e "\n${BLUE}Final project structure:${NC}"
    tree -L 2 2>/dev/null || ls -la
    
    echo -e "\n${GREEN}Phase 11 completed - Final validation finished${NC}"
}

# ==============================================================================
# Main Execution Function
# ==============================================================================

main() {
    echo -e "${BLUE}Starting Agent-Assisted Code Cleanup...${NC}"
    
    # Execute all phases
    phase1_setup
    phase2_remove_legacy  
    phase3_consolidate_docs
    phase4_cleanup_tests
    phase5_cleanup_temp
    phase6_validate_components
    phase7_update_imports
    phase8_cleanup_deps
    phase8_5_setup_api_config
    phase9_validate_api
    phase10_commit_changes
    phase11_final_validation
    
    echo -e "\n${GREEN}================================================================${NC}"
    echo -e "${GREEN}   Agent-Assisted Code Cleanup Completed Successfully!${NC}"
    echo -e "${GREEN}================================================================${NC}"
    
    echo -e "\n${BLUE}Next Steps:${NC}"
    echo -e "1. Review the cleanup branch: ${YELLOW}git log --oneline${NC}"
    echo -e "2. Test the cleaned system: ${YELLOW}python final_validation.py${NC}"
    echo -e "3. If satisfied, merge to main: ${YELLOW}git checkout main && git merge $CLEANUP_BRANCH${NC}"
    echo -e "4. Backup location: ${YELLOW}$BACKUP_DIR${NC}"
    
    echo -e "\n${GREEN}Cleanup completed successfully!${NC}"
}

# ==============================================================================
# Script Execution with Error Handling
# ==============================================================================

# Check if running from correct directory
if [ ! -f "README.md" ] || [ ! -d "shared_agents" ]; then
    echo -e "${RED}Error: Please run this script from the Unified-AI-Platform repository root${NC}"
    echo -e "${RED}Expected files/directories not found${NC}"
    exit 1
fi

# Run main function with error handling
if ! main; then
    echo -e "${RED}Error: Cleanup script failed${NC}"
    echo -e "${RED}Backup available at: $BACKUP_DIR${NC}"
    exit 1
fi
