#!/bin/bash
# Helper script to invoke the Pattern Reviewer agent

set -e

AGENT_NAME="pattern-reviewer"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

show_help() {
    cat << EOF
${BLUE}Pattern Reviewer Agent - Helper Script${NC}

Review, improve, and create test cases for scicode-lint patterns.

${GREEN}Usage:${NC}
  $0 <command> [arguments]

${GREEN}Commands:${NC}
  review <pattern-id>          Review a specific pattern
  review-category <category>   Review all patterns in a category (batch)
  batch-review <pattern-list>  Review multiple specific patterns in parallel
  find-issues                  Find patterns needing improvement (batch)
  create-tests <pattern-id>    Create new test cases for a pattern
  help                         Show this help message

${GREEN}Examples:${NC}
  # Review a specific pattern
  $0 review ml-001-scaler-leakage

  # Review all AI training patterns (batch operation)
  $0 review-category ai-training

  # Review multiple specific patterns in parallel
  $0 batch-review "ml-001 ml-002 ml-003"

  # Find patterns with issues (batch operation)
  $0 find-issues

  # Create test cases
  $0 create-tests ml-001-scaler-leakage

${GREEN}Available Categories:${NC}
  - ai-training              (15 patterns)
  - ai-inference            (3 patterns)
  - ai-data                 (1 pattern)
  - scientific-numerical    (10 patterns)
  - scientific-reproducibility (4 patterns)
  - scientific-performance  (11 patterns)

${GREEN}Direct Claude Code Usage:${NC}
  You can also invoke the agent directly:

  claude --agent pattern-reviewer "Review ml-001-scaler-leakage"

${GREEN}Agent files:${NC}
  - Prompt:     pattern_verification/pattern-reviewer/system_prompt.md
  - Metadata:   pattern_verification/pattern-reviewer/agent.json

EOF
}

# Check if claude is available
if ! command -v claude &> /dev/null; then
    echo -e "${YELLOW}Error: 'claude' command not found.${NC}"
    echo ""
    echo "This script requires Claude Code CLI to function."
    echo "The pattern-reviewer agent is specifically designed for Claude Code"
    echo "and will not work with other AI tools or interfaces."
    echo ""
    echo "Install Claude Code CLI from:"
    echo "  https://github.com/anthropics/claude-code"
    echo ""
    exit 1
fi

# Parse command
case "${1:-}" in
    review)
        if [ -z "$2" ]; then
            echo -e "${YELLOW}Error: Pattern ID required${NC}"
            echo "Usage: $0 review <pattern-id>"
            echo "Example: $0 review ml-001-scaler-leakage"
            exit 1
        fi
        echo -e "${BLUE}Reviewing pattern: $2${NC}"
        claude --agent "$AGENT_NAME" "Review the $2 pattern comprehensively and suggest improvements"
        ;;

    review-category)
        if [ -z "$2" ]; then
            echo -e "${YELLOW}Error: Category required${NC}"
            echo "Usage: $0 review-category <category>"
            echo "Example: $0 review-category ai-training"
            exit 1
        fi
        echo -e "${BLUE}Reviewing category: $2${NC}"
        claude --agent "$AGENT_NAME" "Review all patterns in the $2 category and identify top priorities for improvement"
        ;;

    find-issues)
        echo -e "${BLUE}Finding patterns with issues...${NC}"
        claude --agent "$AGENT_NAME" "Which patterns have incomplete test coverage, unclear detection questions, or missing metadata? Prioritize by severity and impact."
        ;;

    batch-review)
        if [ -z "$2" ]; then
            echo -e "${YELLOW}Error: Pattern list required${NC}"
            echo "Usage: $0 batch-review \"pattern1 pattern2 pattern3\""
            echo "Example: $0 batch-review \"ml-001 ml-002 ml-003\""
            exit 1
        fi
        echo -e "${BLUE}Batch reviewing patterns: $2${NC}"
        # Convert space-separated list to comma-separated for agent
        PATTERN_LIST=$(echo "$2" | tr ' ' ',')
        claude --agent "$AGENT_NAME" "Review patterns $PATTERN_LIST in parallel and provide a batch summary"
        ;;

    create-tests)
        if [ -z "$2" ]; then
            echo -e "${YELLOW}Error: Pattern ID required${NC}"
            echo "Usage: $0 create-tests <pattern-id>"
            echo "Example: $0 create-tests ml-001-scaler-leakage"
            exit 1
        fi
        echo -e "${BLUE}Creating test cases for: $2${NC}"
        claude --agent "$AGENT_NAME" "Create additional test cases (positive, negative, and context-dependent) for $2 to improve coverage"
        ;;

    help|--help|-h)
        show_help
        ;;

    "")
        echo -e "${YELLOW}Error: No command specified${NC}"
        echo ""
        show_help
        exit 1
        ;;

    *)
        echo -e "${YELLOW}Error: Unknown command '$1'${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
