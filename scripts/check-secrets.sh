#!/bin/bash
# Security check script - Run before pushing to GitHub
# Usage: ./scripts/check-secrets.sh

set -e

echo "🔒 Running security checks..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0

# Check 1: Verify .env.local is not tracked
echo "1️⃣  Checking .env.local is gitignored..."
if git ls-files --error-unmatch .env.local 2>/dev/null; then
    echo -e "${RED}❌ CRITICAL: .env.local is tracked by git!${NC}"
    echo "   Run: git rm --cached .env.local"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✅ .env.local is properly ignored${NC}"
fi
echo ""

# Check 2: Verify memory files are not tracked
echo "2️⃣  Checking memory files are gitignored..."
if git ls-files --error-unmatch memory/memory.json 2>/dev/null; then
    echo -e "${RED}❌ CRITICAL: memory/memory.json is tracked by git!${NC}"
    echo "   Run: git rm --cached memory/memory.json"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✅ memory/memory.json is properly ignored${NC}"
fi
echo ""

# Check 3: Scan staged files for API key patterns
echo "3️⃣  Scanning staged files for API keys..."
if git diff --cached | grep -iE "(AIza[A-Za-z0-9_-]{35}|sk-[A-Za-z0-9]{20,}|pk-lf-[A-Za-z0-9]{20,}|sk-lf-[A-Za-z0-9]{20,})" >/dev/null; then
    echo -e "${RED}❌ CRITICAL: Potential API keys found in staged changes!${NC}"
    echo "   Review your staged changes carefully"
    git diff --cached | grep -iE "(AIza|sk-|pk-lf-|sk-lf-)" --color=always
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✅ No API keys detected in staged changes${NC}"
fi
echo ""

# Check 4: Scan for hardcoded credentials
echo "4️⃣  Scanning for hardcoded credentials..."
if git diff --cached | grep -iE "(password|secret|token|credential).*=.*['\"][^'\"]{20,}" >/dev/null; then
    echo -e "${YELLOW}⚠️  Warning: Potential hardcoded credentials found${NC}"
    echo "   Review carefully:"
    git diff --cached | grep -iE "(password|secret|token|credential).*=.*['\"]" --color=always
    echo ""
    echo "   If these are placeholder values, you can ignore this warning"
else
    echo -e "${GREEN}✅ No hardcoded credentials detected${NC}"
fi
echo ""

# Check 5: Verify only .env.example is tracked
echo "5️⃣  Checking environment files..."
ENV_FILES=$(git ls-files | grep -E "\.env" | grep -v ".env.example" || true)
if [ -n "$ENV_FILES" ]; then
    echo -e "${RED}❌ CRITICAL: Environment files are tracked:${NC}"
    echo "$ENV_FILES"
    echo "   Run: git rm --cached <file>"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✅ Only .env.example is tracked${NC}"
fi
echo ""

# Check 6: Verify no credential files are tracked
echo "6️⃣  Checking for credential files..."
CRED_FILES=$(git ls-files | grep -iE "(credentials|service-account|keyfile)\.json" || true)
if [ -n "$CRED_FILES" ]; then
    echo -e "${RED}❌ CRITICAL: Credential files are tracked:${NC}"
    echo "$CRED_FILES"
    echo "   Run: git rm --cached <file>"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✅ No credential files tracked${NC}"
fi
echo ""

# Summary
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ All security checks passed!${NC}"
    echo ""
    echo "   Safe to push to GitHub"
    echo ""
    exit 0
else
    echo -e "${RED}❌ Security checks failed: $ERRORS error(s) found${NC}"
    echo ""
    echo "   DO NOT push until all issues are resolved!"
    echo ""
    exit 1
fi
