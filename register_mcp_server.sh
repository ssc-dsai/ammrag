#!/usr/bin/env bash
# Register the AMMRAG HTTP MCP server with the canchat-v2 application.
#
# Usage:
#   ./register_mcp_server.sh
#   ./register_mcp_server.sh --mcp-url http://172.17.0.3:8001/mcp/
#   ./register_mcp_server.sh --token <admin-token>
#   ./register_mcp_server.sh --dry-run
#
# Options:
#   -u, --mcp-url    URL of the AMMRAG MCP server (default: from mcp_config.json)
#   -t, --token      Admin bearer token (or set CANCHAT_ADMIN_TOKEN)
#   -s, --server-url canchat-v2 base URL (default: http://localhost:8080)
#   -n, --dry-run    Print request without sending

set -euo pipefail

SERVER_URL="${CANCHAT_SERVER_URL:-http://localhost:8080}"
TOKEN="${CANCHAT_ADMIN_TOKEN:-}"
DRY_RUN=false

MCP_URL="http://172.17.0.3:8001/mcp/"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -u|--mcp-url)    MCP_URL="$2";    shift 2 ;;
        -t|--token)      TOKEN="$2";       shift 2 ;;
        -s|--server-url) SERVER_URL="$2";  shift 2 ;;
        -n|--dry-run)    DRY_RUN=true;     shift   ;;
        -h|--help)       sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$TOKEN" && "$DRY_RUN" == false ]]; then
    echo "Error: admin token required. Use --token or set CANCHAT_ADMIN_TOKEN." >&2
    exit 1
fi

echo ""
echo "=================================================="
echo "Registering AMMRAG HTTP MCP server"
echo "=================================================="
echo "  canchat-v2 : $SERVER_URL"
echo "  MCP URL    : $MCP_URL"

if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo "[dry-run] Would POST to $SERVER_URL/api/v1/mcp/config/update"
    echo '  { "ENABLE_MCP_API": true, "MCP_BASE_URLS": ["'"$MCP_URL"'"], "MCP_API_CONFIGS": {} }'
    echo ""
    echo "Done."
    exit 0
fi

CFG_RESP="$(curl -s \
    "$SERVER_URL/api/v1/mcp/config" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Accept: application/json" || true)"

CURRENT_URLS="$(echo "$CFG_RESP" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    urls = d.get('MCP_BASE_URLS') or []
    target = '$MCP_URL'
    if target not in urls:
        urls.append(target)
    print(json.dumps(urls))
except Exception:
    print(json.dumps(['$MCP_URL']))
" 2>/dev/null || echo '["'"$MCP_URL"'"]')"

CURRENT_API_CONFIGS="$(echo "$CFG_RESP" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(json.dumps(d.get('MCP_API_CONFIGS') or {}))
except Exception:
    print('{}')
" 2>/dev/null || echo '{}')"

PAYLOAD="$(python3 -c "
import json
print(json.dumps({
    'ENABLE_MCP_API': True,
    'MCP_BASE_URLS': $CURRENT_URLS,
    'MCP_API_CONFIGS': $CURRENT_API_CONFIGS
}, indent=2))")"

REGISTER_RESPONSE="$(curl -s -w "\n%{http_code}" \
    -X POST "$SERVER_URL/api/v1/mcp/config/update" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -d "$PAYLOAD")"

HTTP_CODE="$(echo "$REGISTER_RESPONSE" | tail -n1)"
BODY="$(echo "$REGISTER_RESPONSE" | head -n -1)"

if [[ "$HTTP_CODE" != "200" ]]; then
    echo "Error $HTTP_CODE: $BODY" >&2
    exit 1
fi

URLS="$(echo "$BODY" | python3 -c "import json,sys; print(json.load(sys.stdin).get('MCP_BASE_URLS', []))" 2>/dev/null || true)"
echo ""
echo "  Registered. MCP_BASE_URLS is now: $URLS"

echo ""
echo "=================================================="
echo "Verifying MCP connection"
echo "=================================================="

VERIFY_RESPONSE="$(curl -s -w "\n%{http_code}" \
    -X POST "$SERVER_URL/api/v1/mcp/verify" \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -H "Accept: application/json" \
    -d "{\"url\": \"$MCP_URL\"}")"

HTTP_CODE="$(echo "$VERIFY_RESPONSE" | tail -n1)"
BODY="$(echo "$VERIFY_RESPONSE" | head -n -1)"

if [[ "$HTTP_CODE" != "200" ]]; then
    echo "  Warning: verification returned $HTTP_CODE: $BODY" >&2
else
    STATUS="$(echo "$BODY" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null || true)"
    TOOLS_COUNT="$(echo "$BODY" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('tools_count','?'))" 2>/dev/null || true)"
    echo "  Status      : $STATUS"
    echo "  Tools found : $TOOLS_COUNT"
    if [[ "$STATUS" != "connected" ]]; then
        echo "  Warning: not connected. Is AMMRAG running at $MCP_URL?" >&2
    fi
fi

echo ""
echo "Done."
