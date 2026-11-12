#!/bin/bash
set -euo pipefail

LABEL="com.f1regulations.update"
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${F1_PYTHON:-$(command -v python3)}"
TARGET="$HOME/Library/LaunchAgents/$LABEL.plist"
DOMAIN="gui/$(id -u)"

case "${1:-status}" in
  install)
    mkdir -p "$PROJECT/logs" "$HOME/Library/LaunchAgents"
    sed -e "s|__PYTHON__|$PYTHON|g" -e "s|__PROJECT__|$PROJECT|g" \
      "$PROJECT/scripts/launchd/$LABEL.plist.template" > "$TARGET"
    plutil -lint "$TARGET"
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    launchctl bootstrap "$DOMAIN" "$TARGET"
    echo "Installed $LABEL: wakes hourly and checks FIA when the last successful check is older than UPDATE_INTERVAL_HOURS"
    ;;
  uninstall)
    launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
    rm -f "$TARGET"
    echo "Removed $LABEL"
    ;;
  run-now)
    launchctl kickstart -k "$DOMAIN/$LABEL"
    ;;
  status)
    if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
      launchctl print "$DOMAIN/$LABEL" | grep -E "state|last exit code|run interval|runs" || true
    else
      echo "$LABEL is not installed"
    fi
    ;;
  *)
    echo "usage: $0 install|uninstall|run-now|status" >&2
    exit 2
    ;;
esac
