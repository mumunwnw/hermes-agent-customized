#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REF="${1:-upstream/main}"
CUSTOM_REF="${2:-custom}"

echo "== Main purity =="
git rev-parse main origin/main "$UPSTREAM_REF"
git diff --stat "$UPSTREAM_REF"..main
git rev-list --left-right --count "$UPSTREAM_REF"...main

echo
echo "== Customization drift: $CUSTOM_REF vs $UPSTREAM_REF =="
git rev-list --left-right --count "$UPSTREAM_REF"..."$CUSTOM_REF"
git diff --stat "$UPSTREAM_REF"..."$CUSTOM_REF"

echo
echo "== Patch-equivalence check =="
git cherry -v "$UPSTREAM_REF" "$CUSTOM_REF"

if [ -d "$HOME/.hermes/hermes-agent/.git" ]; then
  echo
  echo "== Runtime install =="
  git -C "$HOME/.hermes/hermes-agent" branch --show-current
  git -C "$HOME/.hermes/hermes-agent" config --get hermes.updateRemote || true
  git -C "$HOME/.hermes/hermes-agent" config --get hermes.updateBranch || true
fi
