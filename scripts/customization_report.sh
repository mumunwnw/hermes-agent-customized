#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_REF="${1:-upstream/main}"
CUSTOM_REF="${2:-stable}"

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
