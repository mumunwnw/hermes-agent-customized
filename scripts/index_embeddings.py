#!/usr/bin/env python3
"""Index session embeddings for hybrid search.

Usage:
    # Index all sessions
    python scripts/index_embeddings.py
    
    # Index specific number of sessions
    python scripts/index_embeddings.py --limit 100
    
    # Show indexing status
    python scripts/index_embeddings.py --status
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from hermes_state import SessionDB
from tools.hybrid_search import HybridSessionSearch, check_hybrid_search_requirements
from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_index(args):
    """Index session embeddings."""
    if not check_hybrid_search_requirements():
        print("❌ Hybrid search requirements not met:")
        print("   1. Install sqlite-vec: pip install sqlite-vec")
        print("   2. Set SILICONFLOW_API_KEY or QMD_EMBED_API_KEY")
        sys.exit(1)
    
    db_path = get_hermes_home() / "state.db"
    if not db_path.exists():
        print(f"❌ Session database not found: {db_path}")
        sys.exit(1)
    
    db = SessionDB(db_path)
    hybrid = HybridSessionSearch(db)
    
    print("🔍 Indexing all unindexed messages...")
    
    result = hybrid._index_unindexed_batch()
    
    if result.get("error"):
        print(f"❌ Indexing failed: {result['error']}")
        sys.exit(1)
    
    indexed = result.get("indexed", 0)
    failed = result.get("failed", 0)
    total = result.get("total_processed", 0)
    
    print(f"\n📊 Indexing Summary:")
    print(f"   ✅ Indexed: {indexed} messages")
    print(f"   ❌ Failed: {failed} messages")
    print(f"   📋 Total processed: {total} messages")


def cmd_status(args):
    """Show indexing status."""
    db_path = get_hermes_home() / "state.db"
    if not db_path.exists():
        print("❌ Session database not found")
        return
    
    import sqlite3
    conn = sqlite3.connect(db_path)
    
    # Check if vec table exists
    vec_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='message_vec'"
    ).fetchone()
    
    if vec_exists:
        count = conn.execute("SELECT COUNT(*) FROM message_vec").fetchone()[0]
        print(f"✅ Vector index exists: {count} embeddings")
    else:
        print("❌ Vector index does not exist")
        print("   Run: python scripts/index_embeddings.py")
    
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Index session embeddings for hybrid search")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--limit", "-l", type=int, default=10000, help="Max sessions to index")
    parser.add_argument("--status", action="store_true", help="Show indexing status")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    if args.status:
        cmd_status(args)
    else:
        cmd_index(args)


if __name__ == "__main__":
    main()
