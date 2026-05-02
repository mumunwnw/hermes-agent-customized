import logging
from typing import Any

logger = logging.getLogger(__name__)


def _on_session_finalize(session_id: str = None, platform: str = "", **_: Any) -> None:
    try:
        from hermes_cli.config import load_config
        config = load_config()
        ss_config = config.get("auxiliary", {}).get("session_search", {})
        engine = ss_config.get("engine", "bm25")
        if engine not in ("hybrid", "auto"):
            return
        auto_index = ss_config.get("hybrid", {}).get("auto_index", True)
        if not auto_index:
            return
    except Exception:
        return
    try:
        from tools.hybrid_search import HybridSessionSearch
        from hermes_state import SessionDB
        from hermes_constants import get_hermes_home

        db_path = get_hermes_home() / "state.db"
        db = SessionDB(db_path)
        search = HybridSessionSearch(db, config=ss_config)
        search.index_session()
    except Exception as exc:
        logger.warning("hybrid-search-indexer: failed: %s", exc)


def register(ctx) -> None:
    ctx.register_hook("on_session_finalize", _on_session_finalize)
