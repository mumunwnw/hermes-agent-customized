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
        from tools.hybrid_search import get_hybrid_search, check_hybrid_search_requirements, HybridSessionSearch
        if not check_hybrid_search_requirements():
            return
        from tools.hybrid_search import _hybrid_instance
        if _hybrid_instance is not None:
            _hybrid_instance.index_session()
            return
        from hermes_state import SessionDB
        from hermes_constants import get_hermes_home
        db_path = get_hermes_home() / "state.db"
        db = SessionDB(db_path)
        try:
            search = HybridSessionSearch(db, config=ss_config)
            search.index_session()
            search.stop_auto_index_daemon()
        finally:
            try:
                db._conn.close()
            except Exception:
                pass
    except Exception as exc:
        logger.warning("混合搜索索引器: 索引失败: %s", exc)


def register(ctx) -> None:
    ctx.register_hook("on_session_finalize", _on_session_finalize)
