# Fork Customization Registry

This fork uses `main` as a clean mirror of `upstream/main`.

Do not commit local or personalized changes to `main`. All fork-only behavior
must live on dedicated customization branches and be recorded in this directory.

## Branch Policy

- `main`: exact upstream mirror. It may only be advanced by fast-forwarding to
  `upstream/main`, then pushing that same commit to `origin/main`.
- `stable`: integration branch for local customizations that should be usable
  day to day.
- `feature/*`: topic branches for one customization area. Keep each topic as
  small as possible.
- `codex/*`: maintenance, documentation, or automation branches created by
  Codex. These must not be merged into `main`.

Before touching code, start from the right base:

```bash
git switch main
git fetch upstream
git merge --ff-only upstream/main
git push origin main

git switch -c feature/<area> main
```

When integrating a local feature:

```bash
git switch stable
git rebase origin/main
git merge --no-ff feature/<area>
git push origin stable
```

If the feature is a long-lived patch over upstream, prefer rebasing the feature
branch onto current `main` and keeping commits logically grouped:

```bash
git switch feature/<area>
git rebase main
```

## Current Customization Inventory

Snapshot date: 2026-05-13.

Upstream mirror commit:

- `main`, `origin/main`, `upstream/main`:
  `942adf617910f50a39f41bd200d8083bf4cb2bed`

Customization integration branch:

- `stable` / `origin/stable`:
  `5323c57835298132e0f7c09eca198ac4a4fcc056`
- Relationship to `upstream/main`: `348` upstream commits ahead of the old
  fork point, `63` local commits on `stable`.
- Merge base with current upstream:
  `369cee018d46560e7076e209f311756aa5ec1f70`

Tracked local feature branches:

| Branch | Head | Status | Purpose |
| --- | --- | --- | --- |
| `feature/feishu-messages-enhanced` | `5323c5783` | included in `stable` | Feishu paragraph-aware delivery, pacing, list and short-paragraph merging. |
| `feature/hybrid-session-search` | `4eb94a35c` | included in `stable` | Hybrid session search with BM25, vector search, RRF, indexing tools, diagnostics, and logging. |
| `feature/stable-branch-update` | `0e1c36554` | included in `stable` | Changes `hermes update` behavior for the fork's stable branch flow. |

Current local diff surface on `stable` versus current upstream:

- `.plans/hybrid-session-search.md`
- `gateway/config.py`
- `gateway/platforms/base.py`
- `gateway/platforms/feishu.py`
- `hermes_cli/config.py`
- `hermes_cli/logs.py`
- `hermes_cli/main.py`
- `hermes_logging.py`
- `model_tools.py`
- `plugins/hybrid-search-indexer/__init__.py`
- `plugins/hybrid-search-indexer/plugin.yaml`
- `run_agent.py`
- `scripts/index_embeddings.py`
- `tests/gateway/test_feishu.py`
- `tests/tools/test_hybrid_search.py`
- `tools/hybrid_search.py`
- `tools/session_search_tool.py`
- `toolsets.py`
- `website/docs/getting-started/updating.md`

Notes:

- `git cherry -v upstream/main stable` currently marks all customization
  commits with `+`, meaning upstream does not contain patch-equivalent versions
  of these commits yet.
- `stable` currently contains duplicated generations of the hybrid-search work.
  Before long-term maintenance, consider squashing or rebuilding the patch stack
  into smaller topic branches.

## Upstream Coverage Check

Run this after every upstream sync:

```bash
git fetch upstream origin
git switch main
git merge --ff-only upstream/main
git push origin main

git cherry -v upstream/main stable
git diff --stat upstream/main...stable
git rev-list --left-right --count upstream/main...stable
```

Interpretation:

- `git cherry` lines starting with `-` are patch-equivalent to upstream and can
  usually be dropped from the local patch stack after review.
- `git cherry` lines starting with `+` remain fork-only.
- A growing `git diff --stat` means local customizations are spreading across
  the upstream tree and should be reconsidered or moved behind plugin/config
  boundaries where possible.

For a topic branch:

```bash
git range-diff upstream/main...feature/<area>
git diff --stat upstream/main...feature/<area>
```

## Maintenance Rules

1. Keep `main` pure.
2. Prefer plugins, optional skills, config keys, and isolated adapters over
   editing central files such as `run_agent.py`, `cli.py`, `gateway/run.py`, or
   `model_tools.py`.
3. If a central file must change, keep the change behind a small function,
   config gate, or extension hook.
4. Every customization area should have:
   - one topic branch;
   - a short entry in this registry;
   - focused tests for the behavior;
   - a clear "drop condition" describing when upstream replaces it.
5. When upstream adds an equivalent feature, first remove the local feature
   branch from `stable`, then verify behavior with tests, then update this file.

## Suggested Refactor Targets

- Hybrid session search should ideally live as a plugin or behind a narrow
  search-provider interface, leaving `tools/session_search_tool.py`,
  `toolsets.py`, and `run_agent.py` with minimal integration code.
- Feishu message pacing should stay localized to `gateway/platforms/feishu.py`
  and shared gateway message abstractions only when multiple platforms need the
  same behavior.
- Fork update behavior should avoid changing user-facing upstream docs or core
  CLI flow if it can be represented as a fork-only helper script or branch
  maintenance note.
