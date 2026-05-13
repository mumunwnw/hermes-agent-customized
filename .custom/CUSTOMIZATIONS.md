# Custom Branch Registry

This fork now uses a two-track layout:

- `main`: exact mirror of `upstream/main`
- `custom`: fork-only runtime branch

Do not commit local or personalized changes to `main`. All fork behavior should
land on `custom` or on short-lived topic branches that are later merged into
`custom`.

## Branch Policy

- `main`: pure upstream mirror. Only fast-forward it to `upstream/main`, then
  push the same commit to `origin/main`.
- `custom`: the only long-lived fork runtime branch. This is the branch that
  the installed Hermes runtime should track.
- `feature/*`: topic branches for individual custom features that we are still
  studying or rewriting.
- `codex/*`: temporary maintenance or rewrite branches. Merge what we want into
  `custom`, then delete them.

## Runtime Layout

Development and runtime are intentionally separate:

- Development checkout:
  `/Users/mumu/github-repo/hermes-agent-customized`
- Runtime data/config:
  `~/.hermes`
- Runtime code checkout:
  `~/.hermes/hermes-agent`

The runtime checkout must pull from the fork's remote `custom` branch. It must
not share the development checkout, and it must not use `main` as its update
branch.

Current runtime expectations:

- remote: `origin`
- branch: `custom`
- local git config in the runtime checkout:
  - `hermes.updateRemote=origin`
  - `hermes.updateBranch=custom`

## Current State

Snapshot date: 2026-05-14.

Upstream mirror commit:

- `main`, `origin/main`, `upstream/main`:
  `942adf617910f50a39f41bd200d8083bf4cb2bed`

Fork runtime branch:

- `custom` / `origin/custom`:
  `8c928d8e25c2c9871f427b1c67999e6c4acadc36`
- Relationship to `upstream/main`:
  `0` upstream-only commits on `custom`, `7` fork-only commits on `custom`

Custom work already merged into `custom`:

| Area | Source | Status | Purpose |
| --- | --- | --- | --- |
| Runtime update source | `codex/rewrite-update-custom-branch` | merged | `hermes update` reads its remote/branch from config or local git config and can track `origin/custom` instead of forcing `origin/main`. |
| Installer repo selection | `codex/rewrite-update-custom-branch` | merged | `scripts/install.sh` supports `--repo` / `HERMES_REPO_URL` and records runtime update metadata in `.git/config`. |
| Launcher safety | `codex/rewrite-update-custom-branch` | merged | Installer removes an old `~/.local/bin/hermes` symlink before writing the launcher so it cannot overwrite the real venv entrypoint. |
| Custom branch registry | `codex/customization-registry` | merged | This `.custom/` directory and `scripts/customization_report.sh` are now maintained directly on `custom`. |

Custom work still living only on topic branches:

| Branch | Head | Status | Purpose |
| --- | --- | --- | --- |
| `feature/feishu-messages-enhanced` | `5323c5783` | not yet rewritten onto `custom` | Feishu paragraph-aware delivery, pacing, list and short-paragraph merging. |
| `feature/hybrid-session-search` | `4eb94a35c` | not yet rewritten onto `custom` | Hybrid session search with BM25, vector search, RRF, indexing tools, diagnostics, and logging. |
| `stable` | `5323c5783` | legacy branch, no longer authoritative | Old integration branch kept only as historical reference until remaining features are rewritten. |

## Operating Rules

1. Keep `main` pure.
2. Merge only reviewed, intentional fork behavior into `custom`.
3. Keep the runtime checkout on `custom`; upstream sync belongs in the
   development checkout, not in the runtime install.
4. Prefer plugins, optional skills, config keys, and isolated adapters over
   broad edits to central files.
5. If a central file must change, hide the fork behavior behind a narrow
   helper, config gate, or runtime metadata.
6. Every remaining customization should end up with:
   - one topic branch,
   - one short entry in this registry,
   - focused tests,
   - one clear drop condition describing when upstream replaces it.

## Update Flow

Development mirror refresh:

```bash
git switch main
git fetch upstream
git merge --ff-only upstream/main
git push origin main
```

Custom feature work:

```bash
git switch -c feature/<area> main
# implement or rewrite
git switch custom
git merge --no-ff feature/<area>
git push origin custom
```

Runtime install refresh:

```bash
cd ~/.hermes/hermes-agent
hermes update
```

That runtime update must fetch from the remote configured for the runtime
checkout, currently `origin/custom`.

## Verification

Run this after upstream syncs or custom merges:

```bash
git rev-parse main origin/main upstream/main
git diff --stat upstream/main..main
git rev-list --left-right --count upstream/main...main

git rev-list --left-right --count upstream/main...custom
git diff --stat upstream/main...custom
git cherry -v upstream/main custom
```

For the runtime install:

```bash
git -C ~/.hermes/hermes-agent branch --show-current
git -C ~/.hermes/hermes-agent config --get hermes.updateRemote
git -C ~/.hermes/hermes-agent config --get hermes.updateBranch
```

## Next Refactor Targets

- Rewrite `feature/feishu-messages-enhanced` onto a fresh topic branch from
  `main`, then merge the clean version into `custom`.
- Rewrite `feature/hybrid-session-search` onto a fresh topic branch from
  `main`, ideally pushing more of it behind plugin or provider boundaries.
- Delete `stable` after the remaining historical feature content is either
  rewritten or consciously discarded.
