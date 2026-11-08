#!/usr/bin/env bash
# Scheduler-friendly quarterly retrain of the paper model (ExploitationZone/model_v3_altall.pkl).
#
#   DataAnalysisPipeline2/scripts/retrain_altall.sh [extra retrain_altall args, e.g. --skip-refresh]
#
# 1. python -m DataAnalysisPipeline2.v3.retrain_altall  (refresh -> frame -> train -> validate -> promote)
# 2. pytest (trading_agent, features, v3); if it fails after a promotion the previous model is restored
# 3. commit the model + manifest + DataAnalysisPipeline2/results/retrain_log.md and push
#    (a rejected candidate still commits + pushes the log entry; nothing is committed on errors)
#
# Env: PY (default python), BRANCH (default main; refuses to run on another branch),
#      PUSH=0 to commit without pushing, REMOTE (default origin).
# Needs: $BDA_DATA_V3 store, Alpaca keys + SEC_USER_AGENT in DataAnalysisPipeline2/trading_agent/.env,
#        git-lfs. The data refresh rewrites shared research frames in $BDA_DATA_V3: do not schedule
#        it while a research round is running.
#
# Example cron entry (NOT installed by this repo): first Saturday of Jan/Apr/Jul/Oct, 06:00
#   0 6 1-7 1,4,7,10 6 cd /path/to/BDA-P1 && DataAnalysisPipeline2/scripts/retrain_altall.sh >> "$HOME/retrain_altall.log" 2>&1
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 1
PY="${PY:-python}"
BRANCH="${BRANCH:-main}"
REMOTE="${REMOTE:-origin}"
PUSH="${PUSH:-1}"
MODEL=ExploitationZone/model_v3_altall.pkl
MANIFEST=ExploitationZone/model_v3_altall.manifest.json
LOG=DataAnalysisPipeline2/results/retrain_log.md
LOCK="${TMPDIR:-/tmp}/bda_retrain_altall.lock"

log() { echo "[retrain_altall.sh $(date '+%Y-%m-%d %H:%M:%S')] $*"; }

if ! mkdir "$LOCK" 2>/dev/null; then
  log "another retrain is running ($LOCK); exiting"; exit 1
fi
trap 'rmdir "$LOCK"' EXIT

cur="$(git rev-parse --abbrev-ref HEAD)"
if [ "$cur" != "$BRANCH" ]; then
  log "on branch '$cur', expected '$BRANCH'; exiting"; exit 1
fi
if ! git diff --quiet -- "$MODEL" "$MANIFEST" "$LOG" || ! git diff --cached --quiet; then
  log "uncommitted changes to the model/log or staged changes present; exiting"; exit 1
fi
git pull --ff-only "$REMOTE" "$BRANCH" || { log "git pull failed"; exit 1; }

"$PY" -m DataAnalysisPipeline2.v3.retrain_altall "$@"
rc=$?
log "retrain_altall exit code $rc"
case $rc in
  0)
    if git diff --quiet -- "$MODEL"; then
      log "model unchanged (validated but not promoted)"; files=("$LOG"); msg="Retrain log: paper model validated, not promoted"
    else
      files=("$MODEL" "$MANIFEST" "$LOG")
      mid="$("$PY" -c "import json; print(json.load(open('$MANIFEST'))['model_id'])")"
      msg="Quarterly retrain of the paper model: $mid (fixed sn30_altall_hd recipe; validation passed)"
    fi ;;
  3) files=("$LOG"); msg="Retrain log: paper-model candidate rejected, live model kept" ;;
  *) log "retrain failed; nothing committed"; exit "$rc" ;;
esac

if ! "$PY" -m pytest -q DataAnalysisPipeline2/trading_agent/ DataAnalysisPipeline2/features/ DataAnalysisPipeline2/v3/; then
  log "tests FAILED"
  if ! git diff --quiet -- "$MODEL"; then
    "$PY" -m DataAnalysisPipeline2.v3.retrain_altall --rollback && log "previous model restored"
  fi
  git checkout -- "$LOG" 2>/dev/null
  exit 1
fi

git add -- "${files[@]}"
if git diff --cached --quiet; then
  log "nothing to commit"; exit 0
fi
git commit -q -m "$msg" || { log "commit failed"; exit 1; }
log "committed: $msg"
if [ "$PUSH" = "1" ]; then
  git push "$REMOTE" "$BRANCH" || { log "push failed"; exit 1; }
  log "pushed to $REMOTE/$BRANCH"
fi
exit 0
