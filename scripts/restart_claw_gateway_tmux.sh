#!/usr/bin/env bash
set -euo pipefail

TMUX_SESSION="${TMUX_SESSION:-claw}"
TMUX_WINDOW="${TMUX_WINDOW:-gateway}"
PORT="${1:-18789}"

REPO_DIR="/mnt/shared-storage-user/liuhongwei/main_works/repos/repro"
HOME_DIR="/mnt/shared-storage-user/liuhongwei"
NODE_BIN="/mnt/shared-storage-user/liuhongwei/.nvm/versions/node/v22.12.0/bin/node"
OPENCLAW_MJS="/mnt/shared-storage-user/liuhongwei/.nvm/versions/node/v22.12.0/lib/node_modules/openclaw/openclaw.mjs"
PATH_VALUE="/mnt/shared-storage-user/liuhongwei/.nvm/versions/node/v22.12.0/bin:/opt/conda/bin:/kubebrain:/usr/local/nvidia/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

if ! tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${TMUX_SESSION}" -n "${TMUX_WINDOW}"
fi

if ! tmux list-windows -t "${TMUX_SESSION}" -F '#W' | grep -qx "${TMUX_WINDOW}"; then
  tmux new-window -d -t "${TMUX_SESSION}" -n "${TMUX_WINDOW}"
fi

TARGET="${TMUX_SESSION}:${TMUX_WINDOW}.0"
START_CMD="cd ${REPO_DIR} && export HOME=${HOME_DIR} && export PATH=${PATH_VALUE} && ${NODE_BIN} ${OPENCLAW_MJS} gateway --port ${PORT}"

# Use respawn-pane to avoid mixed keystrokes in a busy pane.
tmux respawn-pane -k -t "${TARGET}" "bash -lc '${START_CMD}'"

echo "[restart] started in tmux ${TARGET}"
echo "[restart] waiting for port ${PORT}..."

for _ in $(seq 1 90); do
  NEW_PID="$(ss -lntp | sed -n "s/.*:${PORT} .*pid=\\([0-9]\\+\\).*/\\1/p" | head -n 1 || true)"
  if [ -n "${NEW_PID}" ]; then
    echo "[restart] gateway is listening on ${PORT} (pid ${NEW_PID})"
    break
  fi
  sleep 1
done

if ! ss -lntp | grep -q ":${PORT} "; then
  echo "[restart] gateway did not listen on ${PORT} in time"
  tmux capture-pane -pt "${TARGET}" -S -120 | tail -n 120
  exit 1
fi

ss -lntp | grep ":${PORT} " || true
echo "[restart] recent tmux pane output:"
tmux capture-pane -pt "${TARGET}" -S -80 | tail -n 80
