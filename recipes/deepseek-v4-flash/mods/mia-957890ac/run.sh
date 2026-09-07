#!/usr/bin/env bash
# sparkrun mod: MiaAI-Lab DSpark hotfix stack for DeepSeek-V4-Flash-Vision-Exp
# on the Anemll image (ghcr.io/anemll/dspark-vllm-gx10:0.1.1), pin
# 957890ac5e26ce149646719298169f80603a3e1f (MIT).
#
# Reuses upstream instead of porting it: fetch the pinned repo, stage its
# compose volume bindings, and execute its compose entrypoint verbatim minus
# the trailing `exec vllm serve ...` (sparkrun builds the serve command).
# Hotfix selection/order/hatches are upstream's code; a new pin means a new
# mod dir (mia-<sha8>-dsv4-dspark-anemll). Cold cache needs GitHub egress once
# per node per pin; the digest below guards the persisted cache only (a
# codeload fetch is content-addressed by commit). Digest excludes
# entrypoint.sh, which this script generates; LC_ALL=C sort is mandatory
# (host UTF-8 vs image C collation). Upstream DSPARK_ENABLE_* opt-ins stay
# unwired (unset env).
set -euo pipefail

SHA="957890ac5e26ce149646719298169f80603a3e1f"
REPO="MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark"
DIGEST="6105c5224088052b156f8009578c449ca950e2c78c04a008c4d784934278df6e"
ROOT="${DSPARK_PATCH_CACHE_DIR:-/cache/runtime/dspark-mod}/mia-${SHA:0:8}"

export VLLM_ROOT="${VLLM_ROOT:-/usr/local/lib/python3.12/dist-packages/vllm}"
MOD_DIR="$(cd "$(dirname "$0")" && pwd)"

[ -d "$VLLM_ROOT" ] || { echo "dspark-mod: FATAL: vLLM not found at $VLLM_ROOT" >&2; exit 1; }
# Whole-snapshot manifest check (the only guard against silent cache corruption).
manifest() { (cd "$1" && find . -type f -not -path "./entrypoint.sh" -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | grep -q "$DIGEST"); }
log() { echo "dspark-mod: $*"; }

if ! manifest "$ROOT" 2>/dev/null; then
  log "cache miss — fetching $REPO @ ${SHA:0:8}"
  T=$(mktemp -d)
  if curl -fsSL --connect-timeout 10 --max-time 120 "https://codeload.github.com/$REPO/tar.gz/$SHA" \
      | tar xz -C "$T" --strip-components=1 && manifest "$T"; then
    mkdir -p "$(dirname "$ROOT")"; rm -rf "$ROOT"; mv "$T" "$ROOT"
    log "fetched + verified + cached"
  fi
  rm -rf "$T"
fi
[ -d "$ROOT" ] || { echo "dspark-mod: FATAL: no verified upstream snapshot — GitHub egress is required to boot this recipe from a cold cache" >&2; exit 1; }

# Stage upstream's compose volume list ('${VAR:-./src}:/dst[:ro]') to /opt —
# derived from the pinned compose, not hand-enumerated.
python3 - "$ROOT" <<'PYEOF'
import re, shutil, sys, yaml
from pathlib import Path

upstream = Path(sys.argv[1])
vols = yaml.safe_load(open(upstream / "docker-compose.dspark.yml"))["services"]["vllm-dspark"]["volumes"]
for vol in vols:
    m = re.match(r"^\$\{[A-Za-z0-9_]+:-\./([^{}]+)\}:(.+?)(?::ro)?$", vol)
    if not m or not (upstream / m.group(1)).exists():
        continue  # host-only mounts / absent optional dirs
    src, dst = upstream / m.group(1), Path(m.group(2))
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(f"dspark-mod: staged {m.group(1)} -> {dst}")
PYEOF

# Extract upstream's entrypoint (compose `command:` items 3+, $$ un-escaped)
# minus the final `exec vllm serve` line — sparkrun builds the serve command.
python3 - "$ROOT/docker-compose.dspark.yml" "$ROOT/entrypoint.sh" <<'PYEOF'
import sys, yaml
cmd = [ln.strip() for ln in yaml.safe_load(open(sys.argv[1]))["services"]["vllm-dspark"]["command"] if ln.strip()]
script = "\n".join(cmd[2:] if cmd[:2] == ["bash", "-lc"] else cmd).replace("$$", "$")
assert "exec /usr/local/bin/vllm serve" in script
open(sys.argv[2], "w").write(script.split("exec /usr/local/bin/vllm serve", 1)[0])
PYEOF
log "running upstream entrypoint (pre-serve portion)"
bash "$ROOT/entrypoint.sh"

# Upstream WARNs on a missing encoder; fail closed instead (a silent
# encoder-less boot changes tool-call/reasoning token encoding).
if [ -f "$VLLM_ROOT/tokenizers/deepseek_v4_encoding.py" ]; then
  log "OK — encoder installed, all hotfixes applied"
else
  echo "dspark-mod: FATAL: encoding_dsv4.py not found (set DSPARK_ENCODING_FILE, or check DSPARK_MODEL/DSPARK_REVISION)" >&2
  exit 1
fi
