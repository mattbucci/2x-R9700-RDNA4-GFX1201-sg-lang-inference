#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
entrypoint="$repo_dir/docker/entrypoint.sh"

check() {
    local ids=$1 tp=$2 expected=$3
    GPU_IDS="$ids" TP="$tp" bash -c \
        "source '$entrypoint'; configure_gpu_selection 0; printf '%s/%s/%s' \"\$HIP_VISIBLE_DEVICES\" \"\$TP\" \"\${SGLANG_RDNA4_DISABLE_STORE_CACHE:-0}\"" \
        | grep -qx "$expected"
}

reject() {
    if env "$@" bash -c "source '$entrypoint'; configure_gpu_selection 0" >/dev/null 2>&1; then
        return 1
    fi
}

check 0 1 0/1/1
check 0,1 2 0,1/2/0
env -u GPU_IDS -u TP bash -c "source '$entrypoint'; configure_gpu_selection 0; printf '%s/%s/%s' \"\$GPU_IDS\" \"\$TP\" \"\${SGLANG_RDNA4_DISABLE_STORE_CACHE:-0}\"" | grep -qx '0/1/1'
env -u GPU_IDS -u TP bash -c "source '$repo_dir/scripts/gpu-selection.sh'; configure_gpu_selection 0,1; printf '%s/%s/%s' \"\$GPU_IDS\" \"\$TP\" \"\${SGLANG_RDNA4_DISABLE_STORE_CACHE:-0}\"" | grep -qx '0,1/2/0'
reject GPU_IDS=0,0
reject GPU_IDS=0,x

# A pre-set visibility variable with no GPU_IDS is the documented bare-metal
# form and must be adopted as the default, not rejected as a conflict.
accept_preset() {
    local variable=$1 ids=$2 expected=$3
    env -u GPU_IDS -u TP "$variable=$ids" bash -c \
        "source '$repo_dir/scripts/gpu-selection.sh'; configure_gpu_selection 0,1; printf '%s/%s' \"\$GPU_IDS\" \"\$TP\"" \
        | grep -qx "$expected"
}
for variable in HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL CUDA_VISIBLE_DEVICES; do
    accept_preset "$variable" 0 0/1
    accept_preset "$variable" 0,1 0,1/2
done
env -u GPU_IDS HIP_VISIBLE_DEVICES=0 TP=1 bash -c \
    "source '$repo_dir/scripts/gpu-selection.sh'; configure_gpu_selection 0,1; printf '%s/%s' \"\$GPU_IDS\" \"\$TP\"" \
    | grep -qx '0/1'
for variable in HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL CUDA_VISIBLE_DEVICES; do
    reject GPU_IDS=0 "$variable=1"
done
reject GPU_IDS=0 TP=2

grep -qx 'WORKDIR ${REPO_DIR}' "$repo_dir/Dockerfile"
grep -qx 'USER ${APP_UID}:${APP_GID}' "$repo_dir/Dockerfile"
grep -qx 'EXPOSE 23334' "$repo_dir/Dockerfile"
grep -q 'SGLANG_SECURE_LAUNCH=1 SGLANG_TRUST_REMOTE_CODE=0 SGLANG_ENABLE_METRICS=0' "$repo_dir/Dockerfile"
grep -q 'SGLANG_ALLOW_LOCAL_MEDIA=0 SGLANG_ALLOW_REMOTE_MEDIA=0 SGLANG_USE_PICKLE_IPC=0' "$repo_dir/Dockerfile"
grep -q 'NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1' "$repo_dir/Dockerfile"
head -n 1 "$repo_dir/docker/secure-launch.py" | grep -qx '#!/opt/conda/envs/sglang-rdna4/bin/python'
! grep -q '^COPY \. ' "$repo_dir/Dockerfile"
grep -qxF '**' "$repo_dir/.dockerignore"
(cd / && REPO_DIR="$repo_dir" bash -c 'source "$1"; configure_gpu_selection 0; test -x "$REPO_DIR/scripts/launch.sh"' _ "$entrypoint")

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
mkdir -p "$tmpdir/bin"
printf '%s\n' '#!/usr/bin/env bash' '[[ "$1" == shell.bash ]] && printf "conda() { return 0; }\n"' > "$tmpdir/bin/conda"
printf '%s\n' '#!/usr/bin/env bash' 'printf test' > "$tmpdir/bin/python"
chmod +x "$tmpdir/bin/conda" "$tmpdir/bin/python"
LAUNCH_DRY_RUN=1 CONDA_BASE="$tmpdir" PATH="$tmpdir/bin:$PATH" GPU_IDS=0 TP=1 "$repo_dir/scripts/launch.sh" coder-30b | grep -q -- '--tensor-parallel-size 1'
LAUNCH_DRY_RUN=1 CONDA_BASE="$tmpdir" PATH="$tmpdir/bin:$PATH" GPU_IDS=0,1 TP=2 "$repo_dir/scripts/launch.sh" coder-30b | grep -q -- '--tensor-parallel-size 2'

# The installed entrypoint must not source a helper from an environment-selected
# repository path. Source-tree tests use the entrypoint's own parent directory.
mkdir -p "$tmpdir/override/scripts"
printf '%s\n' 'touch "$INJECTION_MARKER"' > "$tmpdir/override/scripts/gpu-selection.sh"
INJECTION_MARKER="$tmpdir/injected" REPO_DIR="$tmpdir/override" bash -c \
    'source "$1"; configure_gpu_selection 0' _ "$entrypoint"
test ! -e "$tmpdir/injected"

# Container policy requires two file-backed, distinct keys; dry-run output must
# never put either value in argv and must not restore model remote-code execution.
if env -u SGLANG_API_KEY -u SGLANG_ADMIN_API_KEY \
    -u SGLANG_API_KEY_FILE -u SGLANG_ADMIN_API_KEY_FILE \
    SGLANG_SECURE_LAUNCH=1 LAUNCH_DRY_RUN=1 CONDA_BASE="$tmpdir" \
    PATH="$tmpdir/bin:$PATH" GPU_IDS=0 TP=1 \
    "$repo_dir/scripts/launch.sh" coder-30b >/dev/null 2>&1; then
    exit 1
fi
api_key='api_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
admin_key='admin_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'
printf '%s\n' "$api_key" > "$tmpdir/api-key"
printf '%s\n' "$admin_key" > "$tmpdir/admin-api-key"
chmod 0404 "$tmpdir/api-key" "$tmpdir/admin-api-key"
dry_run="$(
    SGLANG_SECURE_LAUNCH=1 SGLANG_TRUST_REMOTE_CODE=0 SGLANG_ENABLE_METRICS=0 \
    SGLANG_MAX_QUEUED_REQUESTS=32 \
    SGLANG_API_KEY_FILE="$tmpdir/api-key" \
    SGLANG_ADMIN_API_KEY_FILE="$tmpdir/admin-api-key" \
    LAUNCH_DRY_RUN=1 CONDA_BASE="$tmpdir" PATH="$tmpdir/bin:$PATH" \
    GPU_IDS=0 TP=1 "$repo_dir/scripts/launch.sh" coder-30b
)"
grep -q -- '/usr/local/libexec/rdna4/secure-launch.py' <<< "$dry_run"
! grep -q -- '--max-queued-requests' <<< "$dry_run"
! grep -q -- '--api-key' <<< "$dry_run"
! grep -q -- '--admin-api-key' <<< "$dry_run"
! grep -q -- '--trust-remote-code' <<< "$dry_run"
! grep -q -- '--enable-metrics' <<< "$dry_run"
! grep -qF "$api_key" <<< "$dry_run"
! grep -qF "$admin_key" <<< "$dry_run"

# Protected flags cannot be smuggled through the launcher's extensibility
# variables. The Python shim independently enforces the parsed ServerArgs too.
if SGLANG_SECURE_LAUNCH=1 SGLANG_API_KEY_FILE="$tmpdir/api-key" \
    SGLANG_ADMIN_API_KEY_FILE="$tmpdir/admin-api-key" \
    EXTRA_ARGS='--config /tmp/untrusted.yaml' LAUNCH_DRY_RUN=1 \
    CONDA_BASE="$tmpdir" PATH="$tmpdir/bin:$PATH" GPU_IDS=0 TP=1 \
    "$repo_dir/scripts/launch.sh" coder-30b >/dev/null 2>&1; then
    exit 1
fi
if SGLANG_SECURE_LAUNCH=1 SGLANG_API_KEY_FILE="$tmpdir/api-key" \
    SGLANG_ADMIN_API_KEY_FILE="$tmpdir/admin-api-key" \
    EXTRA_ENV='SGLANG_TRUST_REMOTE_CODE=1' LAUNCH_DRY_RUN=1 \
    CONDA_BASE="$tmpdir" PATH="$tmpdir/bin:$PATH" GPU_IDS=0 TP=1 \
    "$repo_dir/scripts/launch.sh" coder-30b >/dev/null 2>&1; then
    exit 1
fi
if SGLANG_SECURE_LAUNCH=1 SGLANG_API_KEY_FILE="$tmpdir/api-key" \
    SGLANG_ADMIN_API_KEY_FILE="$tmpdir/admin-api-key" \
    EXTRA_ENV='SGLANG_ALLOW_REMOTE_MEDIA=1' LAUNCH_DRY_RUN=1 \
    CONDA_BASE="$tmpdir" PATH="$tmpdir/bin:$PATH" GPU_IDS=0 TP=1 \
    "$repo_dir/scripts/launch.sh" coder-30b >/dev/null 2>&1; then
    exit 1
fi
if SGLANG_SECURE_LAUNCH=1 SGLANG_API_KEY_FILE="$tmpdir/api-key" \
    SGLANG_ADMIN_API_KEY_FILE="$tmpdir/admin-api-key" \
    EXTRA_ENV='DUMPER_SERVER_PORT=40000' LAUNCH_DRY_RUN=1 \
    CONDA_BASE="$tmpdir" PATH="$tmpdir/bin:$PATH" GPU_IDS=0 TP=1 \
    "$repo_dir/scripts/launch.sh" coder-30b >/dev/null 2>&1; then
    exit 1
fi
if SGLANG_SECURE_LAUNCH=1 SGLANG_API_KEY_FILE="$tmpdir/api-key" \
    SGLANG_ADMIN_API_KEY_FILE="$tmpdir/admin-api-key" \
    EXTRA_ENV='SGLANG_TEST_SCRIPTED_RUNTIME=1' LAUNCH_DRY_RUN=1 \
    CONDA_BASE="$tmpdir" PATH="$tmpdir/bin:$PATH" GPU_IDS=0 TP=1 \
    "$repo_dir/scripts/launch.sh" coder-30b >/dev/null 2>&1; then
    exit 1
fi
if SGLANG_SECURE_LAUNCH=1 SGLANG_API_KEY_FILE="$tmpdir/api-key" \
    SGLANG_ADMIN_API_KEY_FILE="$tmpdir/admin-api-key" \
    EXTRA_ARGS='--kv-events-config {}' LAUNCH_DRY_RUN=1 \
    CONDA_BASE="$tmpdir" PATH="$tmpdir/bin:$PATH" GPU_IDS=0 TP=1 \
    "$repo_dir/scripts/launch.sh" coder-30b >/dev/null 2>&1; then
    exit 1
fi

grep -q 'websocket.close' "$repo_dir/patches/096-sglang-api-auth-hardening.patch"
grep -q '@auth_level(AuthLevel.ADMIN_FORCE)' "$repo_dir/patches/096-sglang-api-auth-hardening.patch"
grep -q 'SGLANG_ALLOW_REMOTE_MEDIA' "$repo_dir/patches/096-sglang-api-auth-hardening.patch"
grep -q 'multimodal/processors/mimo_v2.py' "$repo_dir/patches/096-sglang-api-auth-hardening.patch"
grep -q '_enforce_delegated_media_source_policy' "$repo_dir/patches/096-sglang-api-auth-hardening.patch"
grep -q 'bootstrap_host' "$repo_dir/patches/096-sglang-api-auth-hardening.patch"
python "$repo_dir/tests/test_secure_launch.py"
