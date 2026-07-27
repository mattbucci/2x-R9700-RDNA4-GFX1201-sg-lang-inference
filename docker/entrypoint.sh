#!/usr/bin/env bash
set -euo pipefail

readonly installed_gpu_selection="/usr/local/libexec/rdna4/gpu-selection.sh"
if [[ -r "$installed_gpu_selection" ]]; then
    gpu_selection="$installed_gpu_selection"
elif [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    # Source-tree fallback is for offline unit tests only. The installed
    # entrypoint never sources code from an environment-controlled path.
    repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    gpu_selection="${repo_dir}/scripts/gpu-selection.sh"
else
    echo "ERROR: installed GPU-selection helper is missing" >&2
    exit 70
fi
source "$gpu_selection"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    umask 077
    configure_gpu_selection 0
    if (( $# == 0 )); then
        echo "Usage: docker run IMAGE scripts/launch.sh <preset> [options]" >&2
        exit 64
    fi
    source /opt/conda/etc/profile.d/conda.sh
    conda activate sglang-rdna4
    exec "$@"
fi
