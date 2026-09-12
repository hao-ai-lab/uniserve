#!/usr/bin/env bash
#SBATCH --job-name=uniserve-h3-ref
#SBATCH --partition=hpc-rack-3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --export=NIL
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/lustre/vlm-wlsaidhi/src/uniserve-deploy/serve-h3-ref-%j.log
set -euo pipefail
# Never enable xtrace: UniServe accepts the bearer token as a CLI value.
D=/mnt/lustre/vlm-wlsaidhi/src/uniserve-deploy
R=/mnt/lustre/vlm-wlsaidhi/src/uniserve-wt/omniref
# Positional form (sbatch --export=NIL serve-h3-ref.sbatch ENTRY ROOT): --export=NIL,VAR=... makes
# Slinky run get-user-env, which fails on this cluster ("user env retrieval failed").
if [[ ${1:-} != --inside && -n ${1:-} ]]; then export ENTRY=$1 CHECKPOINT_ROOT=$2; shift 2; fi
export ENTRY=${ENTRY:-ref}
export CHECKPOINT_ROOT=${CHECKPOINT_ROOT:?Set an absolute full checkpoint root}
unset UNISERVE_H3_CONTRACT UNISERVE_H3_VARIANT
case "$ENTRY" in
    ref) export PORT=8102 MODE=quality UNISERVE_H3_VARIANT=ref ;;
    *) echo 'ENTRY must be ref' >&2; exit 2 ;;
esac
[[ $CHECKPOINT_ROOT = /* && -d $CHECKPOINT_ROOT ]]
: "${SLURM_JOB_ID:?Run only in an allocation}"
export AGENTS_HOME=/mnt/lustre/vlm-wlsaidhi/.agents
export RUN_ID=${RUN_ID:-}
export KEY_FILE=/mnt/lustre/vlm-wlsaidhi/deployments/deepseek-v41-flash-rack3-20260910/nvme/api-key
export ENDPOINT_FILE="$D/endpoints/h3-$ENTRY-rack3.json"
if [[ ${1:-} != --inside ]]; then
    # Use the scheduler's actual deadline, including time spent waiting to start.
    end_time=$(squeue --noheader --jobs="$SLURM_JOB_ID" --format=%e)
    EXPIRES_UTC=$(date -u -d "$end_time" +%FT%TZ)
    export EXPIRES_UTC
    export SLURM_EXPORT_ENV=ALL
    cleanup() {
        rc=$?
        /usr/bin/python3 "$R/scripts/h3/h3-endpoint.py" remove || true
        if [[ -n $RUN_ID ]]; then
            status=failed
            [[ $rc == 0 ]] && status=succeeded
            "$AGENTS_HOME/bin/runctl" finish "$RUN_ID" --job-id "$SLURM_JOB_ID" \
                --status "$status" --exit-code "$rc" --summary 'H3 serving allocation exited' || true
        fi
    }
    trap cleanup EXIT
    srun --nodes=1 --ntasks=1 --gpus=4 \
        --container-image=/mnt/lustre/vlm-wlsaidhi/deployments/deepseek-v41-flash-rack3-20260910/ready/images/vllm-deepseekv41-arm64-compute.sqsh \
        --container-remap-root --container-writable \
        --container-mounts=/mnt/lustre:/mnt/lustre --container-workdir="$R" \
        bash "$D/serve-h3-ref.sbatch" --inside
    exit $?
fi

cd "$R"
export HOME="$D/container-home" XDG_CACHE_HOME="$D/cache" TMPDIR="$D/tmp"
export CARGO_HOME="$D/toolchain/cargo" RUSTUP_HOME="$D/toolchain/rustup"
export CUDA_HOME=/usr/local/cuda UV_CACHE_DIR="$D/cache"
export UV_PROJECT_ENVIRONMENT="$R/.venv-gpu"
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$CARGO_HOME/bin:$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/bin"
export TORCH_EXTENSIONS_DIR="$D/cache/torch_extensions" TRITON_CACHE_DIR="$D/cache/triton"
# The vLLM image ships no CUDA dev headers (cusparse.h); the pip nvidia-cu13 wheel does.
NV_INC="$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/cu13/include"
export CPATH="$NV_INC${CPATH:+:$CPATH}" CUDA_INC_PATH="$NV_INC"
export MAX_JOBS=2 CARGO_BUILD_JOBS=2 NCCL_CUMEM_HOST_ENABLE=1
mkdir -p "$HOME" "$TMPDIR" "$D/endpoints"
# Reuse the shared Rust toolchain and the existing worker-venv convention;
# build this checkout, never execute a native extension from another worktree.
apt-get update
apt-get install -y build-essential python3.12-dev pkg-config cmake git curl ca-certificates libclang-dev ninja-build
printf 'UniServe source commit: '
git rev-parse HEAD
git status --short
sha256sum uv.lock Cargo.lock "$D/serve-h3-ref.sbatch"
UV="$D/toolchain/uv/uv"
if [[ ! -x $UV ]]; then
    echo 'Missing shared uv binary; provision it before launching' >&2
    exit 1
fi
cargo --version
"$UV" sync --locked --python /usr/bin/python3.12 --extra h3
python "$R/scripts/h3/h3-endpoint.py" validate
ulimit -l unlimited
shm_bytes=$(df -B1 --output=avail /dev/shm | tail -1)
(( shm_bytes >= 4294967296 ))
if [[ -n $RUN_ID ]]; then
    "$AGENTS_HOME/bin/runctl" runtime "$RUN_ID" --job-id "$SLURM_JOB_ID" \
        --python "$UV_PROJECT_ENVIRONMENT/bin/python" --observed graph_policy=off || true
fi
ADVERTISE_HOST=$(hostname -I | awk '{print $1}')
[[ -n $ADVERTISE_HOST ]]
export ADVERTISE_HOST
key=$(< "$KEY_FILE")
[[ -n $key ]]
server_pid=
monitor_pid=
cleanup_inside() {
    rc=$?
    trap - EXIT
    [[ -z $monitor_pid ]] || kill "$monitor_pid" 2>/dev/null || true
    [[ -z $server_pid ]] || kill "$server_pid" 2>/dev/null || true
    [[ -z $server_pid ]] || wait "$server_pid" 2>/dev/null || true
    python "$R/scripts/h3/h3-endpoint.py" remove || true
    exit "$rc"
}
trap cleanup_inside EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
uniserve serve "$CHECKPOINT_ROOT" --model-description minimax-h3 \
    --worker-python "$UV_PROJECT_ENVIRONMENT/bin/python" --worker-ranks 4 \
    --served-model-name "minimax-h3-$ENTRY" --host 0.0.0.0 --port "$PORT" --api-key "$key" \
    --max-model-len 1024 --max-video-seconds 5 \
    --quantization-config "{\"mode\":\"$MODE\"}" --attention-backend auto --graph-policy off &
server_pid=$!
unset key
python "$R/scripts/h3/h3-endpoint.py" publish &
monitor_pid=$!
wait "$server_pid"
