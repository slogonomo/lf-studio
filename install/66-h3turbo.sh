#!/usr/bin/env bash
# 66-h3turbo.sh: optional, DGX Spark only. Sets up the experimental LightX2V Turbo model (LFSTUDIO_MODELS=sol-h3,
# lightx2v-turbo; docs/INSTALL.md, "Experimental: LightX2V Turbo"): the full MiniMax-H3 in one stage with LightX2V's
# turbo LoRA, as a second model next to Sol-H3.
#  1. The base model, LFSTUDIO_H3TURBO_BASE (default $LFSTUDIO_CHECKPOINTS/MiniMax-H3, from 20-weights.sh), checked
#     only: transformer/, vae/ and audio_vae/ with their configs and weights, scheduler/ and audio_scheduler/. NVIDIA's
#     t2va-only download has vae/config.json without the VAE's weights (they come with fl2va or ref2va), so a base
#     without them is refused: run 20-weights.sh with fl2va in LFSTUDIO_TASKS (its default has it).
#  2. Its Python: the Upscale's environment, $LFSTUDIO_HOME/envs/regen2k (torch 2.11.0+cu130 and diffusers 0.40.0, from
#     install/freeze/regen2k-lock.txt). It is used as it is when 65-regen2k.sh has marked it done with this lock; else
#     this runs 65-regen2k.sh --env-only, which builds it without the Upscale's weights. This script never builds,
#     copies or changes a venv itself. A LFSTUDIO_H3TURBO_PYTHON of your own is used as it is (the check in step 4
#     says whether it fits).
#  3. LightX2V's turbo LoRA files, lightx2v/Minimax-h3-Turbo at revision 30f485ed, each pinned with its size and
#     SHA-256 (install/h3turbo_weights.py; no token is ever sent), into LFSTUDIO_H3TURBO_LORAS (default
#     $LFSTUDIO_CHECKPOINTS/Minimax-h3-Turbo, where 20-weights.sh put NVIDIA's ref2va LoRA): the 4-step v1.3 file the
#     studio runs and the 8-step v1.1 file, 1.38 GB each. Files already there with their SHA-256 are kept.
#  4. The job's own check, without the GPU and without reading weights: jobs/h3turbo_job.py --check-install (its
#     packages, the base model, the LoRA files).
# Downloads 2.8 GB of LoRA files, and without the Upscale's environment about 3.2 GB of packages (the venv takes about
# 5.5 GB). Then add lightx2v-turbo to LFSTUDIO_MODELS in the settings file and restart. Needs 20-weights.sh (the base
# model), and 10-sana.sh for the environment. Every step stops the script when it fails (#74). Idempotent: what is in
# place is kept.
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
if [[ "$LFSTUDIO_BACKEND" != spark ]]; then
    say "LFSTUDIO_BACKEND=$LFSTUDIO_BACKEND: nothing to do (the experimental LightX2V Turbo runs on the DGX Spark only)"
    exit 0
fi
need_cmd sha256sum

BASE=$LFSTUDIO_H3TURBO_BASE                            # install/env.sh: the server's defaults
LORAS=$LFSTUDIO_H3TURBO_LORAS
VENV="$ENVS/regen2k"
OWN_PY=${LFSTUDIO_H3TURBO_PYTHON:-}
[[ "$OWN_PY" == "$VENV/bin/python" ]] && OWN_PY=
PY=${OWN_PY:-$VENV/bin/python}
MARK="$VENV/lf-studio-install.txt"                     # 65-regen2k.sh's: "done <lock sha>" once it is complete
LOCK="$INSTALL_DIR/freeze/regen2k-lock.txt"
JOB=${LFSTUDIO_H3TURBO_SCRIPT:-$REPO/jobs/h3turbo_job.py}
LOCK_SHA=$(sha256sum "$LOCK" | cut -d' ' -f1) || die "cannot read $LOCK"

# Why the base model can't serve LightX2V Turbo (nothing when it can): names only, no weight is read.
base_problem() {
    local part
    [[ -d "$BASE" ]] || { echo "no base model at $BASE (LFSTUDIO_H3TURBO_BASE): run install/20-weights.sh first"; return; }
    for part in transformer vae audio_vae; do
        [[ -f "$BASE/$part/config.json" ]] || { echo "the base model $BASE has no $part/config.json"; return; }
        if ! compgen -G "$BASE/$part/*.safetensors" >/dev/null; then
            if [[ "$part" == vae ]]; then
                echo "the base model $BASE has no VAE weights in vae/ (a t2va-only download has vae/config.json only): run install/20-weights.sh with fl2va in LFSTUDIO_TASKS"
            else
                echo "the base model $BASE has no weights in $part/: run install/20-weights.sh"
            fi
            return
        fi
    done
    for part in scheduler audio_scheduler; do
        [[ -f "$BASE/$part/scheduler_config.json" ]] || { echo "the base model $BASE has no $part/scheduler_config.json"; return; }
    done
}

step "base model $BASE (LFSTUDIO_H3TURBO_BASE)"
problem=$(base_problem)
if [[ -n "$problem" ]]; then
    [[ "$DRY_RUN" == 1 ]] || die "$problem"
    warn "$problem (dry run: going on)"
else
    say "transformer, vae, audio_vae with their weights, and the schedulers: there"
fi

step "Python: $PY"
if [[ -n "$OWN_PY" ]]; then
    [[ -x "$OWN_PY" || "$DRY_RUN" == 1 ]] || die "LFSTUDIO_H3TURBO_PYTHON=$OWN_PY is not an executable Python"
    say "LFSTUDIO_H3TURBO_PYTHON: an environment of your own, used as it is (step 4 checks it)"
elif [[ -f "$MARK" && "$(cat "$MARK")" == "done $LOCK_SHA" ]]; then
    say "the Upscale's environment $VENV, done with install/freeze/regen2k-lock.txt: used as it is"
else
    say "the Upscale's environment $VENV is not there yet (or not done with this lock): 65-regen2k.sh --env-only builds it"
    run env FORCE=0 bash "$INSTALL_DIR/65-regen2k.sh" --env-only ||
        die "65-regen2k.sh --env-only did not finish (above); run this again to go on"
fi

step "LightX2V's turbo LoRA files -> $LORAS (lightx2v/Minimax-h3-Turbo at 30f485ed)"
run "$PY" -I -B "$INSTALL_DIR/h3turbo_weights.py" --loras "$LORAS" ||
    die "the LoRA files are not complete (above); run this again to go on"

step "check: $JOB --check-install (its packages, the base model and the LoRA files; no GPU)"
if [[ ! -f "$JOB" ]]; then
    [[ "$DRY_RUN" == 1 ]] || die "no $JOB: this copy of LF Studio has no LightX2V Turbo job (update it: git pull)"
    warn "no $JOB (dry run: going on)"
fi
run env CUDA_VISIBLE_DEVICES= LFSTUDIO_H3TURBO_BASE="$BASE" LFSTUDIO_H3TURBO_LORAS="$LORAS" \
    LFSTUDIO_H3TURBO_PYTHON="$PY" "$PY" -I -B "$JOB" --check-install ||
    die "LightX2V Turbo's files or environment are not complete (above)"
say "done. To turn the experimental LightX2V Turbo on, set LFSTUDIO_MODELS=sol-h3,lightx2v-turbo in $ENV_FILE and restart the studio (bin/lf-studio --check checks it first)"
