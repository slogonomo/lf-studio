#!/usr/bin/env bash
# memguard-run.sh LOG -- CMD...: run CMD in its own process group with output appended to LOG. If MemAvailable
# stays below MEMGUARD_GIB (default 5) for 3 s, the whole group gets SIGTERM, then SIGKILL 10 s later: on GB10's
# unified memory a run that eats the last few GiB can hang the machine. Also records the lowest MemAvailable.
LOG=$1; shift; [[ ${1:-} == -- ]] && shift
[[ -n "$LOG" && $# -gt 0 ]] || { echo "usage: memguard-run.sh LOG -- CMD..." >&2; exit 2; }
FLOOR=${MEMGUARD_GIB:-5}
setsid "$@" >>"$LOG" 2>&1 & P=$!
low=0; min=999
while kill -0 "$P" 2>/dev/null; do
    a=$(awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo); (( a < min )) && min=$a
    if (( a < FLOOR )); then low=$((low + 1)); else low=0; fi
    if (( low >= 3 )); then
        echo "MEMGUARD: MemAvailable ${a} GiB -> killing process group $P" | tee -a "$LOG"
        free -g >>"$LOG"
        command -v nvidia-smi >/dev/null && nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv >>"$LOG"
        kill -TERM -- "-$P"; sleep 10; kill -KILL -- "-$P" 2>/dev/null
    fi
    sleep 1
done
wait "$P"; rc=$?
echo "MEMGUARD: exit=$rc min_mem_avail_gib=$min" | tee -a "$LOG"
exit $rc
