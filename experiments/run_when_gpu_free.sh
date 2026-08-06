#!/usr/bin/env bash
# Wait for another job to release the GPU, then run the two outstanding arms.
#
# Written for an unattended overnight run. The user's CIFAR-100 job (allocation_seeds.py)
# holds the card; starting alongside it is what stalled a run on the previous machine,
# so this waits rather than competes.
#
# Both arms use the checkpointing runner, so a crash costs at most one seed and a restart
# resumes. Everything is logged; nothing is committed automatically.

set -u
cd "$(dirname "$0")/.."

LOG="results/overnight.log"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

say "watcher started; waiting for the GPU to free up"

# --- wait for the foreign job to exit -------------------------------------------------
# Match on the command line rather than a PID: a PID can be reused, and the job may be
# restarted under a new one. `pgrep` does not exist in this Git-Bash environment, so ask
# Windows directly -- an earlier version used pgrep, silently found nothing, and would
# have started immediately on top of the running job.
foreign_running() {
  powershell.exe -NoProfile -Command \
    "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*allocation_seeds.py*' }) { 'YES' } else { 'NO' }" \
    2>/dev/null | tr -d '\r\n '
}

# Fail closed: if the check itself breaks we must not assume the card is free.
probe=$(foreign_running)
if [ "$probe" != "YES" ] && [ "$probe" != "NO" ]; then
  say "ABORT: cannot determine whether the other job is running (probe returned '$probe')"
  exit 1
fi
say "foreign job detected: $probe"

while [ "$(foreign_running)" = "YES" ]; do
  sleep 300
done
say "allocation_seeds.py has exited"

# Give the driver a moment to release VRAM, then confirm the card is actually free
# rather than trusting the process check alone.
sleep 60
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
say "GPU memory in use: ${USED} MiB"
if [ "${USED:-99999}" -gt 6000 ]; then
  say "ABORT: more than 6 GB still held; not starting so the other job is not disturbed"
  exit 1
fi

# --- arm 1: TabPFN, local weights, three seeds ----------------------------------------
say "arm 1/2: TabPFN local, 3 seeds (referee condition 2)"
python experiments/run_external_multiseed.py \
  --config configs/experiments/shift_stress_external_2axis_multiseed_tabpfn_local.yaml \
  >>"$LOG" 2>&1
say "arm 1 finished with exit code $?"

# --- arm 2: strong baselines, three seeds ---------------------------------------------
# Runs regardless of arm 1's exit code: the two are independent, and a TabPFN failure
# should not also cost the baseline arm a night.
say "arm 2/2: strong baselines (tuned x3 + ensemble), 3 seeds (referee condition 1)"
python experiments/run_external_multiseed.py \
  --config configs/experiments/shift_stress_external_2axis_strong_baselines_multiseed.yaml \
  >>"$LOG" 2>&1
say "arm 2 finished with exit code $?"

say "watcher done"
