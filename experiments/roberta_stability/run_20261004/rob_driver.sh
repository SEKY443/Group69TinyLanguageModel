#!/usr/bin/env bash
# Runs the RoBERTa test one seed per `colab exec`, downloading each seed's result and logs at once (WSL).
W=/mnt/c/Users/qpj3/AppData/Local/Temp/claude/D--Alarms-semester-2-NLP/1725a5bd-f587-4053-a9bf-30e08c280292/scratchpad/colab
C=$W/c.sh
S=group69r3
R=$W/roberta_results
mkdir -p $R
echo "$(date +%H:%M) setup"
$C exec -s $S -f $W/rob_setup.py --timeout 900 2>&1 | tail -3
for spec in "control_lr2e-5 2e-5" "candidate_lr1e-5 1e-5"; do
  set -- $spec; ARM=$1; LR=$2
  for SEED in 42 43 44; do
    [ -f $R/${ARM}_seed$SEED.json ] && { echo "skip $ARM $SEED"; continue; }
    echo "$(date +%H:%M) start $ARM seed $SEED"
    $C exec -s $S -f $W/rob_seed.py --timeout 3600 --env ARM=$ARM --env LR=$LR --env SEED=$SEED > $R/${ARM}_seed$SEED.console 2>&1
    $C download -s $S /content/roberta_exp/$ARM/result_seed$SEED.json $R/${ARM}_seed$SEED.json > /dev/null 2>&1
    for f in $($C ls -s $S /content/roberta_exp/$ARM/logs 2>/dev/null); do
      $C download -s $S /content/roberta_exp/$ARM/logs/$f $R/${ARM}_$f > /dev/null 2>&1
    done
    if [ -f $R/${ARM}_seed$SEED.json ]; then echo "$(date +%H:%M) RESULT $(cat $R/${ARM}_seed$SEED.json | tr -d '\n ')"; else echo "$(date +%H:%M) FAILED $ARM seed $SEED (see console)"; fi
  done
done
echo "$(date +%H:%M) ALLDONE"
