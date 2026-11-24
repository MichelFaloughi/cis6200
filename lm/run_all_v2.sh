#!/bin/sh
# Run every phase-2 job (72 jobs, ~50 min on 12 cores), then summarise and plot.
# Usage: sh lm/run_all_v2.sh [parallel_jobs=6]
cd "$(dirname "$0")"
P=${1:-6}
mkdir -p out/v2/logs
{
  for s in conv_T1_acc conv_T1.5_acc; do for i in 0 1 2; do echo "$s $i"; done; done
  for s in conv_T1 conv_T1.5 early_T1; do for i in 0 1 2 3 4; do echo "$s $i"; done; done
  for s in early_T1_acc conv_T1_acc_fc early_T1_acc_fc conv_T1.5_acc_fc \
           conv_T1.5_lam0.3 conv_T1.5_drop1 conv_T1.5_verify0.5 early_T1_lam0.3 early_T1_drop1 early_T1_verify0.5 \
           conv_T1.5_zipf0.7 conv_T1.5_zipf1.3 early_T1_zipf0.7 early_T1_zipf1.3 \
           conv_T1_para conv_T1.5_para early_T1_para; do
    for i in 0 1 2; do echo "$s $i"; done
  done
} | xargs -P "$P" -L 1 sh -c 'THREADS=2 python3 run_gpt_v2.py $0 $1 > out/v2/logs/$0_s$1.log 2>&1 || echo FAILED $0 $1'
python3 summarize_v2.py && python3 plots.py v2
