#!/bin/bash
# Extra seeds for the final TEST evaluation. Resumable: finished rows are skipped.
cd "$(dirname "$0")"
run() { echo "=== $*"; python -m src.train.final_eval "$@" || echo "FAILED: $*"; }
S5=43,44,45,46
S3=43,44
for lv in binary category; do
  for ds in wsn_ds ciciot2023; do
    run --dataset $ds --level $lv --models dt,lgbm,xgb --seeds $S5
    run --dataset $ds --level $lv --models mlp --seeds $S5
    run --dataset $ds --level $lv --models rf,catboost --seeds $S3
    run --dataset $ds --level $lv --models cnn1d --seeds $S3
  done
done
run --dataset ciciot2023 --level attack_type --models dt,lgbm,xgb --seeds $S5
run --dataset ciciot2023 --level attack_type --models mlp --seeds $S5
run --dataset ciciot2023 --level attack_type --models cnn1d --seeds $S3
echo "STAGE 2 DONE"
