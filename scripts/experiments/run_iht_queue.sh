#!/bin/bash
# Experiment 23 queue (iht_dict.py): starlet vs learnlet dictionaries in iterative hard thresholding, compact sim +
# null test (seed 7), 10,000 iterations (sim20k: 20,000, convergence checks). SET=main: starlet / ImageNet learnlet;
# SET=ext: the ImageNet learnlet with scale-3 filters extrapolated (learnlet_ext) and the FIRE learnlets, per-scale
# (learnlet_fire) and shared across scales (learnlet_fire_shared4 / 6, train_learnlet_fire.py --shared); SET=n600: the
# FIRE learnlets trained on 600 px planes with 7 learnt scales, per scale and shared (learnlet_fire_n600[_shared]);
# SET=real: the real NGC 3110 compact data (real/), each arm at its best simulation setting; SET=aug: the scale-augmented ones (learnlet_fire_aug[_shared]) and learnlet_ext at k = 8, 10 (+ a 20k check at k = 6).
# Workers share one job list: each claims the next job not done or claimed whose dictionary exists (a training still
# running: its jobs wait). A GPU listed as G:FILE starts only once FILE exists (e.g. the .npz of a training on it).
#   SET=ext nohup bash scripts/experiments/run_iht_queue.sh 2 3 0:dictionaries/x.npz > logs/23_iht/queue3.log 2>&1 &
cd "$(dirname "$0")/../.."
PY=/scratch/alahiry/conda/envs/deconvolver2d1d/bin/python
OUT=data/experiments/23_iht; LOGS=logs/23_iht; CLAIM=$LOGS/claims; mkdir -p $OUT/sim $OUT/null $OUT/sim20k $OUT/real $LOGS $CLAIM
jobs=()
# rule pdtau: the PIPELINE's own method (run_model.py --method pdtau: primal-dual, soft thresholding + reweighting,
# 20,000 iterations) with the 2D dictionary swapped (DICT_2D; "name:N" = top N filters per scale, file tag nameN)
# real NGC 3110 only (no simulations): the pipeline's pdtau with each dictionary over k, scored against the joint
# compact + extended CLEAN cubes (iht_real.py). An analysis prior over the starlet planes AND the filter planes
# penalises more at the same k, so the learnlets' k range sits lower (exp 22: best at k ~ 1-2).
if [ "${SET:-main}" = pipe_heavy ]; then
  for k in 1.5 2 1 3; do for d in learnlet learnlet_ext:32; do jobs+=("$d pdtau k $k real"); done; done
elif [ "$SET" = pipe_heavy2 ]; then                          # bracketing the minima found on 2026-10-23 (all at k = 1)
  for k in 0.75 0.5; do for d in learnlet_ext:32 learnlet; do jobs+=("$d pdtau k $k real"); done; done
elif [ "$SET" = pipe_light2 ]; then
  for k in 0.75 1 0.5 1.5; do for d in learnlet_fire_aug learnlet_fire_aug_shared learnlet_fire_n600_shared; do
    jobs+=("$d pdtau k $k real"); done; done
  for k in 0.75 0.5; do for d in learnlet_fire_n600 learnlet_fire_shared4; do jobs+=("$d pdtau k $k real"); done; done
  for k in 6 8 10; do jobs+=("starlet pdtau k $k real"); done
elif [ "$SET" = pipe_deconv2 ]; then                        # v2 (2026-10-24): proper learning rate; trained through pdtau / FISTA
  for k in 1 0.75 1.5 2; do for d in learnlet_deconv_pd learnlet_deconv2; do jobs+=("$d pdtau k $k real"); done; done
  for k in 1 2; do jobs+=("learnlet_deconv2 fista k $k real"); done
elif [ "$SET" = pipe_deconv ]; then                         # the learnlets trained FOR the deconvolution (train_learnlet_deconv.py)
  for k in 1 0.75 1.5 2; do for r in pdtau fista; do for d in learnlet_deconv learnlet_deconv_shared; do
    jobs+=("$d $r k $k real"); done; done; done
elif [ "$SET" = pipe_syn ]; then                            # pdtau with the synthesis W^-1 instead of W^T (pdsyn), then FISTA
  for k in 4 5 3 6 8; do jobs+=("starlet pdsyn k $k real"); done
  for k in 1.5 1 2 3 4; do jobs+=("learnlet_fire_n600 pdsyn k $k real"); done
  for k in 1.5 1 2 3 4; do for d in learnlet_ext:32 learnlet; do jobs+=("$d pdsyn k $k real"); done; done
  for k in 1.5 1 2; do for d in learnlet_fire_n600 learnlet_ext:32 learnlet; do jobs+=("$d fista k $k real"); done; done
  for k in 4 5 3; do jobs+=("starlet fista k $k real"); done
elif [ "$SET" = pipe_fista ]; then                          # the pipeline's FISTA: soft thresholding + reweighting, image
  # rebuilt through W^-1 every step, so a learnlet's SYNTHESIS filters are used too (pdtau only uses W and W^T)
  for k in 1.5 1 2; do for d in learnlet_fire_n600 learnlet_ext:32 learnlet; do jobs+=("$d fista k $k real"); done; done
  for k in 4 5 3; do jobs+=("starlet fista k $k real"); done
elif [ "$SET" = pipe_learnt ]; then                          # pdtau with the learnt per-scale thresholds x s
  for s in 0.5 0.3 0.75; do for d in learnlet_fire_n600 learnlet_fire_n600_shared; do jobs+=("$d pdtaul s $s real"); done; done
elif [ "$SET" = pipe_light ]; then
  for k in 1.5 2 1 3; do for d in learnlet_fire_n600 learnlet_fire_shared4; do jobs+=("$d pdtau k $k real"); done; done
  for k in 4 3 2; do jobs+=("starlet pdtau k $k real"); done
elif [ "$SET" = main ]; then
  for d in learnlet starlet; do
    for k in 2 3 4 5 6 8; do for m in sim null; do jobs+=("$d noise k $k $m"); done; done
    for s in 0.5 0.75 1 1.5 2 3; do for m in sim null; do jobs+=("$d learnt s $s $m"); done; done
  done
  jobs+=("learnlet noise k 5 sim20k" "learnlet learnt s 1.5 sim20k")
elif [ "$SET" = real ]; then                                 # real NGC 3110 compact data, each arm at its best sim setting
  for j in "learnlet_ext noise k 6" "learnlet noise k 6" "learnlet learnt s 1.5" "starlet noise k 2" "starlet noise k 4"            "starlet learnt s 0.5" "learnlet_fire learnt s 2" "learnlet_fire_n600 noise k 6" "learnlet_fire_shared4 noise k 6"            "learnlet_fire noise k 5" "learnlet_fire_shared6 noise k 5"; do jobs+=("$j real"); done
elif [ "$SET" = aug ]; then                                  # scale-augmented FIRE learnlets; ImageNet extrapolated, larger k
  for d in learnlet_fire_aug learnlet_fire_aug_shared; do
    for k in 4 5 6; do for m in sim null; do jobs+=("$d noise k $k $m"); done; done
    for s in 1.5 2; do for m in sim null; do jobs+=("$d learnt s $s $m"); done; done
  done
  for k in 8 10; do for m in sim null; do jobs+=("learnlet_ext noise k $k $m"); done; done
  jobs+=("learnlet_ext noise k 6 sim20k")
elif [ "$SET" = n600 ]; then                                 # the 600 px FIRE learnlets, per scale and shared
  for d in learnlet_fire_n600 learnlet_fire_n600_shared; do
    for k in 4 5 6; do for m in sim null; do jobs+=("$d noise k $k $m"); done; done
    for s in 1.5 2; do for m in sim null; do jobs+=("$d learnt s $s $m"); done; done
  done
else
  for d in learnlet_ext learnlet_fire_shared6 learnlet_fire_shared4 learnlet_fire; do
    for k in 4 5 6; do for m in sim null; do jobs+=("$d noise k $k $m"); done; done
  done
  for d in learnlet_fire_shared6 learnlet_fire; do
    for s in 1.5 2; do for m in sim null; do jobs+=("$d learnt s $s $m"); done; done
  done
fi

worker() {   # $1 = GPU[:file to wait for]
  local g=${1%%:*} need=${1#*:}
  [ "$need" != "$1" ] && until [ -f "$need" ]; do sleep 60; done
  while :; do
    local left=0 ran=0
    for job in "${jobs[@]}"; do
      read d thr p v m <<< "$job"
      local dt=${d//:/}                                       # file tag: learnlet_ext:32 -> learnlet_ext32
      local name=${dt}_${thr}_${p}${v} out=$OUT/$m/${dt}_${thr}_${p}${v}.npy
      [ -f "$out" ] && continue
      left=1
      [ "$d" != starlet ] && [ ! -f dictionaries/${d%%:*}.npz ] && continue
      mkdir "$CLAIM/${m}_$name" 2>/dev/null || continue
      local flag n
      flag=$(case $m in null) echo "--null 7";; real) echo "";; *) echo --sim;; esac); n=$([ $m = sim20k ] && echo 20000 || echo 10000)
      echo "$(TZ=Europe/Athens date '+%F %T') GPU $g start $m $name"
      if [ "$thr" = pdtaul ]; then                            # pdtau, lambda = the learnt thresholds x s (coarse: 1.5 x noise)
        CUDA_VISIBLE_DEVICES=$g DICT_2D=$d $PY scripts/experiments/run_model.py --method pdtau --lam-profile learnlet \
          --lam-scale $v --k 1.5 $flag --n-iter 20000 --out $out > $LOGS/${m}_$name.log 2>&1
      elif [ "$thr" = pdsyn ]; then                           # pdtau, dual brought back by W^-1 (common.solve pdtau-syn)
        CUDA_VISIBLE_DEVICES=$g DICT_2D=$d $PY scripts/experiments/run_model.py --method pdtau-syn --k $v $flag \
          --n-iter 20000 --out $out > $LOGS/${m}_$name.log 2>&1
      elif [ "$thr" = fista ]; then
        CUDA_VISIBLE_DEVICES=$g DICT_2D=$d $PY scripts/experiments/run_model.py --method fista --k $v $flag \
          --n-iter 20000 --out $out > $LOGS/${m}_$name.log 2>&1
      elif [ "$thr" = pdtau ]; then
        CUDA_VISIBLE_DEVICES=$g DICT_2D=$d $PY scripts/experiments/run_model.py --method pdtau --k $v $flag \
          --n-iter 20000 --out $out > $LOGS/${m}_$name.log 2>&1
      else
        CUDA_VISIBLE_DEVICES=$g $PY scripts/experiments/iht_dict.py --dict $d --thr $thr --$p $v $flag \
          --n-iter $n --out $out > $LOGS/${m}_$name.log 2>&1
      fi
      local rc=$?
      echo "$(TZ=Europe/Athens date '+%F %T') GPU $g done $m $name ($rc)"
      ran=1; break
    done
    [ $left = 0 ] && return
    if [ $ran = 0 ]; then                                    # nothing claimable: everything left is claimed / waiting
      local open=0
      for job in "${jobs[@]}"; do read d thr p v m <<< "$job"
        d=${d//:/}; [ -f $OUT/$m/${d}_${thr}_${p}${v}.npy ] || [ -d "$CLAIM/${m}_${d}_${thr}_${p}${v}" ] || open=1; done
      [ $open = 0 ] && return
      sleep 60
    fi
  done
}
for g in "$@"; do worker "$g" & done
wait
echo "$(TZ=Europe/Athens date '+%F %T') ALL DONE"
