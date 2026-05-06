#!/bin/sh

stage=${1:-Stage1}
env="MultipleCombat"
scenario="2v2/Research/${stage}"
algo="mappo"
exp="attention_progressive_${stage}"
seed=${SEED:-1}
model_dir=${MODEL_DIR:-}

echo "env=${env} scenario=${scenario} algo=${algo} exp=${exp} seed=${seed}"

cmd="python train/train_jsbsim.py \
    --env-name ${env} --algorithm-name ${algo} --scenario-name ${scenario} --experiment-name ${exp} \
    --seed ${seed} --n-training-threads 1 --n-rollout-threads 16 --cuda --log-interval 1 --save-interval 1 \
    --num-mini-batch 4 --buffer-size 512 --num-env-steps 2e7 \
    --lr 3e-4 --gamma 0.99 --ppo-epoch 4 --max-grad-norm 2 --entropy-coef 1e-3 \
    --hidden-size '128 128' --act-hidden-size '128 128' --recurrent-hidden-size 128 --recurrent-hidden-layers 1 --data-chunk-length 8 \
    --use-attention-policy --attention-embed-dim 128 --attention-num-heads 4 --attention-sparse-topk 2 \
    --use-progressive-action-discretization --progressive-action-strides '10 10 10 7|5 5 5 4|2 2 2 2|1 1 1 1' \
    --progressive-stage-boundaries '0.0 0.35 0.7 0.9' --progress-bar-width 84 \
    --use-selfplay --selfplay-algorithm 'fsp' --n-choose-opponents 1 \
    --use-eval --n-eval-rollout-threads 1 --eval-interval 5 --eval-episodes 8"

if [ -n "${model_dir}" ]; then
    cmd="${cmd} --model-dir ${model_dir}"
fi

eval "${cmd}"
