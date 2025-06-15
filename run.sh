python scripts/run_inference.py \
    --model_name OpenGVLab/InternVL3-8B-Instruct \
    --dataset operator_induction \
    --data_dir ./data \
    --generation_mode free
    --n_shot 0 1 2 4 8 \
    --debug \
    --debug_samples 5