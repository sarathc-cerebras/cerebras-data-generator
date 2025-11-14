
python packages/synth-gen-client/synth_gen_client/main.py \
    --dataset-repo "BAAI/Infinity-Instruct" \
    --dataset-subset "7M_core" \
    --n-samples 1000 \
    --max-concurrency 100 \
    --model "qwen-3-32b" \
    --output-file results_1000.jsonl