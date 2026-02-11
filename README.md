# LoSARE

## Pre-training LLaMA (60M~7B) on the C4 dataset

`./pre_training_c4` includes the code for pre-training LLaMA models on the C4 dataset.

### Set up the environment
```bash
cd pre_training_c4
pip install -r requirements.txt
```
Our experiment scripts are validated on Python 3.9 with PyTorch 2.2.2.

### Code Structure
`./pre_training_c4/torchrun_main.py` script is used for pre-training LLaMA models on the C4 dataset. 
`./pre_training_c4/scripts` directory stores the benchmark scripts across different LLaMA model sizes (60M, 130M, 350M, 1B, 7B).

For instance, to pre-train a 60M model on C4 dataset, execute the following command:
```bash
# LLaMA-60M, LoSARE-Adam, 1 A100, 1 Node
torchrun --standalone --nproc_per_node 2 torchrun_main.py \
    --model_config llama_configs/llama_60m.json \
    --lr 0.01 \
    --alpha 0.25 \
    --rank 128 \
    --update_proj_gap 200 \
    --adaptive_update \
    --adapt_lr 0.1 \
    --dynamic_alpha 1.5 \
    --batch_size 128 \
    --total_batch_size 256 \
    --num_training_steps 10000 \
    --warmup_steps 1000 \
    --weight_decay 0 \
    --dtype bfloat16 \
    --eval_every 1000 \
    --optimizer losare_adamw
```
