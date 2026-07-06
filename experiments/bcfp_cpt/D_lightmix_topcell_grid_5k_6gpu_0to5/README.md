# D Light-Mix Topcell Grid

Purpose: test a light data mixture instead of pure top-cell sampling.

Runs:

- `D_mix90_topcell10_screenlike_5k_6gpu_0to5_nofused_clip0_ckptfinal`
- `D_mix80_topcell20_screenlike_5k_6gpu_0to5_nofused_clip0_ckptfinal`

Both runs use D EMA-bias routing, 6 GPUs (`0,1,2,3,4,5`), 5k training steps, one final checkpoint, and fixed natural held-out eval on the expanded FineWeb held-out shards.

Start from a normal GPU-visible terminal:

```bash
cd /data1/pengfeigao/BP-MoE
tmux new -s d_lightmix_topcell_5k
bash experiments/bcfp_cpt/D_lightmix_topcell_grid_5k_6gpu_0to5/queue.sh
```

Before starting this queue, make sure no existing run is using GPUs `0-5`.
At the time this script was created, `D_mix90_bal10_screenlike_5k_4gpu_2to5_nofused_clip0_ckptfinal`
was still advancing and using GPUs `2,3,4,5`.

Monitor:

```bash
tail -f experiments/bcfp_cpt/D_lightmix_topcell_grid_5k_6gpu_0to5/queue.log
tail -f experiments/bcfp_cpt/D_mix90_topcell10_screenlike_5k_6gpu_0to5_nofused_clip0_ckptfinal/train.log
```

Primary decision metrics:

- fixed natural held-out BPB from `heldout_eval_expanded_100k_8000batches_gpu0/results.json`
- train pair byte distribution from `metrics.jsonl`
- `moe/pair_max_byte_fraction_mean`, `moe/pair_min_byte_fraction_mean`, `moe/pair_load_cv_mean`, `moe/pair_dead_count_mean`

Do not promote to 10k unless held-out BPB is close to the C/D natural 10k reference and pair load is materially better than natural D.
