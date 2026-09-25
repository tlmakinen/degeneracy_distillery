# gw_imrphenomd: seed-sweep summary

Across 10 independent end-to-end trials (500 training simulations each), the pipeline recovered the expected coordinate in 7/10 trials. Symbolic-regression augmentation used 2000 inexpensive evaluations of the learned coordinate map and required no additional simulator calls.

| Metric | Value |
|---|---|
| Recovery count | 7/10 |
| Training simulations | 500 |
| Held-out evaluation simulations | 500 |
| Augmented coordinate evaluations | 2000 |
| Physics alignment (median [IQR]) | 0.997 [0.986, 0.999] |
| Held-out geometric loss, symbolic (median [IQR]) | 1.887 [1.218, 2.240] |
| Expression complexity (median [IQR]) | 24.000 [19.000, 26.000] |
| Expression MDL (median [IQR]) | 564361.906 [386468.641, 665230.312] |

Representative trial (median-success, `gw_imrphenomd_seed2`):
- `0.005*m2 - 0.025`
- `1.605*(0.006*m1 + 0.006*m2 + 0.491)**1.896 - 0.541`

## Threshold misses (3) -- ran cleanly, did not clear the discovery criteria

| run_id | seed | physics_alignment |
|---|---|---|
| gw_imrphenomd_seed0 | 0 | 1.000 |
| gw_imrphenomd_seed1 | 1 | 0.994 |
| gw_imrphenomd_seed7 | 7 | 0.713 |
