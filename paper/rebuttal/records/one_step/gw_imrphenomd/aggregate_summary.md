# gw_imrphenomd: seed-sweep summary

Across 10 independent end-to-end trials (500 training simulations each), the pipeline recovered the expected coordinate in 5/10 trials. Symbolic-regression augmentation used 2000 inexpensive evaluations of the learned coordinate map and required no additional simulator calls.

| Metric | Value |
|---|---|
| Recovery count | 5/10 |
| Training simulations | 500 |
| Held-out evaluation simulations | 500 |
| Augmented coordinate evaluations | 2000 |
| Physics alignment (median [IQR]) | 0.991 [0.986, 0.994] |
| Held-out geometric loss, symbolic (median [IQR]) | n/a |
| Expression complexity (median [IQR]) | 17.000 [12.500, 17.500] |
| Expression MDL (median [IQR]) | n/a |

Representative trial (median-success, `gw_imrphenomd_oneshot_d2_trial6`):

## Threshold misses (5) -- ran cleanly, did not clear the discovery criteria

| run_id | seed | physics_alignment |
|---|---|---|
| gw_imrphenomd_oneshot_d2_trial0 | 62 | 0.996 |
| gw_imrphenomd_oneshot_d2_trial1 | 7981 | 0.991 |
| gw_imrphenomd_oneshot_d2_trial2 | 15900 | 0.701 |
| gw_imrphenomd_oneshot_d2_trial5 | 39657 | 1.000 |
| gw_imrphenomd_oneshot_d2_trial8 | 63414 | 0.981 |
