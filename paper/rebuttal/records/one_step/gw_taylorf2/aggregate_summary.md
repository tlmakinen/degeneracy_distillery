# gw_taylorf2: seed-sweep summary

Across 10 independent end-to-end trials (500 training simulations each), the pipeline recovered the expected coordinate in 8/10 trials. Symbolic-regression augmentation used 2000 inexpensive evaluations of the learned coordinate map and required no additional simulator calls.

| Metric | Value |
|---|---|
| Recovery count | 8/10 |
| Training simulations | 500 |
| Held-out evaluation simulations | 500 |
| Augmented coordinate evaluations | 2000 |
| Physics alignment (median [IQR]) | 0.983 [0.980, 0.987] |
| Held-out geometric loss, symbolic (median [IQR]) | n/a |
| Expression complexity (median [IQR]) | 8.500 [5.000, nan] |
| Expression MDL (median [IQR]) | n/a |

Representative trial (median-success, `gw_taylorf2_oneshot_d2_trial9`):

## Threshold misses (2) -- ran cleanly, did not clear the discovery criteria

| run_id | seed | physics_alignment |
|---|---|---|
| gw_taylorf2_oneshot_d2_trial2 | 15900 | 0.690 |
| gw_taylorf2_oneshot_d2_trial7 | 55495 | 0.681 |
