# sir: seed-sweep summary

Across 10 independent end-to-end trials (500 training simulations each), the pipeline recovered the expected coordinate in 10/10 trials. Symbolic-regression augmentation used 10000 inexpensive evaluations of the learned coordinate map and required no additional simulator calls.

| Metric | Value |
|---|---|
| Recovery count | 10/10 |
| Training simulations | 500 |
| Held-out evaluation simulations | 500 |
| Augmented coordinate evaluations | 10000 |
| Physics alignment (median [IQR]) | 0.667 [0.635, 0.783] |
| Held-out geometric loss, symbolic (median [IQR]) | n/a |
| Expression complexity (median [IQR]) | 12.500 [10.000, 15.000] |
| Expression MDL (median [IQR]) | n/a |

Representative trial (median-success, `sir_oneshot_d3_trial8`):
