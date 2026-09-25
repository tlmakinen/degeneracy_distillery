# gw_taylorf2: seed-sweep summary

Across 10 independent end-to-end trials (500 training simulations each), the pipeline recovered the expected coordinate in 10/10 trials. Symbolic-regression augmentation used 2000 inexpensive evaluations of the learned coordinate map and required no additional simulator calls.

| Metric | Value |
|---|---|
| Recovery count | 10/10 |
| Training simulations | 500 |
| Held-out evaluation simulations | 500 |
| Augmented coordinate evaluations | 2000 |
| Physics alignment (median [IQR]) | 0.972 [0.966, 0.981] |
| Held-out geometric loss, symbolic (median [IQR]) | 4.344 [2.637, 6.077] |
| Expression complexity (median [IQR]) | 24.000 [22.000, 26.000] |
| Expression MDL (median [IQR]) | 379371.172 [295586.672, 511224.062] |

Representative trial (median-success, `gw_taylorf2_seed0`):
- `0.005*m1 - 0.002*m2 + 0.057`
- `0.010416*m1 + 0.010428*m2 + 1.821404`
