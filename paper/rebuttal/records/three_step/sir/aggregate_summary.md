# sir: seed-sweep summary

Across 10 independent end-to-end trials (500 training simulations each), the pipeline recovered the expected coordinate in 10/10 trials. Symbolic-regression augmentation used 7554 inexpensive evaluations of the learned coordinate map and required no additional simulator calls.

| Metric | Value |
|---|---|
| Recovery count | 10/10 |
| Training simulations | 500 |
| Held-out evaluation simulations | 500 |
| Augmented coordinate evaluations | 7554 |
| Physics alignment (median [IQR]) | 0.674 [0.633, 0.745] |
| Held-out geometric loss, symbolic (median [IQR]) | 5.743 [4.348, 6.717] |
| Expression complexity (median [IQR]) | 17.000 [14.000, 26.000] |
| Expression MDL (median [IQR]) | 14801.219 [6735.945, 23562.036] |

Representative trial (median-success, `sir_seed2`):
- `1.65*gamma - 0.119*(1.116*beta + 0.886)**2 + 1.0/(0.447*beta + 0.894*gamma + 0.71)**1.43 + 0.655`
- `1.65*gamma + 1.176`
