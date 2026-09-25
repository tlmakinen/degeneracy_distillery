# rosenbrock: seed-sweep summary

Across 10 independent end-to-end trials (500 training simulations each), the pipeline recovered the expected coordinate in 10/10 trials. Symbolic-regression augmentation used 2000 inexpensive evaluations of the learned coordinate map and required no additional simulator calls.

| Metric | Value |
|---|---|
| Recovery count | 10/10 |
| Training simulations | 500 |
| Held-out evaluation simulations | 500 |
| Augmented coordinate evaluations | 2000 |
| Physics alignment (median [IQR]) | 0.929 [0.812, 0.963] |
| Held-out geometric loss, symbolic (median [IQR]) | 0.483 [0.444, 0.535] |
| Expression complexity (median [IQR]) | 28.000 [24.000, 28.000] |
| Expression MDL (median [IQR]) | 324.176 [273.398, 395.663] |

Representative trial (median-success, `rosenbrock_seed5`):
- `0.473*theta1**2 + 0.331*theta1 - 0.622*theta2 + 4.585`
- `4.564 - 1.882*theta1`
