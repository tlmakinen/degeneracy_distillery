# Provenance

## What is here

The **metadata tier** only: `run_record.json`, `config_manifest.json`,
`metrics.csv` per seed, plus `aggregate_summary.{md,json}` where the source
tree had one. That is everything needed to rebuild the table, the success
criteria, the representative expressions, the runtimes and full seed/git
provenance.

Everything large is regenerable intermediate state and stays on scratch:
`*_flatten.npz`, `fishnets_outputs.npz`, `sr_results_*.zip`, Operon
`final_population.csv`. The full three-step trees are 2.4 GB for these four
experiments, of which SIR alone is 2.3 GB, against a `$HOME` quota with ~1.9 GB
of headroom.

`$SCRATCH` = `/data103/makinen/degeneracy_experiments`.

## Sources

| Arm | Experiment | Path under `$SCRATCH` | Layout |
|---|---|---|---|
| three-step | rosenbrock | `follow_up_results/rosenbrock/rebuttal` | `seed_{0..9}` |
| three-step | sir | `follow_up_results/sir/rebuttal` | `seed_{0..9}` |
| three-step | gw_taylorf2 | `follow_up_results/gw_taylorf2/rebuttal` | `seed_{0..9}` |
| three-step | gw_imrphenomd | `follow_up_results/gw_imrphenomd/rebuttal` | `seed_{0..9}` |
| one-step | rosenbrock | `rosenbrock_oneshot_scaling_N500/oneshot/d2` | `trial_{0..9}`, renamed to `seed_*` on copy |
| one-step | sir | `oneshot_rebuttal/sir` | `seed_{0..9}` |
| one-step | gw_taylorf2 | `oneshot_rebuttal/gw_taylorf2` | `seed_{0..9}` |
| one-step | gw_imrphenomd | `oneshot_rebuttal/gw_imrphenomd` | `seed_{0..9}` |
| variant | sir, info-floor 0.5 | `oneshot_rebuttal/sir_rerun_mdl` | `seed_{0..9}` |
| variant | gw_imrphenomd, forced `m=2` | `oneshot_rebuttal/gw_imrphenomd_m2_discovery` | `seed_{0..9}` |
| variant | gw_taylorf2, replicate | `oneshot_rebuttal/gw_taylorf2_rerun_mdl` | `seed_{0..9}` |

The one-step Rosenbrock arm is the `d=2` slice of the N=500 scaling sweep, not
a dedicated rebuttal array. Its held-out set is 1000 simulations where the other
arms use 500.

## Commits

- `db4b146` — SIR and GW adapters, registry clauses, `oneshot_sweep.py` record
  fixes (`expression_mdl`, `picks_agree`, `m_fit`, `n_sr_aug`, the `mdl_total`
  correction, `--sr-allowed-symbols`).
- `22d6153` — handoff note `agent_scripts/oneshot_rebuttal_ports.md`.

Records written before `db4b146` have `expression_mdl: null`. The MDL pick was
always computed correctly; it simply never reached disk. That affects the
one-step `sir`, `gw_taylorf2` and `gw_imrphenomd` baseline trees here.

## Slurm jobs

| Job | What |
|---|---|
| 3833366 | one-step SIR array |
| 3833451 | one-step GW TaylorF2 array |
| 3833465 | one-step GW IMRPhenomD array |
| 3833491 | GW IMRPhenomD forced `m=2` array |
| 3833571 | SIR rerun at info-floor 0.5 |
| 3834268 | GW TaylorF2 replicate, identical config on current code |
| 3834298 | requeue of replicate seed 1; task `3834268_1` hung on node h13 with `do_ypcall: clnt_call: RPC: Timed out` before reaching the screen stage |
| 3833384-87 | step-count diagnostic that set `--probe-steps 2000 --ensemble-steps 2000 --frozen-steps 1500` |

Smoke and cancelled jobs: 3833363, 3833380, 3833462 (smokes); 3833444 and
3833489 (cancelled).

## Not copied

`qm7b`, `kolmogorov`, `kuramoto` and `rayleigh_benard` were in the three-step
rebuttal but have no one-step adapter. Their records remain under
`$SCRATCH/follow_up_results/` and `$SCRATCH/rebuttal_discovery/rayleigh_benard/`.
`follow_up_results/ising/rebuttal` has no `run_record.json` files at all.
