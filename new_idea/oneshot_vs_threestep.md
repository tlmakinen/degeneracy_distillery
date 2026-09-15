# One-step map versus Fishnets-then-flatten

Default budget on a CPU: 2000 train pairs, 1000 test pairs.
Script: `new_idea/compare_rosen_ab.py`.
The score is cubic \(R^2\) of the true coordinates against the top-\(\hat r\) axes of \(\eta\).
Also record whether \(\hat r\) matches the true rank.
No symbolic regression.

## One-step loss

Train a data-independent map \(\eta(\theta)\) and an estimator \(\hat\eta(x)\) with

\[
\mathcal{L}
=\mathbb{E}_{\theta,x}
\Bigl[
\tfrac12\bigl\|\eta(\theta)-\hat\eta(x)\bigr\|^2
-\tfrac12\log\det(JJ^\top)
\Bigr].
\]

For fixed \(\eta\), the unique minimizer is \(\hat\eta(x)=\mathbb{E}[\eta(\theta)\mid x]\).
The residual asks \(\hat\eta\) to extract from \(x\) whatever \(\eta(\theta)\) encodes.
The volume term wants a large Gram volume.
Stretch only a direction that \(x\) can predict. Stretch of a silent direction raises the residual.

The split at a stationary point is:

- \(x\) informs the axis. Posterior variance of \(\eta_i\) is 1. \(\hat\eta_i\) tracks that combination.
- \(x\) is silent. \(\hat\eta_i\) is constant. Prior variance of \(\eta_i\) is 1 (\(\lambda_i=1\)).

Information here means: can \(\hat\eta(x)\) beat the prior mean at predicting \(\eta(\theta)\)?
The loss does not build a Fisher in \(\theta\).

## Variational bound

For any normalised \(q(\theta\mid x)\),

\[
\mathbb{E}_{\theta,x}\bigl[-\log q(\theta\mid x)\bigr]
=\mathbb{E}_x\bigl[\mathrm{KL}(p(\theta\mid x)\,\|\,q(\theta\mid x))\bigr]
+H(\theta\mid x).
\]

The entropy term does not depend on \(q\).
The left-hand side is an upper bound on \(\mathbb{E}[-\log p(\theta\mid x)]\).
Equality holds iff \(q=p\).

Fishnets are the case \(\eta=\mathrm{id}\):

\[
q(\theta\mid x)=\mathcal{N}\bigl(\hat\theta(x),F(x)^{-1}\bigr).
\]

One-step gives the same object when \(q\) is normalised.

**Square diffeomorphism** (\(m=d\), \(\eta\) invertible):

\[
q(\theta\mid x)=\mathcal{N}\bigl(\eta(\theta);\hat\eta(x),I\bigr)\,|\det J|.
\]

The one-step loss is \(-\log q\).

**Hard cut** (active set \(A\), complement at the prior):

\[
q(\theta\mid x)
=\mathcal{N}\bigl(\eta(\theta_A);\hat\eta(x),I\bigr)\,|\det J_A|\;\pi_B(\theta_B).
\]

The term \((d-m)\log w\) in the script is \(-\log\pi_B\).
This is again a proper variational NLL.

**Rectangular \(m<d\)** with no fibre measure is not an ELBO.
Coarea leaves a Hausdorff factor \(\mathcal{H}^{d-m}(\eta^{-1}(u))\) that the loss drops.
The integral is not 1, so the score is not \(-\log q(\theta\mid x)\).

You still have a bound on the pushforward \(u=\eta(\theta)\):

\[
\mathbb{E}\bigl[\tfrac12\|\eta-\hat\eta\|^2\bigr]+\tfrac m2\log 2\pi
\ge H(u\mid x).
\]

That bound lives on \(\mathbb{R}^m\), not on \(\theta\).
The volume term is not part of it.

**Hybrid.** Use rectangular geometry to read \(\hat r\) and the active set.
Then retrain the hard-cut \(q\) if you need \(\mathbb{E}[-\log q]\ge H(\theta\mid x)\).
Do not hard-cut when every coordinate enters and only \(m\) combinations carry information.

## Three-step failure on a scalar potential

Observe only \(x\sim\mathcal{N}(f(\theta),\sigma^2)\).
The likelihood Fisher is rank 1:

\[
F_{\mathrm{like}}(\theta)
=\frac{1}{\sigma^2}\nabla f(\theta)\,\nabla f(\theta)^\top.
\]

The posterior lives on the level set \(f(\theta)\approx x\).
That set is a curved fibre. It is not an ellipsoid in \(\theta\).

Fishnets fit a Gaussian in \(\theta\).
That family cannot sit on the fibre.
A Cholesky floor also keeps every eigenvalue away from zero.
The fit returns a tightened prior: all \(\lambda_F=O(1)\), no distinguished axis.

Flatten then solves \(J^{-\top}FJ^{-1}\approx I\).
If \(F\approx cI\), the map is a scaled rotation.
Then \(\eta\) is a linear remix of \(\theta\) and is almost uncorrelated with \(f\).

One-step never builds that Gaussian.
For \(m=1\), \(\eta(\theta)\) lines up with \(f(\theta)\) and \(\hat\eta(x)\) reads \(x\).
The fibre is \(\ker J\). The loss does not model it.

This is the heater failure in a different \(f\).
Unused coordinates at \(d=3,4\) did not trigger it.
The trigger is: all \(\theta_j\) enter, rank \(r\ll d\).

## Problems

| name | what you observe | true rank | one-step mode |
|---|---|---|---|
| `banana2`, `banana3`, `banana4` | \(\mu=(\theta_0,\theta_1-\theta_0^2)\), leftover \(\theta\) unused | 2 | hard cut |
| `uncoupled4` | two independent bananas | 4 | hard cut |
| `coupled3`, `coupled4` | every \(\theta_i\) and every link \(\theta_{i+1}-\theta_i^2\) | \(N\) | hard cut |
| `scalar3`, `scalar4` | coupled potential \(f(\theta)\) only (\(b=1\)) | 1 | rectangular |

## Results (default CPU budget)

### Banana plus unused coordinates

| \(d\) | unused | oneshot \(R^2\) | three-step \(R^2\) | three-step \(F\) eigs | three-step time |
|---|---|---|---|---|---|
| 2 | 0 | 0.9996, 0.9993 | 1.0000, 0.9994 | 265, 17 | 34 s |
| 3 | 1 | 0.9997, 0.9989 | 0.9999, 0.9983 | 160, 4.0, 0.35 | 36 s |
| 4 | 2 | 0.9998, 0.9982 | 1.0000, 0.9999 | 166, 8.6, 0.35, 0.32 | 38 s |

Both methods report \(\hat r=2\). This is a tie.

### Uncoupled \(N=4\) and coupled chains

| problem | oneshot \(\hat r\) / \(R^2_{\min}\) | three-step \(\hat r\) / \(R^2_{\min}\) |
|---|---|---|
| uncoupled4 | 4 / 0.9956 | 2 / 0.9991 |
| coupled3 | 3 / 0.9915 | 3 / 0.9996 |
| coupled4 | 4 / 0.9934 | 4 / 0.9994 |

Both recover the coordinates.
Three-step miscounts rank on `uncoupled4`.
The two smaller banana eigenvalues sit at 0.90 and 0.60, under the relative floor \(10^{-2}\).

### Scalar coupled potential

| problem | oneshot \(\hat r\) / \(R^2\) | three-step \(\hat r\) / \(R^2\) | three-step \(F\) eigs |
|---|---|---|---|
| scalar3 | 1 / 0.9988 | 3 / 0.2094 | 0.60, 0.51, 0.39 |
| scalar4 | 1 / 0.9974 | 4 / 0.0346 | 0.49, 0.47, 0.43, 0.34 |

One-step keeps all \(\theta\) in \(\eta\), reads one large \(\lambda\), and recovers \(f\).
Fishnets report an almost isotropic \(F\). Flatten has no axis to straighten.

## How to run

```bash
python new_idea/compare_rosen_ab.py --dims 2 3 4
python new_idea/compare_rosen_ab.py --problems uncoupled4 coupled3 coupled4
python new_idea/compare_rosen_ab.py --problems scalar3 scalar4
python new_idea/compare_rosen_ab.py --full --problems scalar3 scalar4
```

`--quick` is too starved to trust \(\hat r\).
One GPU helps the three-step arm at `--full`. It does not change the scalar conclusion at the default budget.
