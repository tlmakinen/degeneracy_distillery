# Procedure description update
_Exported on 9/7/2026 at 10:30:55 GMT+1 from Cursor (3.7.36)_

---

**User**

look at @new_idea/gw_detection_extrinsics.py  and @new_idea/rosenbrock20_oneshot_noscreen.py , as well as @new_idea/main.tex . Please go through the .tex file and update the description of the procedure that we're following. maybe make a new main_mod.tex for comparison.

---

**Cursor**

I'll read the two scripts and `main.tex`, then draft an updated procedure description in `main_mod.tex`.

The scripts use a one-shot geometry objective with all coordinates and a soft Jacobian screen — not the hard active-set screen in `main.tex`. I'll check the screened oneshot for contrast, then write `main_mod.tex`.

Creating `main_mod.tex` with the no-screen geometry objective and one-shot protocol from the scripts.

Fixing one awkward sentence in the Estimation opener.

Wrote [`new_idea/main_mod.tex`](new_idea/main_mod.tex). Theory (model, loss, propositions) is unchanged. The Estimation section now matches the no-screen scripts.

**Main shifts vs `main.tex`**
- Primary objective is the rectangular geometry loss \(\mathcal{L}_{\mathrm{geom}}\) with \(\tfrac12\log\det(JJ^\top)\), all \(d\) inputs, no \((d-m)\log w\).
- Hard group-lasso screen is optional (box oracle only).
- Soft attribution is post-hoc via Jacobian column norms \(c_j\).
- Protocol: probe \(m\) → spectrum \(\hat r\) → refit → soft screen → bootstrap → paired NLL / align+SR.
- Hard-cut tables kept as reference; rectangular oneshot and a short GW sketch are called out separately.

Absolute geometry NLL is explicitly not comparable to the screened box floor. Diff against `main.tex` if you want side-by-side reading.

---

**User**

please use the writing skill to make the outputs readable.

---

**Cursor**

I'll rewrite the new `main_mod.tex` prose with the STE writing skill and lint it.

Rewriting the Estimation section and new demo paragraphs in STE-flavored prose, then linting.

Fixing the four remaining STE violations.

Rewrote Estimation and the new demo paragraphs in `main_mod.tex` with STE-flavored prose.

Lint on stripped prose: **2.18 → 0.11** violations per 100 words. The one leftover “nominalization” is the technical phrase *description length*.

---

**User**

I still dont understand why we need the subset theta_A and theta_B. By default, aren't we feeding all of theta into eta?

---

**Cursor**

Yes — in the current rectangular protocol you feed all of \(\theta\) into \(\eta_\phi:\mathbb{R}^d\to\mathbb{R}^m\). There is no hard \(A\)/\(B\) cut in the network.

The split \(\theta=(\theta_A,\theta_B)\) is only for the older hard-cut model in `main.tex`. That model builds a full density \(q(\theta\mid x)\) on all of \(\mathbb{R}^d\). The map \(\eta\) only covers \(m\) coordinates, so something must be said about the rest. The model puts those leftover coordinates at the prior \(\pi_B\) and uses volume \(\log|\det J_A|\). Under a box prior, \(\pi_B\) is tractable only if \(B\) is a coordinate subset.

The geometry objective drops that claim. It trains
\[
\tfrac12\|\eta-\hat\eta\|^2-\tfrac12\log\det(JJ^\top)
\]
with rectangular \(J\in\mathbb{R}^{m\times d}\). Absolute NLL is then not a normalised box posterior, so you do not need \(\theta_B\) or \(\pi_B\).

Use \(A\)/\(B\) only if you want the hard-cut normalised \(q\). For oneshot rank detection and soft ranking by column norms, all of \(\theta\) enters \(\eta\).

---

**User**

so the "rectangular" model cannot be regarded as an accurate log-probability. is it a bound on -logp(theta | x) ?

---

**Cursor**

No. For \(m<d\) the rectangular objective is not \(-\log q(\theta\mid x)\) for any normalised \(q\), so it is not a variational upper bound on \(-\log p(\theta\mid x)\).

When \(m=d\) and \(\eta\) is a diffeomorphism,
\[
\int \mathcal{N}\bigl(\eta(\theta);\hat\eta(x),I\bigr)\,|\det J|\,d\theta=1,
\]
so the geometry NLL is a proper change-of-variables density (Lebesgue on \(\theta\)). Then
\[
\mathbb{E}[-\log q]\ge H(\theta\mid x)=\mathbb{E}[-\log p(\theta\mid x)],
\]
with equality iff \(q=p\).

When \(m<d\), coarea gives
\[
\int \mathcal{N}\bigl(\eta(\theta);\hat\eta,I\bigr)\sqrt{\det(JJ^\top)}\,d\theta
=\int \mathcal{N}(u;\hat\eta,I)\,\mathcal{H}^{d-m}\bigl(\eta^{-1}(u)\bigr)\,du,
\]
which is not \(1\). The missing piece is a fibre measure on \(\eta^{-1}(u)\). The hard-cut model supplies that with \(\pi_B\). The rectangular loss does not.

So \(\mathcal{L}_{\mathrm{geom}}\) is a training score on the \(m\)-dimensional summary: match \(\eta\) to \(\hat\eta(x)\), and keep \(\sqrt{\det(JJ^\top)}\) from collapsing. It is not a bound on \(-\log p(\theta\mid x)\). Compare candidates only within the same objective (paired NLL, spectrum, column norms).

---

**User**

can you re-render this response for cursor chat.

---

**Cursor**

**No.** For \(m < d\), the rectangular objective is not \(-\log q(\theta \mid x)\) for any normalised \(q\). So it is not a variational upper bound on \(-\log p(\theta \mid x)\).

### When \(m = d\)

If \(\eta\) is a diffeomorphism,

\[
\int \mathcal{N}\bigl(\eta(\theta);\hat\eta(x), I\bigr)\,|\det J|\,d\theta = 1.
\]

Then the geometry NLL is a proper change-of-variables density (Lebesgue on \(\theta\)), and

\[
\mathbb{E}[-\log q] \ge H(\theta \mid x) = \mathbb{E}[-\log p(\theta \mid x)],
\]

with equality iff \(q = p\).

### When \(m < d\)

Coarea gives

\[
\int \mathcal{N}\bigl(\eta(\theta);\hat\eta, I\bigr)\sqrt{\det(JJ^\top)}\,d\theta
= \int \mathcal{N}(u;\hat\eta, I)\,\mathcal{H}^{d-m}\bigl(\eta^{-1}(u)\bigr)\,du,
\]

which is not \(1\). You still need a fibre measure on \(\eta^{-1}(u)\). The hard-cut model supplies that with \(\pi_B\). The rectangular loss does not.

### What \(\mathcal{L}_{\mathrm{geom}}\) is

A training score on the \(m\)-dimensional summary: match \(\eta\) to \(\hat\eta(x)\), and keep \(\sqrt{\det(JJ^\top)}\) from collapsing. It is not a bound on \(-\log p(\theta \mid x)\).

Compare candidates only within the same objective (paired NLL, spectrum, column norms).

---

**User**

im seeing double outputs for each equation

---

**Cursor**

**No.** For \(m < d\), the rectangular objective is not \(-\log q(\theta \mid x)\) for any normalised \(q\), so it is not a variational upper bound on \(-\log p(\theta \mid x)\).

**When \(m = d\).** If \(\eta\) is a diffeomorphism, \(\int \mathcal{N}(\eta(\theta);\hat\eta(x), I)\,|\det J|\,d\theta = 1\). The geometry NLL is then a proper change-of-variables density (Lebesgue on \(\theta\)), and \(\mathbb{E}[-\log q] \ge H(\theta \mid x) = \mathbb{E}[-\log p(\theta \mid x)]\), with equality iff \(q = p\).

**When \(m < d\).** Coarea gives \(\int \mathcal{N}(\eta(\theta);\hat\eta, I)\sqrt{\det(JJ^\top)}\,d\theta = \int \mathcal{N}(u;\hat\eta, I)\,\mathcal{H}^{d-m}(\eta^{-1}(u))\,du\), which is not \(1\). You still need a fibre measure on \(\eta^{-1}(u)\). The hard-cut model supplies that with \(\pi_B\). The rectangular loss does not.

**What \(\mathcal{L}_{\mathrm{geom}}\) is.** A training score on the \(m\)-dimensional summary: match \(\eta\) to \(\hat\eta(x)\), and keep \(\sqrt{\det(JJ^\top)}\) from collapsing. It is not a bound on \(-\log p(\theta \mid x)\). Compare candidates only within the same objective (paired NLL, spectrum, column norms).

---

**User**

ok is there any way the geometry loss could go wrong, since it's not a proper lower bound?

---

**Cursor**

Yes. Because it is not a normalised \(-\log q(\theta\mid x)\), several failure modes stay open.

**Absolute score is not trustworthy.** Lower \(\mathcal{L}_{\mathrm{geom}}\) does not mean a better posterior on \(\theta\). Only paired comparisons under the same objective, plus spectrum / calibration / column norms, are meaningful.

**Volume can cheat.** The \(-\tfrac12\log\det(JJ^\top)\) term wants a large Gram volume. The residual couples that to predictability from \(x\), but the trade can still favour stretched, weakly informative axes. Learnable scales make this worse. That is why \(\hat r\) comes from the spectrum floor, not from \(\arg\min_m\mathcal{L}_{\mathrm{geom}}\).

**Spare axes are under-penalised.** With no \(\pi_B\) term, an unused axis does not pay the hard-cut box cost \((d-m)\log w\). The null can sit a bit above \(1\), and a weak true axis can hide under the floor \(\tau\).

**Fibre structure is ignored.** Anything in \(\ker J\) or along level sets of \(\eta\) is invisible to the loss. The map can discard an informative combination, or mix nuisance directions into \(\eta\), and still look fine on residuals.

**Support mismatch.** Residual terms use \((\theta,x)\) pairs. Geometry terms use fresh \(\theta\sim\pi\). If those draws are not from \(\pi\), volume and residual train on different supports.

**Practical guardrails.** Check \(z\sim\mathcal{N}(0,I)\), spectrum vs \(\tau\), soft column norms, bootstrap vote on \(\hat r\), and paired candidate scores. Those catch the failure modes the missing bound does not.

---

**User**

is there a pernicious case we can construct where this loss is likely to fail?

---

**Cursor**

Yes. A clean one is a **mean-blind parameter**.

**Construction.** Let \(\theta=(\mu,\alpha)\) and draw \(x_{1:n}\) from a location–shape family whose mean depends only on \(\mu\), while \(\alpha\) changes only variance or skewness. Keep the estimator close to what the scripts use: a net on pooled features that are effectively mean-like (or any \(t(x)\) with \(\mathbb{E}[t\mid\theta]\) independent of \(\alpha\)).

**What happens.** The residual \(\tfrac12\|\eta-\hat\eta(x)\|^2\) only rewards coordinates that move those features. The fit settles on \(\eta\approx\eta(\mu)\), spectrum gives \(\hat r=1\), \(z\) looks calibrated, and the soft column norm on \(\alpha\) is near zero. The volume term never pays for missing \(\alpha\), because \(\alpha\) sits in the fibre of \(\eta\).

**Why it is pernicious.** Every geometry diagnostic you trust can pass, while \(p(\alpha\mid x)\) is not captured. A proper \(-\log q(\theta\mid x)\) (or a residual on a fuller summary of \(x\)) would still feel \(\alpha\).

**Weaker sibling.** Put the same information in a coordinate that is almost prior-determined by an informative proxy (strong prior correlation, likelihood almost ignores it). Rank can look right, but column norms attribute the axis to the wrong \(\theta_j\).

**Practical tell.** If you only trust \(\mathcal{L}_{\mathrm{geom}}+z+\lambda\), you miss shape parameters. Append variance/skew summaries to the estimator input, or check leakage of each \(\theta_j\) onto richer functions of \(x\).

---

**User**

hm, so would the correct procedure be: 1) rectangular loss, check spectrum. 2) re-train using the A/B cut for participating parameters 

---

**Cursor**

Yes, when you need a normalised box posterior and a coordinate complement exists.

**Recommended hybrid**
1. Rectangular probe → spectrum → \(\hat r\).
2. Soft-screen column norms → candidate set \(A\) (top coordinates that actually move \(\eta\)).
3. Hard-cut retrain with that \(A\), map \(\eta(\theta_A)\), volume \(\log|\det J_A|\), and complement \(\pi_B\).
4. Check \(z\), complement PIT / leakage, then score candidates.

Step 1 finds rank and who participates without committing to \(\pi_B\). Step 3 restores a proper \(-\log q(\theta\mid x)\).

**Do not hard-cut when** every coordinate enters the likelihood and only \(m\) combinations carry information (heater; often GW). Then no honest \(B\) exists. Stay rectangular, or use a Gaussian / whitened subspace model.

**Detail.** \(\hat r\) is the number of informative axes, not automatically \(|A|\). If column norms light up more than \(\hat r\) coordinates, take a nest of size about \(\hat r\) (or a bit larger), refit hard-cut at \(m=\hat r\), and confirm with leakage. For pure nuisance Rosenbrock, \(\hat r=|A|=2\) and the hybrid collapses to what `rosenbrock20_oneshot.py` does with a pre-screen.

---

**User**

ok interesting. when all parameters enter the likelihood, are we susceptible to missing something ? or does the full-rank of the problem help us avoid the fiber problem

---

**Cursor**

**All parameters enter the likelihood** does not fix the fibre problem. **Fisher rank \(r=d\) with \(m=d\)** does, in principle.

**All enter \(\neq\) full rank.** Extrinsics can enter \(h(f)\) while the spectrum still has only ~2 strong axes. Each \(\theta_j\) has some score, but only \(r<d\) combinations carry appreciable information. Rectangular \(\eta:\mathbb{R}^d\to\mathbb{R}^{\hat r}\) still has \((d-\hat r)\)-dimensional fibres. Anything you leave out of \(\eta\) (or that sits below \(\tau\)) is invisible to the loss.

**When the fibre goes away.** If the problem is truly full rank (\(r=d\)) and you fit a diffeomorphism at \(m=d\), then \(\ker J=\{0\}\) and there is no fibre left to hide in. Coarea pathologies for \(m<d\) do not apply. You still have a thresholding problem: a tiny eigenvalue can look like null and get dropped if \(\tau\) is coarse or \(m_{\mathrm{probe}}<d\).

**What you remain open to (even if all \(\theta_j\) enter).**
- Weak but real directions near the spectrum floor (typical for GW extrinsics).
- Estimator blindness (mean-only features vs shape parameters).
- Bad local minima with \(\operatorname{rank}J<\hat r\).

**Practical read.** For GW-like problems, rectangular + spectrum is the right first step. Do not expect “all enter” to protect you. Read the graded profile \(\tfrac12\log(\lambda_i/\lambda_{\mathrm{null}})\), and treat near-null axes as “maybe keep” rather than “safe to discard.”

