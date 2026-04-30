"""
Post-pruning atom-level regrouping for flat symbolic expressions.

Motivation
----------
``postprocessing_utils.optimize_sparse_rotation`` rotates in the space of ESR
*linear-parameter slots*.  That means two different slots that happen to
multiply the same symbolic atom (e.g. ``exp(0.268*X5) * X1`` in one row and
``exp(0.268*X5) * X3`` in another) are treated as unrelated columns of ``M``.
The rotation can therefore only align bare ``c * X_j`` terms — not the
``c * f(X)`` blocks that dominate realistic outputs.  Combined with the
smooth-L1 loss and the non-smooth gradient of the ``QR + sign`` projection,
the net result is that grouping across components rarely fires in practice.

This module addresses that by operating *after* the usual prune:

1. Parse each pruned η_i into a list of ``(coefficient, atom)`` pairs using
   sympy, where an *atom* is the symbolic structure after pulling out the
   leading numeric scalar.  Atoms are identified by canonical sympy form, so
   ``0.758*X1*exp(0.268*X5)`` and ``0.447*X3*exp(0.268*X5)`` share the atom
   ``exp(0.268*X5)`` *only if* they are linear in different monomial cofactors
   — otherwise the atom is the full ``X1*exp(0.268*X5)`` / ``X3*exp(0.268*X5)``.

2. Build a coefficient matrix ``C[i, k] = coeff of atom k in row i``.

3. Optimise an orthogonal ``R`` to sparsify ``R @ C`` directly (Cayley
   parameterisation of ``SO(n)``; reweighted-L1 loss).  Orthogonality
   preserves the ``Q = (A J⁺)ᵀ F (A J⁺) ≈ I`` flattening property.

4. Apply ``R`` to the expressions as a *linear combination of rows*,
   ``η'_i = Σ_j R_{ij} η_j``, collect like atoms, threshold small
   coefficients and verify the flatness score.

5. Optionally snap numerically-close coefficients of the same atom across
   rows to a shared value, subject to the same flatness check.

A cheap alternative to step 3 — :func:`greedy_givens_sparsify` — performs
2-row Givens rotations per atom and often recovers most of the gains with no
nonlinear optimisation.

Limitations
-----------
Orthogonal rotations cannot reduce atom count between two rows unless those
rows share enough atoms that zeroing one shared atom doesn't spawn too many
*new* atoms in the partner row.  In realistic pruning outputs, rows often
have mostly disjoint atoms, which makes the rotation unhelpful.  When that
happens:

- :func:`analyze_atom_sharing` will tell you which atom/row-pair combinations
  are worth considering and which are structurally trapped.
- :func:`snap_shared_coefficients` remains useful: it doesn't change the atom
  count but aligns the numeric values of repeated atoms so the output reads
  more uniformly.
- As a last resort, :func:`regroup_like_terms` with ``orthogonal=False`` will
  accept a non-orthogonal ``R`` and carry ``A_new = A_old R^{-1}`` through to
  :func:`check_flattening`.

Non-orthogonal regrouping
-------------------------
If you are willing to lose strict ``R R^T = I``, you can use an arbitrary
invertible ``R`` as long as the combined rotation ``A_new = A_old R^{-1}`` is
carried through to :func:`check_flattening`.  A convenience path for this is
noted in :func:`regroup_like_terms` (``orthogonal=False``).

Author: companion to ``postprocessing_utils.py``
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import sympy
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any

try:
    from .preprocessing_utils import batch_flatten_fisher
    from .postprocessing_utils import (
        make_check_flattening_fn,
        norm as _frob,
        repair_coordinate_rank_deficiency,
        replace_floats,
    )
except ImportError:
    from preprocessing_utils import batch_flatten_fisher  # type: ignore
    from postprocessing_utils import (  # type: ignore
        make_check_flattening_fn,
        norm as _frob,
        repair_coordinate_rank_deficiency,
        replace_floats,
    )


# =============================================================================
# 1. ATOM EXTRACTION
# =============================================================================

def _sympify_expr(s: str) -> sympy.Expr:
    """Parse ``s`` into a sympy expression, tolerating the output of our
    ESR/postprocessing pipeline.

    Uses the ``^`` -> ``**`` convention and ``Abs``/``log``/etc. as already
    available in sympy's default locals.
    """
    s = str(s).replace("^", "**")
    return sympy.sympify(s)


def _split_coeff_atom(term: sympy.Expr) -> Tuple[float, sympy.Expr]:
    """Split a single additive term into ``(numeric_coeff, atom)``.

    ``sympy.Expr.as_coeff_Mul()`` pulls out the leading ``Number`` factor of a
    product.  If the term is a sum (e.g. ``X1 + 2``) we return ``(1.0, term)``
    — additive structure inside an atom is preserved.
    """
    c, rest = term.as_coeff_Mul()
    try:
        coeff = float(c)
    except (TypeError, ValueError):
        coeff = 1.0
        rest = term
    return coeff, rest


def extract_atoms(
    expressions: Sequence[str],
    expand: bool = False,
) -> Tuple[List[sympy.Expr], np.ndarray, List[List[Tuple[int, float]]]]:
    """Extract unique atoms and build an ``(n_rows, n_atoms)`` coefficient
    matrix ``C`` such that ``η_i = Σ_k C[i, k] * atom_k``.

    Parameters
    ----------
    expressions
        One symbolic expression string per row.
    expand
        If True, apply ``sympy.expand`` to each row first.  This can split
        products like ``(a + b) * exp(c)`` into ``a*exp(c) + b*exp(c)`` which
        creates more, *shared* atoms.  Off by default because it can blow up
        expression size on powers.

    Returns
    -------
    atoms
        Canonical sympy atoms in discovery order, length ``n_atoms``.
    C
        Coefficient matrix, shape ``(n_rows, n_atoms)``.
    per_row_terms
        ``per_row_terms[i]`` is a list of ``(atom_idx, coeff)`` pairs for row
        ``i`` — useful for diagnostics and for reconstructing rows that had
        multiple terms collapse onto the same atom.
    """
    atoms: List[sympy.Expr] = []
    atom_index: Dict[str, int] = {}
    per_row_terms: List[List[Tuple[int, float]]] = []

    for expr_str in expressions:
        expr = _sympify_expr(expr_str)
        if expand:
            expr = sympy.expand(expr)

        if isinstance(expr, sympy.Add):
            terms = list(expr.args)
        else:
            terms = [expr]

        row: List[Tuple[int, float]] = []
        for t in terms:
            coeff, atom = _split_coeff_atom(t)
            if coeff == 0.0:
                continue
            key = sympy.srepr(atom)
            if key not in atom_index:
                atom_index[key] = len(atoms)
                atoms.append(atom)
            row.append((atom_index[key], coeff))
        per_row_terms.append(row)

    n_rows = len(expressions)
    n_atoms = len(atoms)
    C = np.zeros((n_rows, n_atoms), dtype=float)
    for i, row in enumerate(per_row_terms):
        for k, c in row:
            C[i, k] += c
    return atoms, C, per_row_terms


# =============================================================================
# 1b. SHARING DIAGNOSTICS
# =============================================================================

def analyze_atom_sharing(
    expressions: Sequence[str],
    zero_tol: float = 5e-3,
    expand: bool = False,
) -> Dict[str, Any]:
    """Report which atoms appear in multiple rows, and estimate whether a
    Givens rotation *could* reduce the atom count on each shared atom.

    For each atom shared by rows ``(i, j)``, an orthogonal rotation that
    zeros the atom in row ``i`` mixes *every* atom of row ``j`` into row
    ``i`` with weight ``sin(theta)`` (and scales row ``i``'s own atoms by
    ``cos(theta)``).  The rotation is "L0-profitable" iff the number of
    atoms *unique to row j* (that would be newly added to row ``i`` above
    ``zero_tol``) is less than 1 (i.e. row ``j`` is a subset of row ``i``'s
    atoms in structure).

    This is a quick-look tool that explains *why* :func:`greedy_givens_sparsify`
    often finds nothing on realistic SR output.

    Returns
    -------
    report
        Dict with:

        - ``shared_atoms``: list of ``(atom_str, rows)`` tuples.
        - ``pair_profit_hint``: dict keyed by ``(i, j)`` giving the number of
          atoms that are "row-j-only" and would be introduced into row ``i``
          above ``zero_tol`` if we zeroed out their shared atom in row ``i``.
          Zero means the rotation is fully profitable; larger values mean
          progressively worse.
    """
    atoms, C, _ = extract_atoms(expressions, expand=expand)
    n_rows, n_atoms = C.shape
    present = np.abs(C) > zero_tol

    shared: List[Tuple[str, List[int]]] = []
    for k, atom in enumerate(atoms):
        rows = list(np.where(present[:, k])[0])
        if len(rows) >= 2:
            shared.append((str(atom), rows))

    pair_hint: Dict[Tuple[int, int], int] = {}
    for atom_str, rows in shared:
        # approximate Givens mixing magnitude
        k = next(idx for idx, a in enumerate(atoms) if str(a) == atom_str)
        for i in rows:
            for j in rows:
                if i == j:
                    continue
                # atoms in row j not in row i, weighted by sin(theta)
                theta = np.arctan2(C[i, k], C[j, k])
                s = abs(np.sin(theta))
                newly_introduced = 0
                for kp in range(n_atoms):
                    if kp == k:
                        continue
                    if present[j, kp] and not present[i, kp]:
                        if abs(C[j, kp]) * s > zero_tol:
                            newly_introduced += 1
                pair_hint[(i, j)] = min(
                    newly_introduced, pair_hint.get((i, j), 10**9)
                )

    return dict(
        n_rows=n_rows,
        n_atoms=n_atoms,
        shared_atoms=shared,
        pair_profit_hint=pair_hint,
    )


# =============================================================================
# 2. CAYLEY-PARAMETERISED SO(n)
#
# These now live in ``postprocessing_utils`` (so the rotation optimisers in
# that module can share them); we re-export here for backward compatibility
# with callers of ``postprocess_new``.
# =============================================================================

try:
    from .postprocessing_utils import (
        cayley_rotation,
        expm_rotation,
        _skew_from_params_jax as _skew_from_params,
    )
except ImportError:
    from postprocessing_utils import (  # type: ignore
        cayley_rotation,
        expm_rotation,
        _skew_from_params_jax as _skew_from_params,
    )


# =============================================================================
# 3. SPARSE ROTATION ON THE ATOM MATRIX
# =============================================================================

def _reweighted_l1(
    X: jnp.ndarray,
    weights: jnp.ndarray,
    eps: float,
) -> jnp.ndarray:
    """Smooth reweighted-L1 surrogate: sum(weights * sqrt(X^2 + eps^2)).

    When iterated with ``weights_{k+1} = 1 / sqrt(X_k^2 + eps^2)`` this
    converges to ``||X||_0``-ish sparsity (Candès-Wakin-Boyd reweighted L1).
    """
    return jnp.sum(weights * jnp.sqrt(X * X + eps * eps))


def optimize_sparse_rotation_atoms(
    C: np.ndarray,
    param: str = "cayley",
    n_outer: int = 5,
    n_inner: int = 400,
    eps: float = 1e-2,
    lr: float = 5e-2,
    momentum: float = 0.9,
    seed: int = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Find an orthogonal ``R`` minimising sparsity-inducing loss on ``R @ C``.

    Uses iterative reweighting to approximate the L0 norm.

    Parameters
    ----------
    C
        Coefficient matrix ``(n_rows, n_atoms)`` produced by
        :func:`extract_atoms`.
    param
        ``"cayley"`` or ``"expm"`` — see :func:`cayley_rotation` /
        :func:`expm_rotation`.
    n_outer
        Number of reweighting iterations.
    n_inner
        Inner gradient-descent iterations per outer step.
    eps
        Smoothing parameter for the L1-like surrogate.
    lr, momentum, seed, verbose
        Optimiser settings.

    Returns
    -------
    R
        Optimised rotation, shape ``(n_rows, n_rows)``.
    info
        Diagnostics: ``sparsity_before``, ``sparsity_after``, ``l1_before``,
        ``l1_after``, ``ortho_error``.
    """
    n = C.shape[0]
    n_params = n * (n - 1) // 2
    C_j = jnp.array(C)

    if param == "cayley":
        rot_fn = cayley_rotation
    elif param == "expm":
        rot_fn = expm_rotation
    else:
        raise ValueError(f"Unknown param={param!r}; use 'cayley' or 'expm'.")

    rng = np.random.default_rng(seed)
    w = jnp.array(rng.normal(scale=1e-3, size=(n_params,)))
    weights = jnp.ones_like(C_j)

    @jax.jit
    def loss_fn(w, weights):
        R = rot_fn(w, n)
        RC = R @ C_j
        return _reweighted_l1(RC, weights, eps)

    @jax.jit
    def grad_fn(w, weights):
        return jax.grad(loss_fn)(w, weights)

    velocity = jnp.zeros_like(w)
    best_w = w
    best_obj = float("inf")

    l1_before = float(jnp.abs(C_j).sum())
    sparsity_before = float(jnp.mean(jnp.abs(C_j) < 1e-6))

    for outer in range(n_outer):
        for inner in range(n_inner):
            g = grad_fn(w, weights)
            velocity = momentum * velocity - lr * g
            w = w + velocity
            if jnp.any(jnp.isnan(w)):
                if verbose:
                    print(f"  [warn] NaN at outer {outer}, inner {inner}")
                w = best_w
                velocity = jnp.zeros_like(w)
                break
        R_cur = rot_fn(w, n)
        RC_cur = R_cur @ C_j
        obj = float(jnp.abs(RC_cur).sum())
        if obj < best_obj:
            best_obj = obj
            best_w = w
        # reweight with current values (Candès-Wakin-Boyd)
        weights = 1.0 / jnp.sqrt(RC_cur * RC_cur + eps * eps)
        if verbose:
            print(
                f"  outer {outer+1:2d}/{n_outer}: ||R C||_1 = {obj:.4f} "
                f"(best {best_obj:.4f})"
            )

    R = np.asarray(rot_fn(best_w, n))
    RC = R @ C
    l1_after = float(np.abs(RC).sum())
    sparsity_after = float(np.mean(np.abs(RC) < 1e-6))
    ortho_error = float(np.linalg.norm(R @ R.T - np.eye(n), ord="fro"))

    info = dict(
        l1_before=l1_before,
        l1_after=l1_after,
        sparsity_before=sparsity_before,
        sparsity_after=sparsity_after,
        ortho_error=ortho_error,
    )
    if verbose:
        print(
            f"  done. L1 {l1_before:.4f} -> {l1_after:.4f}; "
            f"exact-zero fraction {sparsity_before:.2%} -> {sparsity_after:.2%}; "
            f"ortho err {ortho_error:.2e}"
        )
    return R, info


# =============================================================================
# 4. GREEDY GIVENS (fast, zero-training alternative)
# =============================================================================

def _givens(n: int, i: int, j: int, theta: float) -> np.ndarray:
    """Givens rotation on rows/cols ``(i, j)`` by ``theta``."""
    G = np.eye(n)
    c, s = np.cos(theta), np.sin(theta)
    G[i, i] = c
    G[j, j] = c
    G[i, j] = -s
    G[j, i] = s
    return G


def greedy_givens_sparsify(
    C: np.ndarray,
    max_sweeps: int = 20,
    atol: float = 1e-6,
    criterion: str = "l0",
    zero_tol: float = 5e-3,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Greedily apply Givens rotations to sparsify ``C`` column-by-column.

    For each column (atom) ``k`` and each pair of rows ``(i, j)`` where
    ``|C_{i,k}| + |C_{j,k}|`` is non-trivial, try the Givens angle that
    zeroes ``C_{i,k}`` (rotates the j-th entry to absorb it).  Accept if the
    chosen ``criterion`` improves.  Sweep until no accept.

    Parameters
    ----------
    C
        Coefficient matrix.
    max_sweeps, atol
        Loop control.
    criterion
        Acceptance rule:

        - ``"l0"`` (default): accept if the number of entries with
          ``|C_{ij}| > zero_tol`` strictly decreases (ties broken by L1).
          This is closer to the user-facing notion of "fewer atoms per row".
        - ``"l1"``: accept if the total absolute sum decreases.  Stricter;
          will reject rotations that swap atoms for different atoms of
          similar magnitude — which is common when rows have very different
          structure.
    zero_tol
        Threshold used by the L0 criterion.  Match this to the reconstruction
        threshold you plan to use downstream.
    verbose
        Print per-sweep diagnostics.

    Returns ``R`` such that ``R @ C`` is the sparsified matrix.

    Notes
    -----
    Even with ``criterion="l0"``, this heuristic cannot reduce the atom count
    if every row that shares an atom also contains atoms that are unique to
    that row.  In that regime, the complementary :func:`snap_shared_coefficients`
    step is what you want.
    """
    if criterion not in ("l0", "l1"):
        raise ValueError(f"criterion must be 'l0' or 'l1', got {criterion!r}")

    n = C.shape[0]
    R = np.eye(n)
    Ccur = C.copy()
    l1_start = float(np.abs(Ccur).sum())
    l0_start = int((np.abs(Ccur) > zero_tol).sum())
    n_accepts = 0

    def _l0(M: np.ndarray) -> int:
        return int((np.abs(M) > zero_tol).sum())

    def _score(M: np.ndarray) -> Tuple[int, float]:
        """Lower-is-better score under the active criterion."""
        if criterion == "l0":
            return (_l0(M), float(np.abs(M).sum()))
        return (0, float(np.abs(M).sum()))

    for sweep in range(max_sweeps):
        accepted_any = False
        for k in range(Ccur.shape[1]):
            col = Ccur[:, k]
            order = np.argsort(-np.abs(col))
            if np.abs(col[order[0]]) < atol:
                continue
            j = order[0]
            for i in order[1:]:
                if np.abs(col[i]) < atol:
                    break
                theta = np.arctan2(col[i], col[j])
                G = _givens(n, i, j, theta)
                Ctrial = G @ Ccur
                if _score(Ctrial) < _score(Ccur):
                    Ccur = Ctrial
                    R = G @ R
                    col = Ccur[:, k]
                    accepted_any = True
                    n_accepts += 1
        if verbose:
            print(
                f"  sweep {sweep+1:2d}: L0 = {_l0(Ccur):3d}, "
                f"L1 = {np.abs(Ccur).sum():.4f} "
                f"(accepts so far: {n_accepts})"
            )
        if not accepted_any:
            break
    info = dict(
        l1_before=l1_start,
        l1_after=float(np.abs(Ccur).sum()),
        l0_before=l0_start,
        l0_after=_l0(Ccur),
        n_accepts=n_accepts,
        ortho_error=float(np.linalg.norm(R @ R.T - np.eye(n), ord="fro")),
        criterion=criterion,
    )
    return R, info


# =============================================================================
# 5. APPLY ROTATION + RECOLLECT EXPRESSIONS
# =============================================================================

def reconstruct_expressions(
    R: np.ndarray,
    atoms: List[sympy.Expr],
    C: np.ndarray,
    threshold: float = 1e-3,
    decimal: int = 3,
) -> List[str]:
    """Reassemble ``η'_i = Σ_k (R @ C)_{i,k} * atom_k`` as sympy strings.

    Coefficients below ``threshold`` in absolute value are dropped.  Remaining
    coefficients are rounded to ``decimal`` places.
    """
    new_C = R @ C
    new_exprs: List[str] = []
    for i in range(new_C.shape[0]):
        terms: List[sympy.Expr] = []
        for k, atom in enumerate(atoms):
            c = float(new_C[i, k])
            if abs(c) < threshold:
                continue
            terms.append(sympy.Float(round(c, decimal)) * atom)
        if not terms:
            new_exprs.append("0")
            continue
        expr = sympy.Add(*terms)
        # Collect like terms where sympy can (safety net — by construction
        # atoms should already be distinct).  Avoid ``nsimplify`` here: it
        # silently converts floats to rationals, which we do NOT want.
        expr = sympy.expand(
            expr,
            mul=False,
            power_exp=False,
            power_base=False,
            multinomial=False,
        )
        new_exprs.append(str(expr))
    return new_exprs


# =============================================================================
# 6. FLATTENING VERIFICATION
# =============================================================================

def _flat_score(
    expressions: Sequence[str],
    X: np.ndarray,
    Fs: np.ndarray,
    n_params: int,
    check_flattening_fn: Optional[Callable] = None,
    A: Optional[np.ndarray] = None,
) -> float:
    """Mean ``||flat - I||_F`` over the dataset.  Mirrors the score used by
    :func:`postprocessing_utils._mean_flattening_score`.
    """
    if check_flattening_fn is None:
        check_flattening_fn = make_check_flattening_fn(X, Fs)
    A_j = jnp.eye(n_params) if A is None else jnp.array(A)
    flats, _ = check_flattening_fn(list(expressions), A=A_j)
    return float(jax.vmap(_frob)(flats - jnp.eye(n_params)).mean())


# =============================================================================
# 7. COEFFICIENT SNAP-TO-SHARED-VALUE
# =============================================================================

def snap_shared_coefficients(
    expressions: Sequence[str],
    X: np.ndarray,
    Fs: np.ndarray,
    n_params: int,
    rel_tol: float = 0.2,
    flat_tol: float = 0.05,
    A: Optional[np.ndarray] = None,
    check_flattening_fn: Optional[Callable] = None,
    verbose: bool = True,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Snap numerically-close coefficients of the same atom to their mean.

    For every atom that appears in ≥2 rows, compute the coefficients
    ``c_1, ..., c_m``.  If ``max|c - mean| / |mean|`` is below ``rel_tol`` and
    replacing them all with the mean keeps the relative change in the
    flatness score below ``flat_tol``, commit the replacement.

    This is independent of, and complementary to, the rotation-based
    regrouping: often a rotation will align coefficients to be *close but not
    equal*, and this step finishes the job.

    Returns the new expressions and a per-atom log of actions taken.
    """
    atoms, C, _per_row = extract_atoms(expressions)
    if check_flattening_fn is None:
        check_flattening_fn = make_check_flattening_fn(X, Fs)
    ref = _flat_score(expressions, X, Fs, n_params, check_flattening_fn, A)

    current = list(expressions)
    log: List[Dict[str, Any]] = []
    for k, atom in enumerate(atoms):
        col = C[:, k]
        present = np.where(np.abs(col) > 1e-10)[0]
        if len(present) < 2:
            continue
        vals = col[present]
        mean = float(np.mean(vals))
        if abs(mean) < 1e-12:
            continue
        spread = float(np.max(np.abs(vals - mean)) / abs(mean))
        if spread > rel_tol:
            continue

        # Build candidate expressions
        C_snap = C.copy()
        C_snap[present, k] = mean
        candidate = reconstruct_expressions(np.eye(C.shape[0]), atoms, C_snap)
        try:
            score = _flat_score(candidate, X, Fs, n_params, check_flattening_fn, A)
        except Exception:
            continue
        rel_delta = abs(score - ref) / max(ref, 1e-30)
        accept = rel_delta < flat_tol
        action = dict(
            atom=str(atom),
            rows=list(present.tolist()),
            values=[float(v) for v in vals],
            snapped_to=mean,
            rel_spread=spread,
            flat_rel_delta=rel_delta,
            accepted=accept,
        )
        log.append(action)
        if accept:
            C = C_snap
            current = candidate
            ref = score
            if verbose:
                print(
                    f"  snapped atom {sympy.sstr(atom)[:40]!r:42} "
                    f"across rows {list(present)} -> {mean:.4g} "
                    f"(spread {spread:.2%}, Δflat {rel_delta:.2%})"
                )
        else:
            if verbose:
                print(
                    f"  rejected snap atom {sympy.sstr(atom)[:40]!r:42} "
                    f"(Δflat {rel_delta:.2%} > {flat_tol:.2%})"
                )

    return current, log


# =============================================================================
# 8. TOP-LEVEL WRAPPER
# =============================================================================

def regroup_like_terms(
    expressions: Sequence[str],
    X: np.ndarray,
    Fs: np.ndarray,
    n_params: int,
    method: str = "atoms",
    A: Optional[np.ndarray] = None,
    threshold: float = 1e-3,
    flat_tol: float = 0.1,
    decimal: int = 3,
    expand: bool = False,
    do_snap: bool = True,
    snap_rel_tol: float = 0.2,
    snap_flat_tol: float = 0.05,
    orthogonal: bool = True,
    rotation_kwargs: Optional[Dict[str, Any]] = None,
    check_flattening_fn: Optional[Callable] = None,
    repair_rank_deficiency: bool = True,
    rank_repair_scale: str = "mean_whitened",
    rank_repair_rtol: float = 1e-8,
    rank_repair_atol: float = 1e-10,
    verbose: bool = True,
) -> Tuple[List[str], np.ndarray, Dict[str, Any]]:
    """End-to-end regrouping: atom extraction → sparsifying rotation → snap.

    Call this *after* your usual :func:`postprocess_eqs` run.

    Parameters
    ----------
    expressions
        Pruned expressions, one per component.
    X, Fs, n_params
        Data and Fisher matrices used to score flatness.
    method
        - ``"atoms"``: Cayley-parameterised nonlinear optimisation
          (:func:`optimize_sparse_rotation_atoms`).
        - ``"greedy"``: Greedy Givens (:func:`greedy_givens_sparsify`).
        - ``"none"``: Skip rotation; only do optional snapping.
    A
        Existing rotation used by your flattening pipeline (so flatness is
        scored consistently with the rest of the codebase).  When
        ``orthogonal=False``, the returned ``A_new`` is ``A @ R^{-1}`` so that
        downstream ``check_flattening(A_new)`` evaluations stay flat.
    threshold
        Drop atom coefficients below this absolute value in the final
        reconstruction.
    flat_tol
        If the relative change in the flatness score after regrouping exceeds
        this, fall back to the input expressions (and report it).
    expand
        Forwarded to :func:`extract_atoms`.
    do_snap
        Run :func:`snap_shared_coefficients` after the rotation.
    snap_rel_tol, snap_flat_tol
        Forwarded to :func:`snap_shared_coefficients`.
    orthogonal
        If False, use the (possibly non-orthogonal) rotation as-is and fold
        ``R^{-1}`` into ``A``.  Not recommended unless you need that extra
        flexibility.
    rotation_kwargs
        Extra keyword arguments for the chosen optimiser.
    repair_rank_deficiency
        If True, run a final Jacobian-rank/coverage repair.  Duplicate or
        dependent output rows are flagged and redundant rows are replaced with
        Fisher-scaled linear coordinates for missing input variables.
    rank_repair_scale
        Scaling / linear-map rule for injected repair coordinates.
        ``"mean_whitened"`` (default) uses row ``j`` of the symmetric
        square-root of ``mean(Fs)`` to account for off-diagonal Fisher
        couplings. ``"median_diag"`` uses ``sqrt(median(Fs[:, j, j])) * Xj``;
        ``"mean_diag"`` uses ``sqrt(mean(Fs[:, j, j])) * Xj``; ``"unit"`` uses
        ``1.0 * Xj``.
    rank_repair_rtol, rank_repair_atol
        Relative and absolute tolerances for detecting row-rank deficiency and
        missing input columns.

    Returns
    -------
    new_expressions
        Regrouped expression strings.
    R
        Rotation matrix applied (identity if skipped / rejected).
    info
        Diagnostics.
    """
    rotation_kwargs = rotation_kwargs or {}

    if check_flattening_fn is None:
        check_flattening_fn = make_check_flattening_fn(X, Fs)

    atoms, C, _per_row = extract_atoms(expressions, expand=expand)
    if verbose:
        print(
            f"extract_atoms: {len(expressions)} rows, {len(atoms)} unique atoms, "
            f"||C||_1 = {np.abs(C).sum():.4f}"
        )

    ref = _flat_score(expressions, X, Fs, n_params, check_flattening_fn, A)
    if verbose:
        print(f"reference flatness score: {ref:.6f}")

    if method == "atoms":
        R, rot_info = optimize_sparse_rotation_atoms(
            C, verbose=verbose, **rotation_kwargs
        )
    elif method == "greedy":
        R, rot_info = greedy_givens_sparsify(C, verbose=verbose, **rotation_kwargs)
    elif method == "none":
        R = np.eye(len(expressions))
        rot_info = dict(l1_before=float(np.abs(C).sum()),
                        l1_after=float(np.abs(C).sum()))
    else:
        raise ValueError(f"Unknown method={method!r}")

    # Reconstruct with R applied.
    rotated = reconstruct_expressions(R, atoms, C,
                                      threshold=threshold,
                                      decimal=decimal)

    # For orthogonal R, the flatness is preserved.  For non-orthogonal R we
    # also update A so that check_flattening still sees flat Q.
    if not orthogonal:
        try:
            A_base = np.eye(n_params) if A is None else np.asarray(A)
            A_new_for_check = A_base @ np.linalg.inv(R)
        except np.linalg.LinAlgError:
            A_new_for_check = A
    else:
        A_new_for_check = A

    post = _flat_score(rotated, X, Fs, n_params,
                       check_flattening_fn, A_new_for_check)
    rel_delta = abs(post - ref) / max(ref, 1e-30)
    if verbose:
        print(
            f"post-rotation flatness: {post:.6f} "
            f"(rel Δ = {rel_delta:.2%}, tol = {flat_tol:.2%})"
        )

    info: Dict[str, Any] = dict(
        ref_flat=ref,
        post_flat=post,
        rel_delta=rel_delta,
        rotation_info=rot_info,
        n_atoms=len(atoms),
    )

    if rel_delta > flat_tol:
        if verbose:
            print("rotation rejected (flatness degraded too much); "
                  "falling back to input expressions.")
        info["rotation_accepted"] = False
        current = list(expressions)
        R = np.eye(len(expressions))
    else:
        info["rotation_accepted"] = True
        current = rotated

    if do_snap:
        if verbose:
            print("snapping shared coefficients...")
        current, snap_log = snap_shared_coefficients(
            current, X, Fs, n_params,
            rel_tol=snap_rel_tol,
            flat_tol=snap_flat_tol,
            A=A_new_for_check if info["rotation_accepted"] else A,
            check_flattening_fn=check_flattening_fn,
            verbose=verbose,
        )
        info["snap_log"] = snap_log
        info["final_flat"] = _flat_score(
            current, X, Fs, n_params, check_flattening_fn,
            A_new_for_check if info["rotation_accepted"] else A,
        )
        if verbose:
            print(f"final flatness: {info['final_flat']:.6f}")

    if repair_rank_deficiency:
        A_for_rank = A_new_for_check if info["rotation_accepted"] else A
        current, rank_info = repair_coordinate_rank_deficiency(
            current,
            X,
            Fs,
            n_params,
            check_flattening_fn=check_flattening_fn,
            A=A_for_rank,
            rank_repair_scale=rank_repair_scale,
            rank_rtol=rank_repair_rtol,
            rank_atol=rank_repair_atol,
            decimal=decimal,
            verbose=verbose,
        )
        info["rank_repair"] = rank_info
        if rank_info.get("repaired", False):
            info["final_flat"] = _flat_score(
                current, X, Fs, n_params, check_flattening_fn, A_for_rank,
            )
            if verbose:
                print(f"final flatness after rank repair: {info['final_flat']:.6f}")

    return current, R, info


# =============================================================================
# 9. SELF-TEST
# =============================================================================

if __name__ == "__main__":
    # Tiny smoke test: fabricate a 3-row expression set that shares atoms.
    # Rotation should concentrate X7 in one row.
    import sympy as sp
    X1, X2, X3, X4, X5, X6, X7 = sp.symbols("X1 X2 X3 X4 X5 X6 X7")
    eqs = [
        "0.139*X4",
        "-0.133*X3 + 0.116*X5 + 0.043*X7",
        "0.21*X1*X5 - 0.694*X1*exp(-0.875*X1) + 0.058*X7",
        "0.058*X2 + 0.141*X3 + 0.046*X7",
    ]
    atoms, C, _ = extract_atoms(eqs)
    print("atoms:")
    for a in atoms:
        print("  ", a)
    print("C:")
    print(np.round(C, 3))

    R, info = greedy_givens_sparsify(C, verbose=True)
    print("R:")
    print(np.round(R, 3))
    print("R @ C:")
    print(np.round(R @ C, 3))

    # Also try the nonlinear version (no flatness data, so just regroup).
    R2, info2 = optimize_sparse_rotation_atoms(
        C, n_outer=3, n_inner=200, verbose=True
    )
    print("R2 @ C:")
    print(np.round(R2 @ C, 3))
