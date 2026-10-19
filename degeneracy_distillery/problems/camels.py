"""CAMELS SB35 adapter for the one-step sweep driver.

Dataset-backed and fixed-size: 1024 LH simulations, 35 parameters, 344
scaling-relation summary features compressed by PCA. There is no box prior
and no closed-form coordinate, so ``truth_coords`` is None and the gates
only check that the ladder returned something and that the ensemble agrees
on the rank.

Two things differ from the standalone ``new_idea/camels_oneshot.py`` this
replaces:

* screening is the driver's :func:`whittle_ladder` with ``rank_rule="info"``,
  which counts axes by *held-out information* rather than by prior spread;
* the ensemble draws a different ``eta(x)`` width, depth and activation per
  member (:func:`estimator_bank`), the way the Fishnets ensemble does, so
  ``y_std`` measures architecture disagreement and not just seed noise.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from degeneracy_distillery.oneshot import estimator_bank, estimator_factory_for

OBSERVABLES = (
    "MBH_Mh_s61", "MBH_Mh_s90", "Mg_Mh_s61", "Mg_Mh_s90",
    "Ms_Mh_s61", "Ms_Mh_s90", "Rs_Ms_s61", "Rs_Ms_s90",
    "SFRH", "SFRH_100Myr", "SFR_Ms_s61", "SFR_Ms_s90",
    "Zs_Ms_s61", "Zs_Ms_s90",
)
PARAM_NAMES = (
    "Omega_m", "sigma_8", "A_SN1", "A_AGN1", "A_SN2", "A_AGN2", "Omega_b",
    "H_0", "n_s", "MaxSfrTimescale", "FactorForSofterEQS", "IMFslope",
    "SNII_MinMass", "ThermalWindFraction", "VariableWindSpecMomentum",
    "WindFreeTravelDensFac", "MinWindVel", "WindEnergyReductionFactor",
    "WindEnergyReductionMetallicity", "WindEnergyReductionExponent",
    "WindDumpFactor", "SeedBlackHoleMass", "BH_AccretionFactor",
    "BH_EddingtonFactor", "BH_FeedbackFactor", "BH_RadiativeEfficiency",
    "QuasarThreshold", "QuasarThresholdPower", "UVB_H0_beta", "UVB_H0_Deltaz",
    "UVB_Hep_beta", "UVB_Hep_Deltaz", "SNIa_Rate_Norm", "SNIa_Rate_DTD_power",
    "SofteningLength",
)
N_SIMS = 1024
N_PARAMS = 35
DEFAULT_DATA = "data_scratch/data_L50_TNG_v3.hdf5"

#: theta is min-max scaled into this range, so ``use_log_features`` is safe.
THETA_RANGE = (1.0, 2.0)

#: eta(x) trunk for the ladder and for ensemble member 0.
EST_FEATURES = (256, 256, 128)


class CamelsProblem:
    """CAMELS SB35, all 35 parameters, no pre-selection.

    ``n_pca`` PCA coefficients of the standardised 344 features are the
    data vector. Features are standardised *before* PCA because the raw
    columns span ~2 orders of magnitude in scale, and unstandardised PCA
    would then track the loudest scaling relation rather than the
    informative directions.
    """

    def __init__(
        self,
        data_path: str | Path = DEFAULT_DATA,
        n_pca: int = 40,
        seed: int = 0,
        test_size: float = 0.2,
        m_probe: int = 12,
        prior_draw: str = "uniform",
        **_ignored,
    ):
        self.name = "camels"
        if prior_draw not in ("uniform", "empirical"):
            raise ValueError(f"prior_draw must be uniform|empirical, got {prior_draw!r}")
        self.prior_draw = prior_draw
        self.d = N_PARAMS
        self.param_names = PARAM_NAMES
        self.use_log_features = True
        self.m_probe = int(m_probe)
        self.expected_rank = None
        self.n_pca = int(n_pca)
        self.seed = int(seed)
        self.data_path = Path(data_path)
        self.theta_lo = np.full(self.d, THETA_RANGE[0], dtype=np.float64)
        self.theta_hi = np.full(self.d, THETA_RANGE[1], dtype=np.float64)
        self._load(float(test_size))

    # ---------------------------------------------------------------- data
    def _load(self, test_size: float) -> None:
        import h5py
        from sklearn.decomposition import PCA
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        from degeneracy_distillery.sr_utils import fit_theta_scaler

        with h5py.File(self.data_path, "r") as f:
            P = np.asarray(f["Parameters"][...], dtype=np.float32)
            blocks = []
            for name in OBSERVABLES:
                a = np.asarray(f[name][...], dtype=np.float32)
                blocks.append(a[:, :N_SIMS].T if a.shape[0] != N_SIMS else a[:N_SIMS])
        theta_raw = P[:, :N_SIMS].T if P.shape[0] == N_PARAMS else P[:N_SIMS]
        data_raw = np.nan_to_num(
            np.concatenate(blocks, axis=1), nan=0.0, posinf=0.0, neginf=0.0
        )

        th_tr, th_te, x_tr, x_te = train_test_split(
            theta_raw, data_raw, test_size=test_size, random_state=self.seed
        )
        fs = StandardScaler().fit(x_tr)
        if int(self.n_pca) <= 0:
            # Raw standardised features, as the three-step pipeline uses
            # (camels_notebook_run.py feeds all 344 to Fishnets). Not the
            # same as full-rank PCA: that is capacity-equivalent for a Dense
            # first layer but leaves components with wildly unequal variance,
            # which changes optimisation conditioning.
            pca = None
            self.n_pca = 0
            self.cumvar = 1.0
        else:
            n_pca = int(min(self.n_pca, x_tr.shape[0] - 1, x_tr.shape[1]))
            pca = PCA(
                n_components=n_pca, svd_solver="randomized",
                random_state=self.seed, iterated_power=4,
            ).fit(fs.transform(x_tr))
            self.n_pca = n_pca
            self.cumvar = float(pca.explained_variance_ratio_.cumsum()[-1])

        scaler = fit_theta_scaler(th_tr, feature_range=THETA_RANGE)
        self.theta_scaler = scaler
        self.th_tr = scaler.transform(th_tr).astype(np.float64)
        self.th_te = scaler.transform(th_te).astype(np.float64)
        def _embed(a):
            z = fs.transform(a)
            return (z if pca is None else pca.transform(z)).astype(np.float64)

        self.x_tr = _embed(x_tr)
        self.x_te = _embed(x_te)
        self.x_dim = int(self.x_tr.shape[1])
        # Per-parameter empirical quantile function for ``sample_prior``.
        self._prior_quantiles = np.sort(self.th_tr, axis=0)
        self._cursor = 0

    def _draw(self, n: int, rng: np.random.Generator):
        """Serve ``n`` pairs from the train split, without replacement.

        The dataset is fixed at 1024 rows, so a request for more than the
        split holds returns the whole split rather than silently
        duplicating rows into both the fit and validation sets.
        """
        pool = int(self.th_tr.shape[0])
        n = int(n)
        if n >= pool:
            return self.th_tr, self.x_tr
        idx = rng.choice(pool, size=n, replace=False)
        return self.th_tr[idx], self.x_tr[idx]

    def sample(self, n: int, rng: np.random.Generator):
        return self._draw(n, rng)

    def sample_prior(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Fresh theta for SR augmentation. Uniform on the scaled box by default.

        This is a design choice for a function fit, not a statistical
        prior: SR is approximating the already-fitted map eta(theta), so
        we sample wherever we want the expression to be accurate. Uniform
        on ``[1, 2]^35`` gives even coverage of the box.

        What it must *not* do is resample ``th_tr``: 20000 draws from 819
        rows is 819 unique theta repeated ~24 times, which adds no
        coverage and leaks theta across the shuffled SR train/test split.

        ``prior_draw="empirical"`` instead draws each parameter
        independently by inverse-CDF interpolation of its own training
        quantiles. The LH design is near-independent (max |rho| = 0.062),
        but 19 of 35 marginals are non-uniform on the scaled box because
        CAMELS samples them log-uniformly upstream, so uniform sampling
        puts a median 11% of draws per coordinate (31% for
        QuasarThreshold) outside the training 5-95% range -- still inside
        the support, just sparser. Use it when you want the fit weighted
        toward where the map is best constrained rather than across the
        whole box.
        """
        n = int(n)
        if self.prior_draw == "empirical":
            qs = self._prior_quantiles       # (n_train, d), sorted per column
            grid = np.linspace(0.0, 1.0, qs.shape[0])
            u = rng.random((n, self.d))
            out = np.empty((n, self.d), dtype=np.float64)
            for j in range(self.d):
                out[:, j] = np.interp(u[:, j], grid, qs[:, j])
            return out
        return rng.uniform(
            self.theta_lo, self.theta_hi, size=(n, self.d)
        ).astype(np.float64)

    def held_out(self) -> tuple[np.ndarray, np.ndarray]:
        """The 20% test split, for the ladder's validation arm."""
        return self.th_te, self.x_te

    # ------------------------------------------------------------ networks
    def estimator_factory(self, m: int):
        return estimator_factory_for(EST_FEATURES, "gelu")(m)

    def estimator_factories(self, k: int, seed: Optional[int] = None) -> list:
        """``k`` eta(x) factories varying width, depth and activation.

        Member 0 is pinned to ``EST_FEATURES`` so the ladder map can be
        reused warm as the ensemble reference.
        """
        return estimator_bank(
            int(k),
            seed=self.seed if seed is None else int(seed),
            hids=(64, 320),
            n_layers=(2, 4),
            first=EST_FEATURES,
        )

    # -------------------------------------------------------------- scoring
    def truth_coords(self, theta: np.ndarray) -> None:
        return None

    def gates(self, payload: dict) -> dict[str, bool]:
        r_hat = int(payload.get("r_hat", -1))
        agree = payload.get("jtj_rank_agreement_k")
        k = payload.get("n_ensemble")
        out = {"rank_found": bool(1 <= r_hat <= self.m_probe)}
        if agree is not None and k:
            out["ensemble_rank_majority"] = bool(int(agree) > int(k) // 2)
        return out

    def sr_hints(self) -> dict:
        return {
            "allowed_symbols": "add,mul,div,pow,constant,variable,square,exp",
            "max_length": 30,
            "max_depth": 6,
        }
