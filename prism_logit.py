"""
PRISM-Logit: Interpretable Sequential Logistic Regression with
Automatic Transformation Discovery and Exact Deviance Attribution

A binary-classification extension of PRISM (Progressive Refinement with
Interpretable Sequential Modeling) that combines sequential transformation
discovery with exact pathwise deviance attribution under a Bernoulli-logit model.

Author: Gavin Symanowitz
Version: 1.0
Date: March 2026
Repository: https://github.com/gsymanowitz/prism-logit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    roc_auc_score, log_loss, accuracy_score, precision_score,
    recall_score, f1_score, brier_score_loss
)
import time
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class PRISMLogit:
    """
    PRISM-Logit: Interpretable Sequential Logistic Regression

    Automatically discovers optimal non-linear transformations for binary
    classification while maintaining full interpretability and exact
    pathwise deviance attribution.

    Parameters
    ----------
    m : int, default=10
        Number of coordinate descent iterations after each variable addition.
        Controls deviance attribution quality. m=10 achieves MR < 0.01%.
    alpha : float, default=0.05
        Significance level for LRT stopping rule and linear-unless-proven-otherwise
        threshold. Non-linear transforms must beat Linear at this significance level.
    interaction_penalty : float, default=5.0
        BIC complexity penalty multiplier for interactions (k×5 default).
    ridge_lambda : float, default=1e-4
        Ridge penalty on slope coefficients for numerical stability.
        Intercept is not penalised. Set to 0 for exact logistic equivalence
        in linear attribution mode.
    max_iterations : int, default=200
        Maximum coordinate descent iterations in Step 3.
    convergence_tolerance : float, default=1e-8
        Convergence criterion for D² change.
    top_k_screening : str or int, default='auto'
        Number of candidates to evaluate exactly in Stage B of Step 2.
        'auto' uses ceil(p/2) with minimum 3.

    Attributes
    ----------
    step1_results : dict
        Step 1 screening results (univariate D² per variable per transform).
    step2_results : dict
        Step 2 sequential selection results (selected variables, transforms, ΔD²).
    step3_results : dict
        Step 3 convergence results (final D², MR, coefficient stability).
    step4_results : dict
        Step 4 interaction testing results (tested/selected interactions).
    final_model : dict
        Complete model specification including attribution table.
    runtime : float
        Total fitting time in seconds.

    Examples
    --------
    >>> model = PRISMLogit(m=10)
    >>> model.fit(X_train, y_train, categorical=['sex', 'cp'])
    >>> probs = model.predict_proba(X_test)
    >>> model.evaluate(X_test, y_test)
    >>> attribution = model.get_deviance_attribution()
    """

    def __init__(self,
                 m: int = 10,
                 alpha: float = 0.05,
                 interaction_penalty: float = 5.0,
                 ridge_lambda: float = 1e-4,
                 max_iterations: int = 200,
                 convergence_tolerance: float = 1e-8,
                 top_k_screening: str = 'auto'):

        self.m = m
        self.alpha = alpha
        self.interaction_penalty = interaction_penalty
        self.ridge_lambda = ridge_lambda
        self.max_iterations = max_iterations
        self.convergence_tolerance = convergence_tolerance
        self.top_k_screening = top_k_screening

        # Results storage
        self.step1_results = None
        self.step2_results = None
        self.step3_results = None
        self.step4_results = None
        self.final_model = None
        self.runtime = None
        self.n_obs = None
        self.null_deviance = None
        self.categorical_features = set()

    # ==================================================================
    # Transformation Library
    # ==================================================================

    def _apply_transform(self, x: np.ndarray, transform_type: str) -> np.ndarray:
        """
        Apply parametric transformation to array.

        The seven transforms are: Linear, Logarithmic (sign-preserving),
        Sqrt (sign-preserving), Square, Cubic, Inverse, Exponential (dampened).
        """
        x = np.array(x, dtype=np.float64).flatten()

        if transform_type == 'Linear':
            result = x
        elif transform_type == 'Logarithmic':
            result = np.sign(x) * np.log1p(np.abs(x))
        elif transform_type == 'Sqrt':
            result = np.sign(x) * np.sqrt(np.abs(x))
        elif transform_type == 'Square':
            result = x ** 2
        elif transform_type == 'Cubic':
            result = x ** 3
        elif transform_type == 'Inverse':
            result = 1.0 / (x + 1.0)
        elif transform_type == 'Exponential':
            sigma = np.std(x) + 1.0
            result = np.exp(np.clip(x / sigma, -10, 10))
        else:
            raise ValueError(f"Unknown transform: {transform_type}")

        if np.any(~np.isfinite(result)):
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    def _get_transforms_for_feature(self, feat: str) -> List[str]:
        """Return applicable transforms: Linear only for categorical, all 7 for continuous."""
        if feat in self.categorical_features:
            return ['Linear']
        return ['Linear', 'Logarithmic', 'Sqrt', 'Square', 'Cubic', 'Inverse', 'Exponential']

    # ==================================================================
    # Logistic Helpers
    # ==================================================================

    @staticmethod
    def _sigmoid(eta: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))

    @staticmethod
    def _deviance(y: np.ndarray, p: np.ndarray) -> float:
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -2.0 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    def _null_deviance_calc(self, y: np.ndarray) -> float:
        return self._deviance(y, np.full_like(y, np.mean(y), dtype=np.float64))

    def _d_squared(self, y: np.ndarray, p: np.ndarray) -> float:
        return 1.0 - self._deviance(y, p) / self.null_deviance

    def _deviance_residuals(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        p = np.clip(p, 1e-15, 1 - 1e-15)
        d_i = -2.0 * (y * np.log(p) + (1 - y) * np.log(1 - p))
        return np.sign(y - p) * np.sqrt(np.abs(d_i))

    def _fit_logistic(self, X_matrix, y, max_iter=50, tol=1e-8):
        """Fit logistic regression via IRLS with ridge penalty. X_matrix has NO intercept column."""
        n = len(y)
        if X_matrix.ndim == 1:
            X_matrix = X_matrix.reshape(-1, 1)
        X_design = np.column_stack([np.ones(n), X_matrix])
        p_vars = X_matrix.shape[1]
        beta = np.zeros(X_design.shape[1])
        beta[0] = np.log(np.clip(np.mean(y), 1e-5, 1 - 1e-5) /
                         (1 - np.clip(np.mean(y), 1e-5, 1 - 1e-5)))

        lam = self.ridge_lambda
        R = lam * np.eye(p_vars + 1)
        R[0, 0] = 0.0  # no ridge on intercept

        for _ in range(max_iter):
            p = self._sigmoid(X_design @ beta)
            w = np.clip(p * (1 - p), 1e-10, None)
            z = X_design @ beta + (y - p) / w
            try:
                XtWX = X_design.T @ (w[:, None] * X_design) + R
                XtWz = X_design.T @ (w * z) - lam * beta
                XtWz[0] += lam * beta[0]  # undo ridge on intercept
                beta_new = np.linalg.solve(XtWX, XtWz)
            except np.linalg.LinAlgError:
                break
            if np.max(np.abs(beta_new - beta)) < tol:
                beta = beta_new
                break
            beta = beta_new

        p_pred = self._sigmoid(X_design @ beta)
        return {
            'intercept': beta[0],
            'coefficients': beta[1:],
            'beta_full': beta,
            'deviance': self._deviance(y, p_pred),
            'predictions': p_pred
        }

    # ==================================================================
    # Coordinate Descent: Single Global Intercept + Ridge
    # ==================================================================

    def _coordinate_descent_cycle(self, xtd, y, feats, coefs, intercept, n_cycles=1):
        """Cyclic block coordinate Newton/IRLS with single global intercept and ridge."""
        coefs = coefs.copy()
        b0 = intercept
        lam = self.ridge_lambda

        for _ in range(n_cycles):
            eta = b0 + sum(coefs[f] * xtd[f] for f in feats)
            p = self._sigmoid(eta)
            w = np.clip(p * (1 - p), 1e-10, None)

            # Update intercept (no ridge)
            b0 += np.sum(y - p) / np.sum(w)

            # Update each coefficient
            for feat in feats:
                eta = b0 + sum(coefs[f] * xtd[f] for f in feats)
                p = self._sigmoid(eta)
                w = np.clip(p * (1 - p), 1e-10, None)
                x_j = xtd[feat]
                grad = np.sum((y - p) * x_j) - lam * coefs[feat]
                hess = np.sum(w * x_j ** 2) + lam
                coefs[feat] += grad / hess

        return coefs, b0

    def _compute_predictions(self, xtd, feats, coefs, intercept):
        eta = intercept + sum(coefs[f] * xtd[f] for f in feats)
        return self._sigmoid(eta)

    # ==================================================================
    # Main Fit Method
    # ==================================================================

    def fit(self, X: pd.DataFrame, y, include_interactions: bool = True,
            verbose: bool = True, categorical: Optional[List[str]] = None):
        """
        Fit PRISM-Logit model to data.

        Parameters
        ----------
        X : pd.DataFrame
            Predictor variables (n_samples, n_features).
        y : array-like
            Binary response variable (0/1).
        include_interactions : bool, default=True
            Whether to test and include interaction terms.
        verbose : bool, default=True
            Whether to print progress messages.
        categorical : list of str, optional
            Column names of categorical/discrete predictors.
            These are restricted to Linear transforms only.

        Returns
        -------
        self : PRISMLogit
            Fitted model.
        """
        start_time = time.time()
        y = np.array(y, dtype=np.float64).flatten()
        assert set(np.unique(y)) <= {0.0, 1.0}, "Response must be binary (0/1)"

        self.n_obs = len(X)
        self.null_deviance = self._null_deviance_calc(y)
        self.categorical_features = set(categorical) if categorical else set()
        cc = np.bincount(y.astype(int))
        n_cat = len(self.categorical_features)
        n_cont = X.shape[1] - n_cat

        if verbose:
            print("=" * 70)
            print("PRISM-LOGIT: LOGISTIC REGRESSION ANALYSIS")
            print("=" * 70)
            print(f"\nDataset: n={len(X)}, p={X.shape[1]} "
                  f"({n_cont} continuous, {n_cat} categorical)")
            print(f"Class balance: {cc[0]} (0) / {cc[1]} (1) [{cc[1]/len(y):.1%} positive]")
            print(f"Null deviance: {self.null_deviance:.2f}")
            print(f"Parameters: m={self.m}, \u03b1={self.alpha}, "
                  f"\u03bb={self.ridge_lambda}, "
                  f"interactions={'Yes' if include_interactions else 'No'}")
            if n_cat > 0:
                print(f"Categorical (Linear only): {sorted(self.categorical_features)}")
            print()

        # Step 1: Initial transformation screening
        if verbose:
            print("STEP 1: TRANSFORMATION SCREENING")
            print("-" * 70)
        self.step1_results = self._step1_screening(X, y, verbose)

        # Step 2: Sequential selection with transformation retesting
        if verbose:
            print("\nSTEP 2: SEQUENTIAL SELECTION (LRT with linear-unless-proven-otherwise)")
            print("-" * 70)
        self.step2_results = self._step2_sequential_selection(X, y, verbose)

        # Step 3: Final convergence
        if verbose:
            print("\nSTEP 3: FINAL CONVERGENCE")
            print("-" * 70)
        self.step3_results = self._step3_convergence(X, y, verbose)

        # Step 4: Interaction testing (optional)
        if include_interactions and len(self.step2_results['selected_features']) >= 2:
            if verbose:
                print(f"\nSTEP 4: INTERACTION TESTING (BIC k\u00d7{self.interaction_penalty:.0f})")
                print("-" * 70)
            self.step4_results = self._step4_interactions(X, y, verbose)
        else:
            self.step4_results = {
                'interactions_tested': 0,
                'interactions_selected': [],
                'interaction_results': pd.DataFrame()
            }

        # Compile final model
        self._compile_final_model(X, y, include_interactions)

        self.runtime = time.time() - start_time
        if verbose:
            self._print_final_summary()

        return self

    # ==================================================================
    # Step 1: Initial Transformation Screening
    # ==================================================================

    def _step1_screening(self, X, y, verbose):
        chi2_nl = stats.chi2.ppf(1 - self.alpha, df=1)
        results = []

        for feat in X.columns:
            T = self._get_transforms_for_feature(feat)
            fits = {}
            for t in T:
                xt = self._apply_transform(X[feat].values, t)
                if np.std(xt) < 1e-10:
                    continue
                r = self._fit_logistic(xt, y)
                fits[t] = {'d2': 1.0 - r['deviance'] / self.null_deviance, 'dev': r['deviance']}

            if not fits:
                results.append({'Feature': feat, 'Transform': None, 'D\u00b2': 0.0,
                                'Deviance': self.null_deviance})
                continue

            best_t = max(fits, key=lambda t: fits[t]['d2'])
            best_d2, best_dev = fits[best_t]['d2'], fits[best_t]['dev']

            # Linear-unless-proven-otherwise
            if best_t != 'Linear' and 'Linear' in fits:
                lrt_vs_linear = fits['Linear']['dev'] - best_dev
                if lrt_vs_linear < chi2_nl:
                    best_t = 'Linear'
                    best_d2, best_dev = fits['Linear']['d2'], fits['Linear']['dev']

            results.append({'Feature': feat, 'Transform': best_t,
                            'D\u00b2': best_d2, 'Deviance': best_dev})

        df = pd.DataFrame(results).sort_values('D\u00b2', ascending=False)

        if verbose:
            print("\nInitial Screening Results (best univariate D\u00b2):\n")
            disp = df.copy()
            disp['D\u00b2'] = disp['D\u00b2'].apply(lambda x: f"{x * 100:.2f}%")
            disp['Deviance'] = disp['Deviance'].apply(lambda x: f"{x:.2f}")
            disp['Type'] = disp['Feature'].apply(
                lambda f: 'cat' if f in self.categorical_features else '')
            print(disp[['Feature', 'Transform', 'D\u00b2', 'Deviance', 'Type']].to_string(index=False))
            print()

        return {'results_df': df, 'feature_order': df['Feature'].tolist()}

    # ==================================================================
    # Step 2: Sequential Selection with Transformation Retesting
    # ==================================================================

    def _step2_sequential_selection(self, X, y, verbose):
        n = len(X)
        sel, td, coefs, xtd, rounds = [], {}, {}, {}, []
        intercept = 0.0
        cur_dev = self.null_deviance
        cur_d2 = 0.0
        cur_pred = np.full(n, np.mean(y))
        remaining = list(self.step1_results['feature_order'])
        chi2_crit = stats.chi2.ppf(1 - self.alpha, df=1)

        if verbose:
            print()

        rd = 0
        while remaining:
            rd += 1
            d2_before = cur_d2
            dr = self._deviance_residuals(y, cur_pred)

            # Stage A: Fast residual-based screening
            scores = []
            for feat in remaining:
                T = self._get_transforms_for_feature(feat)
                best_r2, best_t = -np.inf, None
                for t in T:
                    xt = self._apply_transform(X[feat].values, t)
                    if np.std(xt) < 1e-10:
                        continue
                    r2 = LinearRegression().fit(xt.reshape(-1, 1), dr).score(
                        xt.reshape(-1, 1), dr)
                    if r2 > best_r2:
                        best_r2, best_t = r2, t
                scores.append((feat, best_r2, best_t))
            scores.sort(key=lambda x: x[1], reverse=True)

            top_k = (min(len(remaining), max(3, int(np.ceil(len(remaining) / 2))))
                     if self.top_k_screening == 'auto'
                     else min(int(self.top_k_screening), len(remaining)))

            # Stage B: Exact LRT evaluation
            best_lrt, best_feat, best_trans, best_res = -np.inf, None, None, None
            linear_results = {}

            for feat, _, _ in scores[:top_k]:
                T = self._get_transforms_for_feature(feat)
                for t in T:
                    xt = self._apply_transform(X[feat].values, t)
                    if np.std(xt) < 1e-10:
                        continue
                    cols = [xtd[f] for f in sel] + [xt]
                    Xm = np.column_stack(cols) if len(cols) > 1 else xt.reshape(-1, 1)
                    res = self._fit_logistic(Xm, y)
                    lrt = cur_dev - res['deviance']
                    if t == 'Linear':
                        linear_results[feat] = {'lrt': lrt, 'res': res, 'dev': res['deviance']}
                    if lrt > best_lrt:
                        best_lrt, best_feat, best_trans, best_res = lrt, feat, t, res

            # Linear-unless-proven-otherwise check
            if (best_trans != 'Linear'
                    and best_feat not in self.categorical_features
                    and best_feat in linear_results):
                dev_linear = linear_results[best_feat]['dev']
                dev_nonlin = best_res['deviance']
                if dev_linear - dev_nonlin < chi2_crit:
                    best_trans = 'Linear'
                    best_lrt = linear_results[best_feat]['lrt']
                    best_res = linear_results[best_feat]['res']

            # Stopping rule
            if best_lrt < chi2_crit:
                if verbose:
                    print(f"Round {rd}: {best_feat} ({best_trans})")
                    print(f"  LRT \u03c7\u00b2 = {best_lrt:.2f} < {chi2_crit:.2f} (critical) \u2192 STOP")
                break

            # Add variable
            sel.append(best_feat)
            remaining.remove(best_feat)
            td[best_feat] = best_trans
            xtd[best_feat] = self._apply_transform(X[best_feat].values, best_trans)

            # Extract coefficients from full model fit
            intercept = best_res['intercept']
            for i, f in enumerate(sel):
                coefs[f] = best_res['coefficients'][i]

            # Incremental refinement
            if self.m > 0 and len(sel) > 1:
                coefs, intercept = self._coordinate_descent_cycle(
                    xtd, y, sel, coefs, intercept, self.m)

            cur_pred = self._compute_predictions(xtd, sel, coefs, intercept)
            cur_dev = self._deviance(y, cur_pred)
            cur_d2 = 1.0 - cur_dev / self.null_deviance
            d2g = cur_d2 - d2_before

            rounds.append({
                'Round': rd, 'Feature': best_feat, 'Transform': best_trans,
                'LRT \u03c7\u00b2': best_lrt, '\u0394D\u00b2': d2g, 'Cumulative D\u00b2': cur_d2
            })

            if verbose:
                cat_tag = " [cat]" if best_feat in self.categorical_features else ""
                print(f"Round {rd}: Added {best_feat} ({best_trans}){cat_tag}")
                print(f"  LRT \u03c7\u00b2 = {best_lrt:.1f}  |  "
                      f"\u0394D\u00b2 = {d2g * 100:.2f}%  |  "
                      f"Cum D\u00b2 = {cur_d2 * 100:.2f}%")

        if verbose:
            print()

        return {
            'selected_features': sel, 'transform_dict': td,
            'coefficients': coefs.copy(), 'intercept': intercept,
            'round_results': pd.DataFrame(rounds), 'final_d2': cur_d2,
            'final_deviance': cur_dev, 'X_transformed_dict': xtd
        }

    # ==================================================================
    # Step 3: Final Convergence
    # ==================================================================

    def _step3_convergence(self, X, y, verbose):
        sel = self.step2_results['selected_features']
        coefs = self.step2_results['coefficients'].copy()
        intercept = self.step2_results['intercept']
        xtd = self.step2_results['X_transformed_dict']
        s2c = coefs.copy()

        pred = self._compute_predictions(xtd, sel, coefs, intercept)
        d2 = self._d_squared(y, pred)
        conv_iters = 1

        for it in range(self.max_iterations):
            d2_old = d2
            coefs, intercept = self._coordinate_descent_cycle(
                xtd, y, sel, coefs, intercept, 1)
            pred = self._compute_predictions(xtd, sel, coefs, intercept)
            d2 = self._d_squared(y, pred)
            if abs(d2 - d2_old) < self.convergence_tolerance:
                conv_iters = it + 1
                break
        else:
            conv_iters = self.max_iterations

        mr = d2 - self.step2_results['final_d2']

        # Coefficient stability
        stab = []
        for f in sel:
            pct = abs((coefs[f] - s2c[f]) / s2c[f]) * 100 if abs(s2c[f]) > 1e-10 else 0
            stab.append({
                'Feature': f, 'Step2 Coef': s2c[f],
                'Step3 Coef': coefs[f], 'Change %': pct
            })

        if verbose:
            print(f"\nConvergence Summary:")
            print(f"  Starting D\u00b2 (end of Step 2): {self.step2_results['final_d2'] * 100:.4f}%")
            print(f"  Final D\u00b2 (after convergence): {d2 * 100:.4f}%")
            print(f"  Multivariate Refinement (MR): {mr * 100:.4f}%")
            print(f"  Iterations to convergence: {conv_iters}")
            if stab:
                sdf = pd.DataFrame(stab)
                print(f"  Coefficient stability: avg {sdf['Change %'].mean():.1f}%, "
                      f"max {sdf['Change %'].max():.1f}%")
            print()

        return {
            'final_d2': d2, 'final_deviance': self._deviance(y, pred),
            'mr': mr, 'convergence_iterations': conv_iters,
            'coefficients': coefs, 'intercept': intercept,
            'predictions': pred, 'coefficient_stability': pd.DataFrame(stab)
        }

    # ==================================================================
    # Step 4: Interaction Testing (BIC with k×5 penalty)
    # ==================================================================

    def _step4_interactions(self, X, y, verbose):
        sel = self.step2_results['selected_features']
        xtd = self.step2_results['X_transformed_dict']
        coefs = self.step3_results['coefficients'].copy()
        intercept = self.step3_results['intercept']
        T_all = ['Linear', 'Logarithmic', 'Sqrt', 'Square', 'Cubic', 'Inverse', 'Exponential']
        n = len(y)
        pred_base = self._compute_predictions(xtd, sel, coefs, intercept)
        dev_base = self._deviance(y, pred_base)
        bic_base = dev_base + (len(sel) + 1) * np.log(n)

        results, nt = [], 0
        for i, fj in enumerate(sel):
            for fk in sel[i + 1:]:
                nt += 1
                # Interactions use original untransformed variables
                zjk = X[fj].values * X[fk].values
                best_dbic, best_t, best_d2g = -np.inf, None, 0
                for t in T_all:
                    zt = self._apply_transform(zjk, t)
                    if np.std(zt) < 1e-10:
                        continue
                    Xm = np.column_stack([xtd[f] for f in sel] + [zt])
                    res = self._fit_logistic(Xm, y)
                    bic_new = res['deviance'] + (len(sel) + 1 + self.interaction_penalty * 2) * np.log(n)
                    dbic = bic_base - bic_new
                    if dbic > best_dbic:
                        best_dbic, best_t = dbic, t
                        best_d2g = (dev_base - res['deviance']) / self.null_deviance
                results.append({
                    'Interaction': f"{fj} \u00d7 {fk}", 'Feature_j': fj, 'Feature_k': fk,
                    'Transform': best_t, '\u0394D\u00b2': best_d2g, '\u0394BIC': best_dbic
                })

        idf = pd.DataFrame(results).sort_values('\u0394BIC', ascending=False)

        # Greedy sequential selection of interactions
        selected = []
        cur_dev, cur_bic = dev_base, bic_base
        cur_coefs, cur_intercept = coefs.copy(), intercept
        cur_xtd, cur_feats = xtd.copy(), list(sel)

        for _, row in idf.iterrows():
            if row['\u0394BIC'] <= 0:
                break
            zjk = X[row['Feature_j']].values * X[row['Feature_k']].values
            zt = self._apply_transform(zjk, row['Transform'])
            Xm = np.column_stack([cur_xtd[f] for f in cur_feats] + [zt])
            res = self._fit_logistic(Xm, y)
            bic_new = res['deviance'] + (len(cur_feats) + 1 + self.interaction_penalty * 2) * np.log(n)

            if cur_bic - bic_new > 0:
                nm = f"{row['Feature_j']}\u00d7{row['Feature_k']}"
                cur_feats.append(nm)
                cur_xtd[nm] = zt
                cur_intercept = res['intercept']
                for idx, f in enumerate(cur_feats):
                    cur_coefs[f] = res['coefficients'][idx]
                d2g = (cur_dev - res['deviance']) / self.null_deviance
                selected.append({
                    'feature_j': row['Feature_j'], 'feature_k': row['Feature_k'],
                    'transform': row['Transform'], 'd2_gain': d2g,
                    'delta_bic': cur_bic - bic_new, 'name': nm
                })
                cur_dev, cur_bic = res['deviance'], bic_new
            else:
                break

        if verbose:
            print(f"\nInteractions Tested: {nt}")
            print(f"Interactions Selected: {len(selected)}")
            if selected:
                print("\nSelected Interactions:")
                for s in selected:
                    print(f"  {s['feature_j']} \u00d7 {s['feature_k']} ({s['transform']})")
                    print(f"    \u0394D\u00b2 = {s['d2_gain'] * 100:.2f}%, \u0394BIC = {s['delta_bic']:.1f}")
            else:
                print("  None (all rejected by BIC penalty)")
            print()

        return {
            'interactions_tested': nt, 'interactions_selected': selected,
            'interaction_results': idf, 'final_coefficients': cur_coefs,
            'final_intercept': cur_intercept, 'final_features': cur_feats,
            'final_X_dict': cur_xtd
        }

    # ==================================================================
    # Compile Final Model
    # ==================================================================

    def _compile_final_model(self, X, y, include_interactions):
        sel = self.step2_results['selected_features']
        td = self.step2_results['transform_dict']
        xtd = self.step2_results['X_transformed_dict']
        has_int = include_interactions and len(self.step4_results['interactions_selected']) > 0

        if has_int:
            fc = self.step4_results['final_coefficients']
            fi = self.step4_results['final_intercept']
            ff = self.step4_results['final_features']
            fxd = self.step4_results['final_X_dict']
        else:
            fc = self.step3_results['coefficients']
            fi = self.step3_results['intercept']
            ff, fxd = list(sel), xtd

        pred = self._compute_predictions(fxd, ff, fc, fi)
        final_d2 = 1.0 - self._deviance(y, pred) / self.null_deviance

        # Build attribution table
        rows = []
        for _, r in self.step2_results['round_results'].iterrows():
            rows.append({
                'Feature': r['Feature'], 'Transform': r['Transform'],
                '\u0394D\u00b2': r['\u0394D\u00b2'], 'Cumulative D\u00b2': r['Cumulative D\u00b2']
            })
        rows.append({
            'Feature': 'Multivariate Refinement', 'Transform': '\u2014',
            '\u0394D\u00b2': self.step3_results['mr'],
            'Cumulative D\u00b2': self.step3_results['final_d2']
        })
        if has_int:
            cum = self.step3_results['final_d2']
            for s in self.step4_results['interactions_selected']:
                cum += s['d2_gain']
                rows.append({
                    'Feature': f"{s['feature_j']} \u00d7 {s['feature_k']}",
                    'Transform': s['transform'],
                    '\u0394D\u00b2': s['d2_gain'],
                    'Cumulative D\u00b2': cum
                })

        # Classification metrics
        yc = (pred >= 0.5).astype(int)
        metrics = {
            'D\u00b2': final_d2,
            'AUC-ROC': roc_auc_score(y, pred),
            'Log-loss': log_loss(y, pred),
            'Accuracy': accuracy_score(y, yc),
            'Precision': precision_score(y, yc, zero_division=0),
            'Recall': recall_score(y, yc, zero_division=0),
            'F1': f1_score(y, yc, zero_division=0),
            'Brier Score': brier_score_loss(y, pred)
        }

        # Logistic baseline for comparison
        bl = self._fit_logistic(X[sel].values, y)

        self.final_model = {
            'selected_features': sel,
            'transform_dict': td,
            'coefficients': fc,
            'intercept': fi,
            'interactions': self.step4_results['interactions_selected'],
            'attribution': pd.DataFrame(rows),
            'final_d2': final_d2,
            'base_d2': self.step3_results['final_d2'],
            'baseline_d2': 1.0 - bl['deviance'] / self.null_deviance,
            'baseline_auc': roc_auc_score(y, bl['predictions']),
            'metrics': metrics,
            'predictions': pred,
            'y_actual': y,
            'final_features': ff,
            'X_transformed_dict': fxd
        }

    # ==================================================================
    # Print Summary
    # ==================================================================

    def _print_final_summary(self):
        fm = self.final_model
        m = fm['metrics']

        print("=" * 70)
        print("FINAL MODEL SUMMARY")
        print("=" * 70)

        print("\nDEVIANCE ATTRIBUTION (D\u00b2):")
        print("-" * 70)
        a = fm['attribution'].copy()
        a['\u0394D\u00b2'] = a['\u0394D\u00b2'].apply(lambda x: f"{x * 100:.2f}%")
        a['Cumulative D\u00b2'] = a['Cumulative D\u00b2'].apply(lambda x: f"{x * 100:.2f}%")
        print(a.to_string(index=False))

        print("\n\nCLASSIFICATION METRICS:")
        print("-" * 70)
        print(f"  D\u00b2 (deviance explained):  {m['D\u00b2'] * 100:.2f}%")
        print(f"  AUC-ROC:                  {m['AUC-ROC']:.4f}")
        print(f"  Log-loss:                 {m['Log-loss']:.4f}")
        print(f"  Accuracy:                 {m['Accuracy'] * 100:.1f}%")
        print(f"  Precision:                {m['Precision'] * 100:.1f}%")
        print(f"  Recall:                   {m['Recall'] * 100:.1f}%")
        print(f"  F1 Score:                 {m['F1']:.4f}")
        print(f"  Brier Score:              {m['Brier Score']:.4f}")

        print("\n\nMODEL COMPARISON:")
        print("-" * 70)
        print(f"  Standard Logistic (linear):  D\u00b2={fm['baseline_d2'] * 100:.2f}%, "
              f"AUC={fm['baseline_auc']:.4f}")
        print(f"  PRISM-Logit (base model):    D\u00b2={fm['base_d2'] * 100:.2f}%")
        print(f"  PRISM-Logit (final):         D\u00b2={fm['final_d2'] * 100:.2f}%, "
              f"AUC={m['AUC-ROC']:.4f}")
        print(f"  Improvement over logistic:   "
              f"+{(fm['final_d2'] - fm['baseline_d2']) * 100:.2f} pp")

        print("\n\nFINAL MODEL PARAMETERS:")
        print("-" * 70)
        print(f"  Intercept (\u03b2\u2080): {fm['intercept']:.6f}")
        print()
        for f in fm['selected_features']:
            cat_tag = " [cat]" if f in self.categorical_features else ""
            print(f"  {f:20s} ({fm['transform_dict'][f]:12s}): "
                  f"\u03b2 = {fm['coefficients'][f]:.6f}{cat_tag}")

        if fm['interactions']:
            print("\n  Interactions:")
            for s in fm['interactions']:
                print(f"  {s['feature_j']} \u00d7 {s['feature_k']} ({s['transform']})")
                print(f"    \u0394D\u00b2 = {s['d2_gain'] * 100:.2f}%, "
                      f"\u0394BIC = {s['delta_bic']:.1f}")

        print(f"\n  Variables selected: {len(fm['selected_features'])}")
        print(f"  Interactions selected: {len(fm['interactions'])}")
        print(f"  Total parameters: {len(fm['selected_features']) + len(fm['interactions']) + 1}")
        print(f"  MR: {self.step3_results['mr'] * 100:.4f}%")
        print(f"  Runtime: {self.runtime:.2f}s")
        print("=" * 70)

    # ==================================================================
    # Prediction Methods
    # ==================================================================

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for new data.

        Parameters
        ----------
        X : pd.DataFrame
            Predictor variables.

        Returns
        -------
        proba : np.ndarray
            Predicted probability of class 1.
        """
        if self.final_model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        fm = self.final_model
        eta = fm['intercept'] + np.zeros(len(X))

        for f in fm['selected_features']:
            xt = self._apply_transform(X[f].values, fm['transform_dict'][f])
            eta += fm['coefficients'][f] * xt

        for s in fm['interactions']:
            zt = self._apply_transform(
                X[s['feature_j']].values * X[s['feature_k']].values, s['transform'])
            eta += fm['coefficients'][s['name']] * zt

        return self._sigmoid(eta)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels for new data.

        Parameters
        ----------
        X : pd.DataFrame
            Predictor variables.
        threshold : float, default=0.5
            Classification threshold.

        Returns
        -------
        labels : np.ndarray
            Predicted class labels (0 or 1).
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    # ==================================================================
    # Evaluation
    # ==================================================================

    def evaluate(self, X: pd.DataFrame, y, set_name: str = "Test") -> Dict:
        """
        Evaluate model on a dataset.

        Parameters
        ----------
        X : pd.DataFrame
            Predictor variables.
        y : array-like
            True binary labels.
        set_name : str, default="Test"
            Label for printout.

        Returns
        -------
        metrics : dict
            Dictionary of classification metrics.
        """
        y = np.array(y, dtype=np.float64).flatten()
        yp = self.predict_proba(X)
        yc = (yp >= 0.5).astype(int)
        d2 = 1.0 - self._deviance(y, yp) / self._null_deviance_calc(y)

        metrics = {
            'D\u00b2': d2,
            'AUC-ROC': roc_auc_score(y, yp),
            'Log-loss': log_loss(y, yp),
            'Accuracy': accuracy_score(y, yc),
            'Precision': precision_score(y, yc, zero_division=0),
            'Recall': recall_score(y, yc, zero_division=0),
            'F1': f1_score(y, yc, zero_division=0),
            'Brier Score': brier_score_loss(y, yp)
        }

        print(f"\n{set_name} Set Evaluation:")
        print("-" * 40)
        print(f"  D\u00b2:          {metrics['D\u00b2'] * 100:.2f}%")
        print(f"  AUC-ROC:     {metrics['AUC-ROC']:.4f}")
        print(f"  Log-loss:    {metrics['Log-loss']:.4f}")
        print(f"  Accuracy:    {metrics['Accuracy'] * 100:.1f}%")
        print(f"  Precision:   {metrics['Precision'] * 100:.1f}%")
        print(f"  Recall:      {metrics['Recall'] * 100:.1f}%")
        print(f"  F1:          {metrics['F1']:.4f}")
        print(f"  Brier Score: {metrics['Brier Score']:.4f}")

        return metrics

    # ==================================================================
    # Accessor Methods
    # ==================================================================

    def get_deviance_attribution(self) -> pd.DataFrame:
        """Get complete deviance attribution table."""
        if self.final_model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.final_model['attribution']

    def get_step1_results(self) -> pd.DataFrame:
        """Get Step 1 screening results."""
        if self.step1_results is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.step1_results['results_df']

    def get_step2_results(self) -> pd.DataFrame:
        """Get Step 2 sequential selection results."""
        if self.step2_results is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.step2_results['round_results']

    def get_step3_results(self) -> Dict:
        """Get Step 3 convergence results."""
        if self.step3_results is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return {
            'coefficient_stability': self.step3_results['coefficient_stability'],
            'convergence_iterations': self.step3_results['convergence_iterations'],
            'mr': self.step3_results['mr'],
            'final_d2': self.step3_results['final_d2']
        }

    def get_step4_results(self) -> pd.DataFrame:
        """Get Step 4 interaction testing results."""
        if self.step4_results is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self.step4_results['interaction_results']

    def get_model_parameters(self) -> Dict:
        """Get all model parameters for prediction."""
        if self.final_model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return {
            'selected_features': self.final_model['selected_features'],
            'transformations': self.final_model['transform_dict'],
            'coefficients': self.final_model['coefficients'],
            'intercept': self.final_model['intercept'],
            'interactions': self.final_model['interactions'],
            'final_d2': self.final_model['final_d2']
        }

    # ==================================================================
    # Visualization
    # ==================================================================

    def plot_results(self, X: pd.DataFrame, y=None, figsize=(15, 10)):
        """
        Create comprehensive visualization of results.

        Parameters
        ----------
        X : pd.DataFrame
            Predictor variables (used for x-axis values).
        y : array-like, optional
            True labels (for ROC curve if provided).
        figsize : tuple, default=(15, 10)
            Figure size.

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        if self.final_model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")

        fm = self.final_model
        n_features = len(fm['selected_features'])
        n_cols = 3
        n_rows = (n_features + 2) // n_cols + 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        # Plot each transformed variable's contribution to log-odds
        for idx, feat in enumerate(fm['selected_features']):
            ax = axes[idx]
            transform = fm['transform_dict'][feat]
            x_raw = X[feat].values
            x_transformed = self._apply_transform(x_raw, transform)

            sort_idx = np.argsort(x_raw)
            contribution = fm['coefficients'][feat] * x_transformed

            ax.plot(x_raw[sort_idx], contribution[sort_idx], 'b-', linewidth=2)
            ax.set_xlabel(feat)
            ax.set_ylabel('Log-odds contribution')
            ax.set_title(f"{feat} ({transform})")
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        # Attribution bar chart
        ax = axes[n_features]
        attr = fm['attribution']
        main_attr = attr[attr['Feature'] != 'Multivariate Refinement']
        if len(main_attr) > 0:
            features = main_attr['Feature'].values
            d2_vals = main_attr['\u0394D\u00b2'].values * 100
            colors = ['#2196F3' if '\u00d7' not in f else '#FF9800' for f in features]
            ax.barh(range(len(features)), d2_vals, color=colors)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize=8)
            ax.set_xlabel('\u0394D\u00b2 (%)')
            ax.set_title('Deviance Attribution')
            ax.invert_yaxis()

        # Hide unused subplots
        for idx in range(n_features + 1, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        return fig


# ==================================================================
# Convenience Functions
# ==================================================================

def fit_prism_logit(X: pd.DataFrame, y,
                    m: int = 10,
                    include_interactions: bool = True,
                    verbose: bool = True,
                    categorical: Optional[List[str]] = None,
                    ridge_lambda: float = 1e-4) -> PRISMLogit:
    """
    Convenience function to fit PRISM-Logit model.

    Parameters
    ----------
    X : pd.DataFrame
        Predictor variables.
    y : array-like
        Binary response variable (0/1).
    m : int, default=10
        Coordinate descent iterations per variable addition.
    include_interactions : bool, default=True
        Whether to test interaction terms.
    verbose : bool, default=True
        Print progress.
    categorical : list of str, optional
        Column names of categorical predictors.
    ridge_lambda : float, default=1e-4
        Ridge penalty for numerical stability.

    Returns
    -------
    model : PRISMLogit
        Fitted model.
    """
    model = PRISMLogit(m=m, ridge_lambda=ridge_lambda)
    model.fit(X, y, include_interactions=include_interactions,
              verbose=verbose, categorical=categorical)
    return model


print("\u2713 PRISM-Logit v1.0 loaded")
