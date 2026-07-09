"""
Explicit polynomial-basis PDS baselines.

Motivation. PDS-LASSO is linear only in the columns it is given; handed a
basis that already contains the relevant nonlinear terms, it can represent
nonlinear nuisance functions. The real limitation is that it selects from a
fixed candidate set rather than constructing the basis. These baselines make
that point empirically: PDS-LASSO run on x augmented with all squares and
pairwise interactions (degree 2) and, optionally, all degree-3 monomials.

The contrast with SR-PDS is the argument:
  * degree-2 contains x4*x5 and so handles DGP-6 (mild), but cannot contain
    the cubic x1^2 x2 and so still fails on DGP-7 (severe);
  * degree-3 does contain the cubic, but at p=50 that is ~23k columns -- the
    penalty rises with log(p*), variance inflates, and the basis is
    impractical at large n. Enriching the hand-built dictionary is not free.

SR-PDS searches a structured space of log-cardinality ~200 instead of
enumerating the monomials, recovering the same terms by name at a fraction
of the dimension.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from estimators import pds_lasso


def _expand(X, degree):
    """x -> all monomials up to `degree` (no bias), constant columns dropped,
    standardised so the LASSO penalty applies evenly across mixed scales.

    Standardising the controls does not change beta_hat on d or its standard
    error; it only affects which controls the LASSO selects.
    """
    Z = PolynomialFeatures(degree=degree, interaction_only=False,
                           include_bias=False).fit_transform(X)
    Z = Z[:, Z.std(axis=0) > 1e-10]      # e.g. squares of +/-1 dummies are constant
    Z = StandardScaler().fit_transform(Z)
    return Z


def pds_lasso_poly2(X, d, y, n, p):
    """PDS-LASSO on the degree-2 basis (squares + pairwise interactions)."""
    Z = _expand(X, 2)
    return pds_lasso(Z, d, y, n, Z.shape[1])


def pds_lasso_poly3(X, d, y, n, p):
    """PDS-LASSO on the full degree-3 basis (~23k columns at p=50).

    Memory-heavy: a 10000 x 23425 design matrix is ~1.9 GB. Cap the sample
    size when running this across a trajectory.
    """
    Z = _expand(X, 3)
    return pds_lasso(Z, d, y, n, Z.shape[1])


POLY_REGISTRY = {
    'pds_poly2': {'fn': pds_lasso_poly2, 'label': 'PDS-LASSO (poly-2)',
                  'requires_serial': False},
    'pds_poly3': {'fn': pds_lasso_poly3, 'label': 'PDS-LASSO (poly-3)',
                  'requires_serial': False},
}


# --- recovery analysis for the polynomial baselines ---------------------
#
# Does the polynomial basis actually *select* the nonlinear confounder it
# contains? This quantifies the basis-construction argument: degree 2 selects
# x4*x5 but cannot represent the cubic x1^2 x2; degree 3 can represent it, at a
# much larger dictionary. Mirrors the pds_lasso selection (same theoretical
# lambda, union over the y- and d-equation LASSO) so the rates correspond to
# what the baseline actually does.

import numpy as _np
from sklearn.preprocessing import PolynomialFeatures as _PF
from sklearn.preprocessing import StandardScaler as _SS
from sklearn.linear_model import Lasso as _Lasso
from estimators import _theoretical_lambda as _lam


def _poly_term_name(term):
    """Truth-term tuple -> PolynomialFeatures column name (None if non-polynomial)."""
    kind, i, j = term
    if kind == 'sq':    return f'x{i}^2'
    if kind == 'int':   return f'x{i} x{j}'      # i < j
    if kind == 'cubic': return f'x{i}^2 x{j}'    # x_i^2 * x_j
    return None  # log / sqrt are not in any polynomial basis


def _expand_named(X, degree):
    """Degree-`degree` basis with column names, constants dropped, standardised."""
    pf = _PF(degree=degree, interaction_only=False, include_bias=False).fit(X)
    names = _np.array(pf.get_feature_names_out([f'x{i}' for i in range(X.shape[1])]))
    Z = pf.transform(X)
    keep = Z.std(axis=0) > 1e-10
    Z, names = Z[:, keep], names[keep]
    Z = _SS().fit_transform(Z)
    return Z, names


def poly_selection(X, d, y, n, degree, truth_terms):
    """For each truth term, whether the degree-`degree` basis represents it and
    whether the PDS-LASSO union selects it on this dataset.
    """
    Z, names = _expand_named(X, degree)
    name_set = set(names)
    lam = _lam(n, Z.shape[1])
    sel = (set(_np.where(_Lasso(alpha=lam, max_iter=10000).fit(Z, y).coef_ != 0)[0]) |
           set(_np.where(_Lasso(alpha=lam, max_iter=10000).fit(Z, d).coef_ != 0)[0]))
    sel_names = {names[i] for i in sel}
    out = {}
    for t in truth_terms:
        nm = _poly_term_name(t)
        out[t] = {'representable': bool(nm in name_set),
                  'selected': bool(nm is not None and nm in sel_names)}
    return out
