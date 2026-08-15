# Rank-based location inference

Specification of two rank procedures and the location inference built on them:

  - the **one-sample procedure**, the Wilcoxon signed rank test, for a single sample of
    differences;
  - the **two-sample procedure**, the Wilcoxon rank sum test, equivalently the
    Mann-Whitney ``U`` test, for two independent samples.

For each: the statistic (§2.1, §3.1), its exact null distribution and the normal
approximation to it (§2.2–§2.4, §3.2–§3.4), and the p-value (§2.5, §3.5). Then, shared by
both, the contrast sets the location inference is built on (§4), the Hodges-Lehmann point
estimate (§5), and the confidence interval obtained by inverting either test (§6).

§9 gives worked values for checking an implementation. §10 maps the specification onto
this package.

## 1. Notation

| | |
|---|---|
| ``d_1, \dots, d_N`` | the one-sample input; for paired data ``d_i = x_i - y_i`` |
| ``n`` | the number of ``d_i`` with ``d_i \ne 0`` |
| ``x_1, \dots, x_{n_x}``, ``y_1, \dots, y_{n_y}`` | the two-sample inputs |
| ``N`` | the number of observations: ``N`` one-sample, ``N = n_x + n_y`` two-sample |
| ``\alpha`` | two-sided error rate; the interval has nominal coverage ``1-\alpha`` |
| ``z_q`` | the ``q`` quantile of the standard normal distribution |
| ``\Phi`` | the standard normal CDF |
| ``m`` | the size of the contrast set of §4 |
| ``V_{(1)} \le \dots \le V_{(m)}`` | that contrast set, sorted |

**Midranks.** Where ranks are taken, tied values receive the average of the ranks they
span. For a vector ``v``, write

```math
T(v) = \sum_{g} (t_g^3 - t_g)
```

where ``t_g`` is the multiplicity of the ``g``-th group of equal values in ``v``. Groups
of size one contribute nothing, so ``T(v) = 0`` exactly when ``v`` has no ties. ``T`` is
the only function of the tie pattern either normal approximation uses.

## 2. The one-sample procedure

### 2.1 Model, estimand, statistic

**Model.** The ``d_i`` are i.i.d. from a continuous distribution ``F``. The null
hypothesis is that ``F`` is symmetric about ``0``.

**Estimand.** The **pseudomedian** of ``F``: the median of ``(d + d')/2`` for independent
``d, d' \sim F``. If ``F`` is symmetric about ``\theta`` then the pseudomedian is
``\theta``, and coincides with the median of ``F`` and, where it exists, with the mean.
For asymmetric ``F`` the pseudomedian is a different functional from the median, and it
is the pseudomedian that §5 and §6 estimate.

**Zeros.** Observations with ``d_i = 0`` are discarded before ranking; a zero has no
sign to rank. All of §2 and all of §4–§6 are computed from the ``n`` retained
observations.

**Statistic.** With ``R_i`` the midrank of ``|d_i|`` among the ``n`` retained absolute
values,

```math
W^+ = \sum_{i \,:\, d_i > 0} R_i .
```

Its support is ``\{0, 1, \dots, n(n+1)/2\}``. Under the null it is symmetric about
``n(n+1)/4``, with

```math
\mathbb{E}[W^+] = \frac{n(n+1)}{4}, \qquad
\operatorname{Var}(W^+) = \frac{n(n+1)(2n+1)}{24} - \frac{T(|d|)}{48} .
```

### 2.2 Exact null distribution, no ties

Under the null the signs of the ``d_i`` are independent, each ``\pm`` with probability
``1/2``, and independent of ``|d|``. With no ties the midranks are the integers
``1, \dots, n``, so

```math
W^+ \;\overset{d}{=}\; \sum_{i=1}^{n} i \, B_i , \qquad B_i \overset{\text{iid}}{\sim} \mathrm{Bernoulli}(1/2) .
```

Let ``c_n(w) = \#\{ S \subseteq \{1,\dots,n\} : \sum_{i \in S} i = w \}``. Then

```math
c_n(w) = c_{n-1}(w) + c_{n-1}(w - n), \qquad
c_0(0) = 1, \quad c_0(w) = 0 \ (w \ne 0), \quad c_n(w) = 0 \ (w < 0),
```

and ``P(W^+ = w) = c_n(w) / 2^n``. Evaluating the recursion over the whole support is
``O(n^3)`` in time and ``O(n^2)`` in space. Write ``F_n(w) = P(W^+ \le w)``.

!!! warning "Numerical care"
    ``c_n(w)`` grows like ``2^n``, so accumulating the counts and dividing at the end
    overflows fixed-width integers for moderate ``n`` — at ``n = 72`` in 64 bits. An
    implementation should carry the normalisation inside the recursion, propagating
    probabilities rather than counts, or use exact arithmetic.

### 2.3 Exact null distribution, ties present

With ties the midranks are not ``1, \dots, n`` and the recursion above does not apply.
The sign-symmetry argument still does: conditionally on the observed multiset of
midranks, the ``2^n`` sign assignments are equally likely under the null. The exact
conditional null distribution is therefore obtained by enumerating them,

```math
P(W^+ \le w) = 2^{-n} \, \#\Bigl\{ \varepsilon \in \{0,1\}^n : \textstyle\sum_i \varepsilon_i R_i \le w \Bigr\} ,
```

at cost ``O(2^n)``. This is a conditional distribution given the tie pattern, which is
what makes it exact.

### 2.4 Normal approximation

``W^+`` is asymptotically normal with the mean and variance of §2.1. Write

```math
\mu = W^+ - \frac{n(n+1)}{4}, \qquad
\sigma = \sqrt{\frac{n(n+1)(2n+1)}{24} - \frac{T(|d|)}{48}}
```

for the centred statistic and the tie-corrected standard deviation. The variance
correction is exact under the conditional distribution of §2.3, not an approximation to
it.

### 2.5 p-values

Exact, no ties:

```math
p_{\text{left}} = F_n(W^+), \qquad
p_{\text{right}} = 1 - F_n(W^+ - 1), \qquad
p_{\text{both}} = \begin{cases}
  2 F_n(W^+) & W^+ \le n(n+1)/4 \\
  2\bigl(1 - F_n(W^+ - 1)\bigr) & \text{otherwise.}
\end{cases}
```

The two-sided value needs no clipping: the branch is selected by which side of the null
mean ``W^+`` falls on, so the doubled tail is never more than ``1``.

Exact, ties present: with ``\ell`` and ``g`` the proportions of sign assignments giving
``W' \le W^+`` and ``W' \ge W^+`` respectively,

```math
p_{\text{left}} = \ell, \qquad p_{\text{right}} = g, \qquad
p_{\text{both}} = \min\bigl(1,\, 2\min(\ell, g)\bigr) .
```

Normal approximation, with a continuity correction of ``1/2``:

```math
p_{\text{left}} = \Phi\!\left(\frac{\mu + 1/2}{\sigma}\right), \qquad
p_{\text{right}} = 1 - \Phi\!\left(\frac{\mu - 1/2}{\sigma}\right), \qquad
p_{\text{both}} = 2\left[1 - \Phi\!\left(\frac{\bigl|\mu - \tfrac{1}{2}\operatorname{sign}\mu\bigr|}{\sigma}\right)\right] .
```

Degenerate cases: if ``n = 0`` — every difference was zero — all three exact p-values are
``1``. If ``\mu = \sigma = 0``, all three approximate p-values are ``1``.

## 3. The two-sample procedure

### 3.1 Model, estimand, statistic

**Model.** The two samples are independent, i.i.d. from continuous ``F_x`` and ``F_y``.
The null hypothesis is ``F_x = F_y``.

**Estimand.** Under the **shift model** ``F_x(t) = F_y(t - \Delta)``, the estimand is
``\Delta``. Without that assumption the test is one of ``P(X > Y) = 1/2``, and §5
estimates the median of ``X - Y`` for independent ``X \sim F_x``, ``Y \sim F_y``. That
quantity is not in general ``\operatorname{median}(F_x) - \operatorname{median}(F_y)``.
Zero observations carry no special meaning here and are not discarded.

**Statistic.** Rank the pooled sample of size ``N``. With ``R_i`` the midrank of ``x_i``,

```math
U = \sum_{i=1}^{n_x} R_i - \frac{n_x(n_x+1)}{2}
  = \#\{(i,j) : x_i > y_j\} + \tfrac{1}{2} \#\{(i,j) : x_i = y_j\} .
```

The two forms are algebraically identical. The first is ``O(N \log N)`` via a sort; the
second is the definition the inversion of §6.2 uses. The support is
``\{0, \tfrac{1}{2}, \dots, n_x n_y\}``, integer-valued absent ties, symmetric under the
null about ``n_x n_y / 2``, with

```math
\mathbb{E}[U] = \frac{n_x n_y}{2}, \qquad
\operatorname{Var}(U) = \frac{n_x n_y}{12}\left(N + 1 - \frac{T([x; y])}{N(N-1)}\right) .
```

### 3.2 Exact null distribution, no ties

Under the null all ``\binom{N}{n_x}`` assignments of pooled ranks to the two samples are
equally likely. Let ``c_{n_x, n_y}(u)`` count those giving ``U = u``. Conditioning on
whether the largest pooled observation belongs to ``x`` or to ``y``,

```math
c_{n_x, n_y}(u) = c_{n_x - 1,\, n_y}(u - n_y) + c_{n_x,\, n_y - 1}(u) ,
```

with ``c_{n_x, 0}(0) = c_{0, n_y}(0) = 1``, and ``c(u) = 0`` for ``u < 0`` or
``u > n_x n_y``. Then ``P(U = u) = c_{n_x,n_y}(u) / \binom{N}{n_x}``. Evaluating over the
support is ``O\bigl((n_x n_y)^2\bigr)`` in time and ``O(n_x n_y)`` in space. Write
``G_{n_x,n_y}(u) = P(U \le u)``.

The numerical caveat of §2.2 applies with more force: the normalising constant
``\binom{N}{n_x}`` exceeds ``2^{63}`` for balanced samples from ``n_x = n_y = 34``, where
``\binom{68}{34} \approx 2.85 \times 10^{19}``.

### 3.3 Exact null distribution, ties present

As in §2.3, the exact conditional null distribution is obtained by enumeration —
here over the ``\binom{N}{\min(n_x, n_y)}`` assignments of the observed midranks to the
smaller sample — and the same three p-value formulas follow.

### 3.4 Normal approximation

``U`` is asymptotically normal. Write ``\mu = U - n_x n_y / 2`` for the centred statistic
and ``\sigma`` for the square root of the variance in §3.1, which carries the tie
correction.

### 3.5 p-values

Exact, no ties:

```math
p_{\text{left}} = G(U), \qquad
p_{\text{right}} = 1 - G(U - 1), \qquad
p_{\text{both}} = \min\bigl(1,\, 2\, G(\min(U,\, n_x n_y - U))\bigr) .
```

The two-sided form folds to the lower tail by the null symmetry of §3.1, and is clipped
because the fold is exact only away from the centre.

Exact under ties, and the normal approximation: exactly as in §2.5, with ``\ell, g`` from
§3.3 and ``\mu, \sigma`` from §3.4.

## 4. Contrast sets

### 4.1 Definitions

**One sample — the Walsh averages.** The ``m = n(n+1)/2`` values

```math
A_{ij} = \frac{d_i + d_j}{2}, \qquad 1 \le i \le j \le n ,
```

over the ``n`` retained observations. The diagonal ``i = j`` is included, so each ``d_i``
is itself a member.

**Two samples — the cross-group differences.** The ``m = n_x n_y`` values

```math
D_{ij} = x_i - y_j , \qquad 1 \le i \le n_x, \ 1 \le j \le n_y .
```

### 4.2 The counting identity

Let ``W^+(\theta)`` denote the statistic of §2.1 recomputed on ``d_1 - \theta, \dots,
d_n - \theta``, and ``U(\Delta)`` the statistic of §3.1 recomputed on ``x_1 - \Delta,
\dots`` against ``y``. Then

```math
W^+(\theta) = \#\{(i,j) : i \le j, \ A_{ij} > \theta\} + \tfrac{1}{2}\,\#\{(i,j) : i \le j, \ A_{ij} = \theta\} ,
```
```math
U(\Delta) = \#\{(i,j) : D_{ij} > \Delta\} + \tfrac{1}{2}\,\#\{(i,j) : D_{ij} = \Delta\} .
```

The half-count is the midrank convention, and it vanishes whenever ``\theta`` is not
itself a member of the contrast set — which is almost surely the case under continuous
``F``, and is the only case §6.2 needs. Both functions are non-increasing in their
argument and change value only at members of the contrast set.

*Proof of the first, for ``\theta`` not a contrast and ``|d|`` untied.*
``R_i = \#\{j : |d_j| \le |d_i|\}``, so
``W^+ = \sum_{i : d_i > 0} \#\{j : |d_j| \le |d_i|\}``. A pair ``\{i, j\}`` is counted
exactly when the one with the larger absolute value is positive, which is exactly when
``d_i + d_j > 0``; the diagonal term ``\{i,i\}`` is counted exactly when ``d_i > 0``.
Applying this to ``d - \theta`` gives the statement. The second is immediate from the
counting form of ``U`` in §3.1. ∎

Everything in §5 and §6 is a consequence of this identity. It is the reason the estimator
and the interval are functions of the contrast set and not of the ranks.

## 5. Point estimation

The **Hodges-Lehmann estimator** [hodges1963](@cite) is the median of the contrast set:

```math
\hat\theta = \operatorname{median}\{A_{ij}\}, \qquad
\hat\Delta = \operatorname{median}\{D_{ij}\} .
```

By §4.2 this is the value at which the statistic sits closest to its null mean ``m/2``,
which is the sense in which it is the estimator the test induces. For even ``m`` the
median is taken as the mean of ``V_{(m/2)}`` and ``V_{(m/2+1)}``, so the estimate need not
be a member of the contrast set.

``\hat\theta`` is a consistent estimator of the pseudomedian of §2.1 and ``\hat\Delta`` of
the shift of §3.1. Neither is the corresponding sample median, and neither is a difference
of sample medians; equality holds only for exactly symmetric configurations.

Both are equivariant under increasing affine maps: for ``a > 0``, the estimator computed
from ``a \cdot d + b`` is ``a\hat\theta + b``. Neither is equivariant under general
monotone maps. Consequently an estimate computed on a log scale and exponentiated is a
ratio estimate, but it is *not* the estimate the procedure would give if run on
untransformed data — the scale on which the procedure runs is part of the specification of
an analysis, not a presentational choice.

## 6. Interval estimation

### 6.1 Form

For an integer ``k \in \{0, 1, \dots, \lfloor m/2 \rfloor\}``, the two-sided interval is

```math
\bigl(\, V_{(k+1)}, \ V_{(m-k)} \,\bigr) ,
```

a pair of order statistics of the contrast set of §4. Equivalently
``(V_{(C_\alpha)}, V_{(m+1-C_\alpha)})`` with ``C_\alpha = k+1``, which is the form used
in most of the literature. Only the choice of ``k`` distinguishes the exact construction
(§6.3) from the approximate one (§6.4).

### 6.2 Inversion

Take the one-sample case; the two-sample case is identical with ``U``, ``D`` and
``\Delta`` throughout. By §4.2, ``W^+(\theta) = \#\{A_{ij} > \theta\}``, so
``\#\{A_{ij} \le \theta\} = m - W^+(\theta)`` and

```math
W^+(\theta) \ge k+1 \iff \theta < A_{(m-k)} , \qquad
W^+(\theta) \le m-k-1 \iff \theta \ge A_{(k+1)} .
```

The two-sided test with rejection region ``\{W^+ \le k\} \cup \{W^+ \ge m-k\}`` therefore
fails to reject exactly on ``\bigl[A_{(k+1)},\, A_{(m-k)}\bigr)``. The interval of §6.1 is
the closure of that set; under continuous ``F`` the endpoints are attained with
probability zero, so the distinction does not affect coverage. Since the null distribution
of ``W^+`` is symmetric about ``m/2``, the two rejection tails have equal probability and

```math
P\bigl(\theta \in (V_{(k+1)}, V_{(m-k)})\bigr) = 1 - 2\,P(W^+ \le k) ,
```

decreasing in ``k``: larger ``k`` gives a narrower interval and less coverage. The
attainable coverages form a finite set, and no construction of this form can achieve a
value between two of them.

### 6.3 Exact index

```math
k = \max\bigl\{\, j \in \{0,\dots,\lfloor m/2 \rfloor\} \;:\; P(W \le j) < \alpha/2 \,\bigr\} ,
```

taken as ``0`` when no such ``j`` exists, with ``P(W \le \cdot)`` the exact null CDF of
§2.2 or §3.2.

By §6.2 the attained coverage is then ``1 - 2P(W \le k) > 1 - \alpha`` strictly, and the
next narrower interval, at ``k+1``, attains at most ``1-\alpha``. This ``k`` therefore
gives the narrowest interval of this form whose coverage still reaches the nominal level.
The excess over ``1-\alpha`` is the discreteness of the null distribution and is not
removable.

Equivalently ``C_\alpha = k+1 = \min\{j : P(W \le j) \ge \alpha/2\}``, the ``\alpha/2``
quantile of the null distribution.

!!! note "This is a choice"
    The alternative is to take the attainable coverage *closest* to ``1-\alpha``, which
    may fall below it. That is not a conservative interval, and it is not what is
    specified here. The two rules coincide whenever the attainable coverage immediately
    below nominal is further from it than the one immediately above.

Since ``P(W \le j)`` is monotone in ``j``, ``k`` may be found by binary search over
``\{0,\dots,\lfloor m/2 \rfloor\}``, at ``O(\log m)`` evaluations of the CDF rather than
``O(m)``.

### 6.4 Normal-approximation index

The target is the exact critical value ``C_\alpha = \min\{j : P(W \le j) \ge \alpha/2\}``.
The statistic is supported on a unit lattice, so with ``\mu_0`` the null **mean** — not
the centred statistic of §2.4 —

```math
P(W \le j) \approx \Phi\!\left(\frac{j + 1/2 - \mu_0}{\sigma}\right) .
```

Setting this to ``\alpha/2`` and solving for ``j``,

```math
C_\alpha = \bigl\lceil\, \mu_0 - z_{1-\alpha/2}\,\sigma - \tfrac{1}{2} \,\bigr\rceil ,
\qquad k = C_\alpha - 1 ,
```

clamped to ``\{0,\dots,\lfloor m/2 \rfloor\}``. In both procedures ``\mu_0 = m/2``: for
one sample ``n(n+1)/4 = m/2``, for two ``n_x n_y / 2 = m/2``.

``\sigma`` is the tie-corrected standard deviation of §2.4 or §3.4, so unlike the exact
construction this one does respond to ties.

!!! note "The continuity correction is a choice"
    Dropping the ``1/2`` makes the index anticonservative. Across 66 one-sample sizes the
    resulting interval is narrower than the exact interval on 44 of them at
    ``1-\alpha = 0.90`` and on 7 at ``0.95``; with the correction, on 10 and none.

The attained coverage of an approximate interval is not computed and is not guaranteed to
reach the nominal level.

### 6.5 One-sided intervals

A one-sided bound at level ``L`` is the corresponding endpoint of the two-sided interval
at level ``2L - 1``; that is, the two-sided ``\alpha`` used is ``2(1-L)`` rather than
``1-L``. The other endpoint is infinite:

```math
\bigl(V_{(k+1)},\, \infty\bigr) \quad\text{or}\quad \bigl(-\infty,\, V_{(m-k)}\bigr) .
```

### 6.6 Zeros, ties, and degeneracy

**Zeros.** By §2.1 the one-sample statistic is computed from the ``n`` non-zero
differences. The contrast set must be formed from the same ``n`` observations, or the
p-value and the interval describe different samples. For a 20-point sample containing five
zeros the contrast set has 120 members, not 210. If every difference is zero, every
contrast is zero and the interval degenerates to the point ``0``.

**Ties on the exact route.** §6.3 inverts the untied null distribution of §2.2 or §3.2.
Under ties the relevant null distribution is the conditional one of §2.3 or §3.3, so the
attained coverage is approximate rather than exact. The classical construction retains the
untied distribution; the alternative is to decline an exact interval under ties and fall
back to §6.4.

**Degeneracy.** If ``P(W \le 0) \ge \alpha/2`` then ``k = 0`` and the widest available
interval, ``(V_{(1)}, V_{(m)})``, does not attain the nominal level. Its coverage is
``1 - 2^{1-n}`` in the untied one-sample case. An implementation should either return it
with a warning or signal that the level is unattainable; returning it silently is a
divergence worth documenting.

## 7. Computational cost

| quantity | cost |
|---|---|
| statistic | ``O(N \log N)`` |
| exact null distribution, no ties | §2.2, §3.2 — polynomial, ``O(n^2)`` space |
| exact null distribution, ties | ``O(2^n)`` or ``O\bigl(\binom{N}{\min(n_x,n_y)}\bigr)`` |
| contrast set, as specified | ``O(m)`` space, ``O(m \log m)`` time; ``m`` is quadratic in the sample size |
| interval index, exact | ``O(\log m)`` CDF evaluations |

The quadratic contrast set is the binding constraint at scale. Both the estimator and the
interval endpoints are order statistics, so a selection algorithm computes them without
materialising the set; the standard approach exploits the fact that the Walsh averages and
the cross differences form a sorted matrix.

## 8. Relation to other implementations

R's `wilcox.test(conf.int = TRUE)` is the usual reference. Where it takes its exact route
it implements §6.3 — via the algorithm of [Bauer (1972)](@cite bauer1972) — and agrees
digit for digit with this specification.

Three deliberate differences. The first two concern its approximate route:

  - It does not return order statistics. It solves numerically, with `uniroot`, for the
    shift at which the statistic crosses its critical value, so its endpoints lie near a
    contrast without being one — `3.0500354` where §6.4 gives `3.05`.
  - It continuity-corrects the interval, as §6.4 does, but not the point estimate, which
    it also root-finds rather than taking as the median of the contrast set. Its reported
    estimate therefore drifts from ``\hat\theta``: `9.71184` against `9.675` on the sample
    of §9.
  - Under ties it declines an exact interval entirely and falls back to its approximate
    route, where §6.6 retains the classical construction instead.

It also warns and substitutes a lower-coverage interval in the degenerate case of §6.6.

Its route-selection rule differs too: R takes the exact route when each sample is under 50
and there are no ties. Comparisons against R must set its `exact` argument explicitly, or
the two implementations may be running different constructions.

## 9. Worked values

Conformance vectors. Values are exact as printed unless a tolerance is implied by the
digits shown.

**One sample, ``n = 15``, no ties, no zeros.**
``d`` = `[-7.8, -6.9, -4.7, 3.7, 6.5, 8.7, 9.1, 10.1, 10.8, 13.6, 14.4, 16.6, 20.2, 22.4, 23.5]``,
``m = 120``.

| quantity | value |
|---|---|
| ``\hat\theta`` (§5) | `9.675` |
| median of ``d``, for contrast | `10.1` |
| exact, ``1-\alpha = 0.95`` (§6.3) | ``k = 25``, ``C_\alpha = 26``, `(3.3, 15.5)` |
| attained coverage | `0.95209`; at ``k=26`` it is `0.94464` |
| approximate, ``1-\alpha = 0.95`` (§6.4) | ``\sigma = 17.60682``, ``C_\alpha = 25``, `(3.05, 15.5)` |
| exact, ``1-\alpha = 0.90`` | `(4.45, 14.45)` |
| exact, one-sided lower, ``L = 0.95`` (§6.5) | `4.45` |

The last two rows illustrate §6.5: the one-sided bound at ``0.95`` is the lower endpoint of
the two-sided interval at ``0.90``.

**One sample, ``N = 20`` with five zeros and ties among the rest.**
``d`` = `[0, 0, 0, 0.5, 0.5, 1, -0.5, -1, 1.5, -1.5, 0.5, 0, 1, -0.5, 2, 0, 0.5, -1, 1, 0.5]``,
so ``n = 15`` and ``T(|d|) = 462``.

| quantity | value |
|---|---|
| ``\hat\theta`` | `0.5` |
| contrast set size (§6.6) | `120`, against `210` if zeros were retained |
| exact, ``1-\alpha = 0.90`` | `(-0.25, 0.75)` |
| two-sided p-value (§2.5, tied branch) | `0.30719` |

**Two samples, no ties.** ``x`` = `1:10`, ``y`` = `2.1, 4.1, …, 20.1`; ``m = 100``,
``U = 20``.

| quantity | value |
|---|---|
| ``\hat\Delta`` | `-5.6` |
| exact, ``1-\alpha = 0.95`` | ``k = 23``, ``C_\alpha = 24``, `(-11.1, -0.1)` |
| ``P(U \le 23)``, ``P(U \le 24)`` | `0.021629`, `0.026213` |
| attained coverage | `0.95674` |

**Two samples, ties.** ``x`` = `1:10`, ``y`` = `2, 4, …, 24`. Exact at ``0.95``:
`(-14.0, -1.0)`.

## 10. In this package

The one-sample procedure is [`SignedRankTest`](@ref), [`ExactSignedRankTest`](@ref) and
[`ApproximateSignedRankTest`](@ref); the two-sample procedure is
[`MannWhitneyUTest`](@ref), [`ExactMannWhitneyUTest`](@ref) and
[`ApproximateMannWhitneyUTest`](@ref). The `Exact*` types implement §6.3, the
`Approximate*` types §6.4, and the dispatchers select between them by sample size and tie
pattern unless the `method` keyword says otherwise.

`pvalue` implements §2.5 and §3.5, `confint` implements §6, and
[`hodgeslehmann`](@ref) implements §5.

Two departures from the specification, both recorded here because §8 asks the same of
other implementations. The degenerate case of §6.6 returns the widest interval silently
rather than warning. And the contrast set is materialised as written in §4 rather than
computed by selection as §7 describes, which bounds the usable sample size; the bound is
[`MAX_RANK_CONTRASTS`](@ref HypothesisTests.MAX_RANK_CONTRASTS), and the tied enumerations
of §2.3 and §3.3 are bounded by
[`MAX_EXACT_ENUMERATION_N`](@ref HypothesisTests.MAX_EXACT_ENUMERATION_N).

## 11. References

The estimator is due to [Hodges and Lehmann (1963)](@cite hodges1963). The two-sample
interval of §6 is commonly called the **Moses interval**, after the chapter L. E. Moses
contributed to Walker and Lev's *Statistical Inference* (1953); the one-sample counterpart
is usually credited to Tukey. Both constructions, with tables, are in
[Hollander and Wolfe (1973)](@cite hollander1973) at pages 27–33 and 68–75.
[Bauer (1972)](@cite bauer1972) gives the order-statistic algorithm.
