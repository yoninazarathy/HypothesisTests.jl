# Rank-based location inference

Specification of two rank procedures and the location inference built on them:

  - the **one-sample procedure**, the Wilcoxon signed rank test, for a single sample of
    differences, and equally for **paired** data, which enters as the within-pair
    differences ``d_i = x_i - y_i``;
  - the **two-sample procedure**, the Wilcoxon rank sum test, equivalently the
    Mann-Whitney ``U`` test, for two independent samples.

[§2](@ref "2. The one-sample procedure (Wilcoxon signed rank)") and
[§3](@ref "3. The two-sample procedure (Wilcoxon rank sum, Mann-Whitney U)") take the two
procedures in turn, each giving the statistic, its exact null distribution, the normal
approximation to it, and the p-value. The location inference is then common to both: the
contrast sets it is read off ([§4](@ref "4. Contrast sets")), the Hodges-Lehmann point
estimate ([§5](@ref "5. Point estimation")), and the confidence interval obtained by
inverting either test ([§6](@ref "6. Interval estimation")). The rest relates the
specification to other implementations
([§7](@ref "7. Relation to other implementations")) and gives worked values for checking
one ([§8](@ref "8. Worked values for the rank tests"))).

**In this package.** The one-sample procedure is [`SignedRankTest`](@ref),
[`ExactSignedRankTest`](@ref) and [`ApproximateSignedRankTest`](@ref); the two-sample
procedure is [`MannWhitneyUTest`](@ref), [`ExactMannWhitneyUTest`](@ref) and
[`ApproximateMannWhitneyUTest`](@ref). [§9](@ref "9. The rank tests in this package") maps the
specification onto them and records where they depart from it.

## 1. Preliminaries and notation

| | |
|---|---|
| ``d_1, \dots, d_N`` | the one-sample input; for paired data ``d_i = x_i - y_i`` |
| ``n`` | the number of ``d_i`` with ``d_i \ne 0`` |
| ``x_1, \dots, x_{n_x}``, ``y_1, \dots, y_{n_y}`` | the two-sample inputs |
| ``N`` | the number of observations: ``N`` one-sample, ``N = n_x + n_y`` two-sample |
| ``\alpha`` | two-sided error rate; the interval has nominal coverage ``1-\alpha`` |
| ``z_q`` | the ``q`` quantile of the standard normal distribution |
| ``\Phi`` | the standard normal CDF |
| ``\ell`` | the number of values being ranked at once |
| ``m`` | the size of the contrast set of [§4](@ref "4. Contrast sets") |
| ``V_{(1)} \le \dots \le V_{(m)}`` | that contrast set, sorted |

**Ranks and midranks.** Ranking always happens on a single vector of ``\ell`` values:
the ``\ell = n`` retained ``|d_i|`` in
[§2](@ref "2. The one-sample procedure (Wilcoxon signed rank)"), and the ``\ell = N``
pooled observations in
[§3](@ref "3. The two-sample procedure (Wilcoxon rank sum, Mann-Whitney U)").

Sort those values. Absent ties, the **rank** of a value is its position in that order, and
the ranks are a permutation of ``1, \dots, \ell``. Tied values occupy a block of
consecutive positions, and each is instead given the average of the positions in its block,
its **midrank**. For a vector ``v``,

```math
R_i = \#\{ j : v_j < v_i \} + \frac{1 + \#\{ j : v_j = v_i \}}{2} ,
```

the second count including ``i`` itself. A value tied with no other therefore has
``\#\{ j : v_j = v_i \} = 1`` and receives its ordinary rank, while a group of ``t``
equal values occupying positions ``r+1, \dots, r+t`` receives the common midrank
``r + (t+1)/2``. The two notions differ only on tied data.

**Tie totals.** For a vector ``v``, write

```math
T(v) = \sum_{g} (t_g^3 - t_g)
```

where ``t_g`` is the multiplicity of the ``g``-th group of equal values in ``v``. Since
``t^3 - t = t(t-1)(t+1)``, groups of size one contribute nothing, and ``T(v) = 0`` exactly
when ``v`` has no ties. Both normal approximations reduce the tie pattern to
``T`` and use nothing else from it, reaching it through the null variance of their
statistic ([§2.1](@ref "2.1 Model, estimand, statistic"),
[§3.1](@ref "3.1 Model, estimand, statistic")). The exact route never uses ``T``: under
ties it conditions on the whole multiset of midranks and enumerates
([§2.3](@ref "2.3 Exact null distribution, ties present"),
[§3.3](@ref "3.3 Exact null distribution, ties present")).

### 1.1 Mathematical observations

Three properties of midranks, used repeatedly below.

**They depend only on the multiset of values.** The closed form counts values and nothing
else, so no input ordering and no tie-breaking convention enters: permuting ``v`` permutes
``R`` the same way, and equal values always receive equal ranks.

**Ties leave the rank total alone.** Whether or not ``v`` has ties,
``\sum_i R_i = \ell(\ell+1)/2``, since a group of tied values contributes one copy of the
average of the positions it occupies for each position it occupies. This is why ties never
affect the null mean of either statistic.

**Ties shrink the rank spread by exactly ``T(v)/12``.** A group of ``t`` tied values at
positions ``r+1, \dots, r+t`` takes the common midrank ``r + (t+1)/2``, lowering the sum of
squares by

```math
\sum_{k=1}^{t} (r+k)^2 \;-\; t\left(r + \tfrac{t+1}{2}\right)^{2}
  = \frac{t(t+1)(2t+1)}{6} - \frac{t(t+1)^2}{4}
  = \frac{t^3 - t}{12} ,
```

independent of ``r``: only the size of a group matters, not where it sits. Summing over
groups,

```math
\sum_i R_i^2 = \frac{\ell(\ell+1)(2\ell+1)}{6} - \frac{T(v)}{12} ,
\qquad
\sum_i (R_i - \bar R)^2 = \frac{\ell(\ell^2-1)}{12} - \frac{T(v)}{12} .
```

So the cube in ``T`` is not a fitted constant, and ``T`` is not a correction bolted onto
the variances: it is twelve times the amount by which ties shrink the spread of the ranks.
Every appearance of ``T`` below is one of these two identities substituted into a variance,
which is why it always arrives divided by ``12``, or by ``48 = 4 \times 12``.

### 1.2 A worked ranking

Both procedures reduce their data to midranks before anything else happens, and both
summarise the ties through ``T``. A small example of each fixes the mechanics. The two
statistics are defined here as they are computed;
[§2.1](@ref "2.1 Model, estimand, statistic") and
[§3.1](@ref "3.1 Model, estimand, statistic") restate them alongside their supports, null
moments and null distributions.

**One sample.** Take ``d = (2.1,\, -0.7,\, 0,\, 1.4,\, -2.1,\, 0.7)``. The zero is
discarded, leaving ``n = 5``, and the remaining absolute values are ranked. Two pairs tie:
the two ``0.7`` occupy positions ``1`` and ``2``, the two ``2.1`` occupy ``4`` and ``5``,
and each takes the average of the positions it spans.

| ``d_i`` | ``2.1`` | ``-0.7`` | ``0`` | ``1.4`` | ``-2.1`` | ``0.7`` |
|---|---|---|---|---|---|---|
| ``\lvert d_i \rvert`` | ``2.1`` | ``0.7`` | discarded | ``1.4`` | ``2.1`` | ``0.7`` |
| midrank ``R_i`` | ``4.5`` | ``1.5`` | | ``3`` | ``4.5`` | ``1.5`` |
| counted in ``W^+`` | yes | no | | yes | no | yes |

The two tied pairs give ``T(\lvert d \rvert) = (2^3 - 2) + (2^3 - 2) = 12``.

The **signed rank statistic** adds up the midranks carried by the positive observations,
discarding the rest:

```math
W^+ = \sum_{i \,:\, d_i > 0} R_i .
```

Here that is the three cells marked *yes*, carrying midranks ``4.5``, ``3`` and ``1.5``,
so ``W^+ = 9``. Its null mean and variance
([§2.1](@ref "2.1 Model, estimand, statistic")) come to ``7.5`` and ``13.5``, the variance
having been reduced from its untied value ``13.75`` by ``T(|d|)/48``.

**Two samples.** Take ``x = (4.2,\, 1.5,\, 3.7,\, 5.4,\, 2.2)`` and
``y = (6.1,\, 2.8,\, 3.7,\, 4.9)``, so ``n_x = 5``, ``n_y = 4`` and ``N = 9``. Here the
two samples are pooled before ranking, and the tie falls *across* them: the two ``3.7``
share positions ``4`` and ``5``.

| pooled value | ``1.5`` | ``2.2`` | ``2.8`` | ``3.7`` | ``3.7`` | ``4.2`` | ``4.9`` | ``5.4`` | ``6.1`` |
|---|---|---|---|---|---|---|---|---|---|
| from | ``x`` | ``x`` | ``y`` | ``x`` | ``y`` | ``x`` | ``y`` | ``x`` | ``y`` |
| midrank | ``1`` | ``2`` | ``3`` | ``4.5`` | ``4.5`` | ``6`` | ``7`` | ``8`` | ``9`` |

The **Mann-Whitney statistic** adds up the pooled midranks that fell to ``x``, then
subtracts the smallest total those ``n_x`` ranks could possibly have taken:

```math
U = \sum_{i=1}^{n_x} R_i - \frac{n_x(n_x+1)}{2} ,
```

with ``R_i`` the pooled midrank of ``x_i``. Here ``x`` received
``\{1,\, 2,\, 4.5,\, 6,\, 8\}``, totalling ``21.5``, and the subtracted minimum is
``1 + 2 + 3 + 4 + 5 = 15``, fixed by ``n_x`` alone, so ``U = 6.5``. That subtraction turns
the rank sum into a count, and counting directly over the ``n_x n_y = 20`` pairs gives the
same number: ``5.4`` beats ``2.8``, ``3.7`` and ``4.9``; ``4.2`` beats ``2.8`` and ``3.7``;
``3.7`` beats ``2.8`` and ties the other ``3.7``; for ``6 + \tfrac{1}{2} = 6.5``. That the
rank sum and the pair count agree is not a coincidence of this example:
[§3.1](@ref "3.1 Model, estimand, statistic") proves they are equal for every sample.
One tied group of two gives ``T([x; y]) = 6``, and the null mean and variance are ``10``
and ``16.52778``.

Both examples are tied, which is the case worth working. Absent ties every group has size
one, ``T = 0``, the midranks are the ordinary ranks ``1, \dots, \ell``, and both variances
collapse to their first term: ``13.75`` here in place of ``13.5``, and ``16.66667`` in
place of ``16.52778``. The correction is small on samples this size and stays small, but it
is the only route by which the tie pattern reaches either normal approximation.

## 2. The one-sample procedure (Wilcoxon signed rank)

### 2.1 Model, estimand, statistic

**Model.** The ``d_i`` are independent. The null hypothesis is that each is symmetric
about ``0``, meaning ``P(d_i > t) = P(d_i < -t)`` for every ``t``.

That is the whole of what [§2.2](@ref "2.2 Exact null distribution, no ties")–[§2.5](@ref "2.5 p-values")
use: symmetry about ``0`` makes the signs independent of ``|d|`` and each ``\pm`` with
probability ``1/2``, and every null distribution below follows from that alone. In
particular the ``d_i`` need not be identically distributed for the null distribution to
hold, though [§5](@ref "5. Point estimation") and [§6](@ref "6. Interval estimation") do
assume a common ``F``, since the pseudomedian they estimate is a functional of one
distribution.

Continuity of ``F`` is a convenience rather than a requirement. It makes zeros and ties
events of probability zero, so the ranks are the integers ``1, \dots, n`` and the lattice
distribution of [§2.2](@ref "2.2 Exact null distribution, no ties") applies. Under a
discrete or mixed ``F`` the procedure remains exact provided ties are handled as in
[§2.3](@ref "2.3 Exact null distribution, ties present"), which also explains in what sense
such a test is exact, and zeros are discarded as described below.

What cannot be weakened is symmetry. Testing ``\operatorname{median}(F) = 0`` is a
different problem, and this test does not solve it. ``W^+`` sits at ``n(n+1)/4`` on average
only when ``F`` is symmetric; an asymmetric ``F`` with median ``0`` has a pseudomedian away
from ``0``, and ``W^+`` follows the pseudomedian, so it is centred somewhere else and the
test rejects more often than ``\alpha``. Take ``F`` the law of
``\mathrm{Exponential}(1) - \ln 2``, whose median is ``0`` and whose pseudomedian is about
``0.145``: at ``n = 20`` the nominal ``0.05`` two-sided test rejects about ``10\%`` of the
time.

**Estimand.** The **pseudomedian** of ``F``: the median of ``(d + d')/2`` for independent
``d, d' \sim F``. If ``F`` is symmetric about ``\theta`` then the pseudomedian is
``\theta``, and coincides with the median of ``F`` and, where it exists, with the mean.
For asymmetric ``F`` the pseudomedian is a different functional from the median, and it is
the pseudomedian that [§5](@ref "5. Point estimation") and
[§6](@ref "6. Interval estimation") estimate.

**Zeros.** Observations with ``d_i = 0`` are discarded before ranking; a zero has no sign
to rank. All of [§2](@ref "2. The one-sample procedure (Wilcoxon signed rank)") and all of
[§4](@ref "4. Contrast sets")–[§6](@ref "6. Interval estimation") are computed from the
``n`` retained observations. ``N`` enters nothing that is computed, so an implementation
reporting a sample size should report both counts, or say which one it means.

The discard is total. The procedure returns exactly what it would have returned had those
observations never been recorded, so appending zeros to a sample moves neither ``W^+``, nor
the p-value, nor the estimate, nor the interval.

Discarding conditions the test on which observations were non-zero, and so on ``n``, in
the sense set out in [§2.3](@ref "2.3 Exact null distribution, ties present"). That costs
nothing in exactness, since the retained signs are still independent and each ``\pm`` with
probability ``1/2`` whatever the zeros did. It does cost information: only ``n`` of the
``N`` observations reach the test, and the support of a statistic built from ``n`` ranks is
coarser than one built from ``N``, so the smallest attainable p-value is larger and the
power lower.

That is a choice rather than a consequence, and it is a strong one, since a difference of
exactly ``0`` is the observation most consistent with symmetry about ``0`` and is given no
weight at all. The alternative is Pratt's convention [pratt1959](@cite), which ranks the
zeros along with the rest and then omits their ranks from the sum: the zeros inflate the
midranks of the retained observations and so do reach the p-value. The two conventions
disagree often on samples containing zeros, and in both directions, so neither is uniformly
the more conservative. This page specifies the discard, which is Wilcoxon's own convention
and the one R's `wilcox.test` follows.

**Statistic.** Re-index the retained observations as ``d_1, \dots, d_n`` and let ``R_i``
be the midrank of ``|d_i|`` among those ``n`` absolute values. Then

```math
W^+ = \sum_{\substack{1 \le i \le n \\ d_i > 0}} R_i ,
```

the index running over the retained observations only. Both the ranking and the sum are
taken after the zeros are gone, so a zero affects ``W^+`` by changing the midranks of the
other observations, not by being skipped in the sum.

Its support is ``\{0, 1, \dots, n(n+1)/2\}``, and under the null it is symmetric about
``n(n+1)/4``.

Under the null the signs are independent of ``|d|`` and each ``\pm`` with probability
``1/2``, so conditionally on the midranks ``W^+ = \sum_i R_i B_i`` with
``B_i \overset{\text{iid}}{\sim} \mathrm{Bernoulli}(1/2)``. Hence
``\mathbb{E}[W^+] = \tfrac{1}{2}\sum_i R_i`` and
``\operatorname{Var}(W^+) = \tfrac{1}{4}\sum_i R_i^2``, and substituting
[§1.1](@ref "1.1 Mathematical observations"),

```math
\mathbb{E}[W^+] = \frac{n(n+1)}{4}, \qquad
\operatorname{Var}(W^+) = \frac{n(n+1)(2n+1)}{24} - \frac{T(|d|)}{48} .
```

The mean is untouched by ties because the rank total is; the variance is not, because the
sum of squares is not.

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
    ``c_n(w)`` grows like ``2^n``, so accumulating the counts and dividing only at the end
    overflows a fixed-width integer at ``n = 72`` in 64 bits, and does so silently: the
    largest count reaches ``1.05 \times 10^{19}`` against a ceiling of
    ``9.22 \times 10^{18}``. Carrying the normalisation inside the recursion avoids it,
    propagating probabilities in place of counts. Exact arithmetic is the alternative.

    This package takes the first route, through `StatsFuns.signrankcdf`, and
    [§9](@ref "9. The rank tests in this package") records the version from which that holds.

**Implementations.** Here ``F_n`` is `StatsFuns.signrankcdf(n, w)`, with `signrankccdf`
for the upper tail. R computes the same distribution in `psignrank`, `dsignrank` and
`qsignrank`, C routines in its `nmath` library, which accumulate the counts ``c_n(w)`` in
`double`: exact while they stay under ``2^{53}`` and approximate beyond, which is a
different way out of the overflow above. The Julia values are tested against R's, as are
the p-values and intervals of [§8](@ref "8. Worked values for the rank tests").

### 2.3 Exact null distribution, ties present

With ties the midranks are not ``1, \dots, n`` and the recursion above does not apply.
The sign-symmetry argument still does: conditionally on the observed multiset of
midranks, the ``2^n`` sign assignments are equally likely under the null. The exact
conditional null distribution is therefore obtained by enumerating them,

```math
P(W^+ \le w) = 2^{-n} \, \#\Bigl\{ \varepsilon \in \{0,1\}^n : \textstyle\sum_i \varepsilon_i R_i \le w \Bigr\} ,
```

at cost ``O(2^n)``. The proportions of assignments falling at or below and at or above the
observed ``W^+`` are written ``q_{\le}`` and ``q_{\ge}``, and are what
[§2.5](@ref "2.5 p-values") uses.

**What "conditional" means here.** The distribution just described is not the distribution
of ``W^+`` over repeated samples from ``F``. It is the distribution over the ``2^n`` sign
patterns with the observed absolute values, and therefore their midranks, held fixed at
what was seen. Every probability in
[§2.5](@ref "2.5 p-values") is computed in that fixed-``|d|`` distribution.

Two things make this the right object rather than a retreat from one. Under the null the
signs are independent of ``|d|``, so fixing ``|d|`` discards nothing that bears on the
null. And a test whose level is exactly ``\alpha`` for every possible value of ``|d|`` has
level exactly ``\alpha`` when averaged over ``|d|``, which is the unconditional statement:
conditional exactness is the stronger property, not a weaker substitute for it. That is
also why the untied case of [§2.2](@ref "2.2 Exact null distribution, no ties") needs no
such discussion. There the midranks are ``1, \dots, n`` whatever the data, so conditioning
on them fixes nothing, and the two distributions coincide.

Discarding zeros ([§2.1](@ref "2.1 Model, estimand, statistic")) conditions in exactly the
same way, on which observations were non-zero, and is exact for the same reason.

**Implementations.** This package computes this distribution, and reaches it often.
`ExactSignedRankTest` enumerates whenever the tie total is non-zero, and `SignedRankTest`
selects that test automatically for tied data up to ``n = 15``, going to the normal
approximation above it. `method = :exact` forces the route as far as
[`MAX_EXACT_ENUMERATION_N`](@ref HypothesisTests.MAX_EXACT_ENUMERATION_N)`= 25`, past which
it refuses rather than enumerate. The tied p-value of [§8](@ref "8. Worked values for the rank tests") is
computed this way.

Nothing in StatsFuns covers the tied case, and base R declines it: `wilcox.test` warns that
it cannot compute an exact p-value with ties, or with zeros, and falls back on its normal
approximation. R's `exactRankTests::wilcox.exact` does compute it, and one of the tied
p-values in this package's test suite is taken from it.

### 2.4 Normal approximation

``W^+`` is asymptotically normal with the mean and variance of
[§2.1](@ref "2.1 Model, estimand, statistic"). Write

```math
\mu = W^+ - \frac{n(n+1)}{4}, \qquad
\sigma = \sqrt{\frac{n(n+1)(2n+1)}{24} - \frac{T(|d|)}{48}}
```

for the centred statistic and the tie-corrected standard deviation. The variance
correction is exact under the conditional distribution of
[§2.3](@ref "2.3 Exact null distribution, ties present"), not an approximation to it.

### 2.5 p-values

Exact, no ties:

```math
p_{\text{left}} = F_n(W^+), \qquad
p_{\text{right}} = 1 - F_n(W^+ - 1), \qquad
p_{\text{both}} = \min\bigl(1,\, 2\min(p_{\text{left}},\, p_{\text{right}})\bigr) .
```

``F_n`` is symmetric about ``n(n+1)/4``, so the smaller tail is the left one exactly when
``W^+ \le n(n+1)/4``; the two-sided value may equivalently be computed by branching on that
comparison and doubling the selected tail.

The clip is not redundant. Where ``n(n+1)/2`` is even the null mean is itself attainable,
and at ``W^+ = n(n+1)/4`` the two tails are equal, each exceeding ``1/2`` by half the atom
sitting on the mean:

```math
2 F_n\bigl(n(n+1)/4\bigr) = 1 + P\bigl(W^+ = n(n+1)/4\bigr) > 1 .
```

At ``n = 3`` with ``W^+ = 3`` that doubled tail is ``1.25``. Doubling a discrete tail can
overshoot, and this is where it does.

Exact, ties present: with ``q_{\le}`` and ``q_{\ge}`` the proportions of sign assignments
giving ``W' \le W^+`` and ``W' \ge W^+`` respectively,

```math
p_{\text{left}} = q_{\le}, \qquad p_{\text{right}} = q_{\ge}, \qquad
p_{\text{both}} = \min\bigl(1,\, 2\min(q_{\le},\, q_{\ge})\bigr) .
```

Normal approximation, with a continuity correction of ``1/2``:

```math
p_{\text{left}} = \Phi\!\left(\frac{\mu + 1/2}{\sigma}\right), \qquad
p_{\text{right}} = 1 - \Phi\!\left(\frac{\mu - 1/2}{\sigma}\right), \qquad
p_{\text{both}} = 2\left[1 - \Phi\!\left(\frac{\bigl|\mu - \tfrac{1}{2}\operatorname{sign}\mu\bigr|}{\sigma}\right)\right] .
```

Degenerate cases: if ``n = 0``, meaning every difference was zero, all three exact p-values are
``1``. If ``\mu = \sigma = 0``, all three approximate p-values are ``1``.

## 3. The two-sample procedure (Wilcoxon rank sum, Mann-Whitney U)

The two names are one procedure, arrived at independently by
[Wilcoxon (1945)](@cite wilcoxon1945) and [Mann and Whitney (1947)](@cite mann1947).
Wilcoxon tabulated the rank sum ``W = \sum_{i=1}^{n_x} R_i`` itself; ``U`` subtracts its
minimum. That minimum is ``n_x(n_x+1)/2``, since the ``n_x`` ranks going to ``x`` are
distinct members of ``\{1, \dots, N\}`` and so total at least ``1 + 2 + \dots + n_x``,
attained exactly when every ``x_i`` falls below every ``y_j``. Hence
``W = U + n_x(n_x+1)/2``, a shift by a constant fixed by ``n_x`` alone and never by the
data. Being a strictly increasing bijection, it leaves the ordering of outcomes untouched,
so the two statistics give the same tail events, the same null distribution up to the
shift, and the same p-value at every level. Subtracting the minimum is what turns the rank
sum into the pair count of [§1.2](@ref "1.2 A worked ranking"), supported from ``0`` and
readable as an estimate of ``P(X > Y)``.

### 3.1 Model, estimand, statistic

**Model.** The two samples are independent of each other, each i.i.d. The null hypothesis
is ``F_x = F_y``, the two distributions equal and otherwise unrestricted.

Equality is what makes the test exact, because it makes the ``N`` observations
exchangeable: every assignment of the pooled midranks to the two samples is then equally
likely, which is the whole of [§3.2](@ref "3.2 Exact null distribution, no ties") and
[§3.3](@ref "3.3 Exact null distribution, ties present"). As in
[§2.1](@ref "2.1 Model, estimand, statistic"), continuity of ``F_x`` and ``F_y`` is a
convenience: it makes ties events of probability zero, so the lattice distribution of
[§3.2](@ref "3.2 Exact null distribution, no ties") applies. Discrete or mixed
distributions keep exactness through the conditional enumeration of
[§3.3](@ref "3.3 Exact null distribution, ties present").

Equality cannot be weakened to ``P(X > Y) = 1/2``. That weaker statement leaves ``F_x`` and
``F_y`` free to differ in spread, and then the pooled observations are no longer
exchangeable, the null variance in this section is no longer the variance of ``U``, and the
test does not hold its level. It fails in both directions, according to which sample is
the larger: for ``X \sim \mathcal{N}(0, 1)`` against ``Y \sim \mathcal{N}(0, 9)``, where
``P(X > Y) = 1/2`` holds exactly by symmetry, the nominal ``0.05`` two-sided test has size
about ``0.13`` at ``(n_x, n_y) = (30, 10)`` and about ``0.016`` at ``(10, 30)``. Under
``F_x = F_y`` both come to ``0.05``. This is the two-sample counterpart of the median
against pseudomedian trap in [§2.1](@ref "2.1 Model, estimand, statistic"), and it is why
that section insists on symmetry and this one on equality.

**Estimand.** Under the **shift model** ``F_x(t) = F_y(t - \Delta)``, the estimand is
``\Delta``. Without that assumption the test is one of ``P(X > Y) = 1/2``, and
[§5](@ref "5. Point estimation") estimates the median of ``X - Y`` for independent
``X \sim F_x``, ``Y \sim F_y``. That quantity is not in general
``\operatorname{median}(F_x) - \operatorname{median}(F_y)``. Zero observations carry no
special meaning here and are not discarded.

**Statistic.** Rank the pooled sample of size ``N``. With ``R_i`` the midrank of ``x_i``,

```math
U = \sum_{i=1}^{n_x} R_i - \frac{n_x(n_x+1)}{2}
  = \#\{(i,j) : x_i > y_j\} + \tfrac{1}{2} \#\{(i,j) : x_i = y_j\} .
```

The two forms are algebraically identical, and the subtracted ``n_x(n_x+1)/2`` is what
makes them so. Split the pooled midrank of ``x_i`` according to which sample each
counted value came from:

```math
R_i = \left[ \#\{j : x_j < x_i\} + \frac{1 + \#\{j : x_j = x_i\}}{2} \right]
    + \#\{j : y_j < x_i\} + \tfrac{1}{2} \#\{j : y_j = x_i\} .
```

The bracket is the midrank of ``x_i`` within ``x`` alone, so summing it over ``i`` gives
``n_x(n_x+1)/2`` however ``x`` is tied
([§1.1](@ref "1.1 Mathematical observations")), which is precisely the term subtracted.
What survives is the pair count. The first form is how the statistic is computed, from one
sort of the pooled sample; the second is the definition the inversion of
[§6.2](@ref "6.2 Inversion") works with. The support is
``\{0, \tfrac{1}{2}, \dots, n_x n_y\}``, integer-valued absent ties, symmetric under the
null about ``n_x n_y / 2``.

Under the null the pooled midranks are fixed and the ``n_x`` of them falling to ``x`` are
a simple random sample without replacement, so ``\sum_{i} R_i`` over that sample has
variance ``\frac{n_x n_y}{N(N-1)}\sum_i (R_i - \bar R)^2``. Substituting
[§1.1](@ref "1.1 Mathematical observations"),

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

The numerical caveat of [§2.2](@ref "2.2 Exact null distribution, no ties") applies with
more force: the normalising constant ``\binom{N}{n_x}`` exceeds ``2^{63}`` for balanced
samples from ``n_x = n_y = 34``, where ``\binom{68}{34} \approx 2.85 \times 10^{19}``.

**Implementations.** Here ``G_{n_x,n_y}`` is `StatsFuns.wilcoxcdf(nx, ny, u)`, with
`wilcoxccdf` for the upper tail. It computes this distribution, but not by the recursion
above: it uses the recurrence of [Löffler (1983)](@cite loeffler1983), a convolution over
divisor sums that reaches the same probabilities with fewer allocations. The recursion
stated here is the one R uses, in the C routines `pwilcox`, `dwilcox` and `qwilcox` of its
`nmath` library, which memoise the counts in `double`. Two algorithms, one distribution;
that the two agree is worth checking, and they do, to floating-point precision at every
attainable ``u`` for ``n_x, n_y \le 7``.

Ties are handled as in [§2.3](@ref "2.3 Exact null distribution, ties present"): by this
package's own enumeration, which `ExactMannWhitneyUTest` reaches whenever the tie total is
non-zero, and which `MannWhitneyUTest` selects automatically for tied data up to
``N = 10``. Neither StatsFuns nor R offers it.

### 3.3 Exact null distribution, ties present

As in [§2.3](@ref "2.3 Exact null distribution, ties present"), the exact conditional null
distribution is obtained by enumeration, here over the ``\binom{N}{\min(n_x, n_y)}``
assignments of the observed midranks to the smaller sample, and the same three p-value
formulas follow.

### 3.4 Normal approximation

``U`` is asymptotically normal with the mean and variance of
[§3.1](@ref "3.1 Model, estimand, statistic"). Write

```math
\mu = U - \frac{n_x n_y}{2}, \qquad
\sigma = \sqrt{\frac{n_x n_y}{12}\left(N + 1 - \frac{T([x; y])}{N(N-1)}\right)}
```

for the centred statistic and the tie-corrected standard deviation. As in
[§2.4](@ref "2.4 Normal approximation"), the tie correction is exact under the conditional
distribution of [§3.3](@ref "3.3 Exact null distribution, ties present") rather than an
approximation to it; what is approximate is only the normal shape.

### 3.5 p-values

Exact, no ties:

```math
p_{\text{left}} = G(U), \qquad
p_{\text{right}} = 1 - G(U - 1), \qquad
p_{\text{both}} = \min\bigl(1,\, 2\, G(\min(U,\, n_x n_y - U))\bigr) .
```

The fold to the lower tail is exact, not an approximation: by the null symmetry of
[§3.1](@ref "3.1 Model, estimand, statistic"),
``G(n_x n_y - U) = P(U' \ge U)``, so ``G(\min(U, n_x n_y - U))`` is the smaller of the two
tails wherever ``U`` sits, the centre included. The clip is needed for the reason it is
needed in [§2.5](@ref "2.5 p-values"): what can exceed ``1`` is the doubling, when ``U``
lands on ``n_x n_y / 2`` and that value carries an atom. At ``n_x = n_y = 2`` with
``U = 2`` the doubled tail is ``4/3``.

Exact under ties, and the normal approximation: exactly as in [§2.5](@ref "2.5 p-values"),
with ``q_{\le}, q_{\ge}`` from
[§3.3](@ref "3.3 Exact null distribution, ties present") and
``\mu, \sigma`` from [§3.4](@ref "3.4 Normal approximation").

Degenerate cases: if ``n_x = 0`` or ``n_y = 0`` there is no pair to compare, ``U = 0`` is
the only attainable value, the null distribution is a point mass there, and all three exact
p-values are ``1``. If ``\mu = \sigma = 0``, all three approximate p-values are ``1``.

## 4. Contrast sets

The two procedures differ in one place only, and this section is where that difference is
isolated. Each forms a set of ``m`` numbers on the scale of the data, one for each pair of
observations its statistic compares. What differs is which pairs those are, and what number
each pair contributes:

  - the one-sample statistic compares every retained observation with every other and with
    itself, giving the ``m = n(n+1)/2`` pairs ``i \le j`` of a single sample, each
    contributing the average ``(d_i + d_j)/2``;
  - the two-sample statistic compares every ``x`` with every ``y``, and never an ``x`` with
    an ``x``, giving the ``m = n_x n_y`` cross pairs, each contributing the difference
    ``x_i - y_j``.

[§4.1](@ref "4.1 Definitions") states those two definitions, and they are the last thing
stated twice for a substantive reason. Everything after them is a statement about a set of
``m`` numbers: sort them as ``V_{(1)} \le \dots \le V_{(m)}``, and the estimate of
[§5](@ref "5. Point estimation") is their sample median while the interval endpoints of
[§6](@ref "6. Interval estimation") are two of their order statistics. This specification
calls the numbers themselves the **contrasts**.

Note which median that is. The median of the ``m`` contrasts is a number computed from the
data, and it is the estimator; the pseudomedian of
[§2.1](@ref "2.1 Model, estimand, statistic") and the shift of
[§3.1](@ref "3.1 Model, estimand, statistic") are the population quantities being
estimated. They are not the same object, and neither is the sample median of the data
itself. [§5](@ref "5. Point estimation") keeps the three apart.

The common treatment rests on the identity of
[§4.2](@ref "4.2 The counting identity"): for both procedures the statistic, recomputed
against a hypothesised location, counts how many contrasts lie above it. Inverting either
test is therefore counting contrasts, and it is the only property
[§5](@ref "5. Point estimation") and [§6](@ref "6. Interval estimation") use.

The notation below nevertheless stays doubled, because the two statistics keep their own
names: results are stated for the one-sample case, in ``W^+`` and ``\theta``, and
[§6.2](@ref "6.2 Inversion") gives the substitution (``U``, ``D``, ``\Delta``) that carries
each of them to the two-sample case. No step of any argument changes under that
substitution. Reading the one-sample line and applying the substitution is enough; the two
are not being developed independently.

The word **contrast** is a convenience of this document rather than established usage. The
one-sample contrasts are normally called the Walsh averages and the two-sample ones simply
the differences, and no standard term covers both. This is also not the
analysis-of-variance sense of *contrast*: the one-sample contrasts are averages of pairs,
whose coefficients sum to one rather than to zero.

### 4.1 Definitions

**One sample: the Walsh averages.** The ``m = n(n+1)/2`` values

```math
A_{ij} = \frac{d_i + d_j}{2}, \qquad 1 \le i \le j \le n ,
```

over the ``n`` retained observations. The diagonal ``i = j`` is included, so each ``d_i``
is itself a member.

**Two samples: the cross-group differences.** The ``m = n_x n_y`` values

```math
D_{ij} = x_i - y_j , \qquad 1 \le i \le n_x, \ 1 \le j \le n_y .
```

**Cost.** Both sets are quadratic in the sample size, and this specification forms them
explicitly: ``m`` numbers, sorted. That is the binding constraint at scale, and it is
avoidable, because the estimate of [§5](@ref "5. Point estimation") and the interval
endpoints of [§6](@ref "6. Interval estimation") are order statistics. A selection
algorithm returns them without materialising the set, exploiting the fact that the Walsh
averages and the cross differences each form a sorted matrix. An implementation that
materialises instead should bound the size it will accept.

### 4.2 The counting identity

Let ``W^+(\theta)`` denote the statistic of [§2.1](@ref "2.1 Model, estimand, statistic")
recomputed on ``d_1 - \theta, \dots, d_n - \theta``, and ``U(\Delta)`` the statistic of
[§3.1](@ref "3.1 Model, estimand, statistic") recomputed on
``x_1 - \Delta, \dots, x_{n_x} - \Delta`` against ``y``. Then

```math
W^+(\theta) = \#\{(i,j) : i \le j, \ A_{ij} > \theta\} + \tfrac{1}{2}\,\#\{(i,j) : i \le j, \ A_{ij} = \theta\} ,
```
```math
U(\Delta) = \#\{(i,j) : D_{ij} > \Delta\} + \tfrac{1}{2}\,\#\{(i,j) : D_{ij} = \Delta\} .
```

The half-count is the midrank convention, and it vanishes whenever ``\theta`` is not
itself a member of the contrast set, which is almost surely the case under continuous
``F`` and is the only case [§6.2](@ref "6.2 Inversion") needs. Both functions are
non-increasing in their argument and change value only at members of the contrast set.

*Proof of the first, for ``\theta`` not a contrast and ``|d|`` untied.*
``R_i = \#\{j : |d_j| \le |d_i|\}``, so
``W^+ = \sum_{i : d_i > 0} \#\{j : |d_j| \le |d_i|\}``. A pair ``\{i, j\}`` is counted
exactly when the one with the larger absolute value is positive, which is exactly when
``d_i + d_j > 0``; the diagonal term ``\{i,i\}`` is counted exactly when ``d_i > 0``.
Applying this to ``d - \theta`` gives the statement. The second is immediate from the
counting form of ``U`` in [§3.1](@ref "3.1 Model, estimand, statistic"). ∎

Everything in [§5](@ref "5. Point estimation") and [§6](@ref "6. Interval estimation") is
a consequence of this identity. It is the reason the estimator and the interval are
functions of the contrast set and not of the ranks.

## 5. Point estimation

The **Hodges-Lehmann estimator** [hodges1963](@cite) is the sample median of the contrast
set of [§4.1](@ref "4.1 Definitions"). The two cases in turn.

**One sample.** The median of the ``n(n+1)/2`` Walsh averages,

```math
\hat\theta = \operatorname{median}\{ A_{ij} : 1 \le i \le j \le n \} ,
```

a consistent estimator of the pseudomedian of
[§2.1](@ref "2.1 Model, estimand, statistic"). It is not the sample median of the ``d_i``,
which estimates the median of ``F``, a different functional except under symmetry.

**Two samples.** The median of the ``n_x n_y`` cross-group differences,

```math
\hat\Delta = \operatorname{median}\{ D_{ij} : 1 \le i \le n_x,\ 1 \le j \le n_y \} ,
```

a consistent estimator of the shift of [§3.1](@ref "3.1 Model, estimand, statistic"), or
without a shift model of the median of ``X - Y``. It is not
``\operatorname{median}(x) - \operatorname{median}(y)``, which is a different quantity
again.

Both statements about what these are *not* matter in practice, because the gap shows up on
ordinary samples. On the one-sample data of [§8](@ref "8. Worked values for the rank tests"),
``\hat\theta = 9.675`` against a sample median of ``10.1``. On a tied nine-against-nine
two-sample set, ``\hat\Delta = 0.56`` against a difference of sample medians of ``0.62``.
Exact symmetry makes each pair agree, but the converse fails: ``d = (0, 2, 2, 7)`` is
symmetric about nothing, and its Walsh averages
``0, 1, 1, 2, 2, 2, 3.5, 4.5, 4.5, 7`` have median ``2``, exactly the sample median.
Agreement on a given sample is therefore no evidence of symmetry.

For either, by [§4.2](@ref "4.2 The counting identity"), the estimate is the value at which
the statistic sits closest to its null mean ``m/2``, which is the sense in which it is the
estimator the test induces. For even ``m`` the median is taken as the mean of
``V_{(m/2)}`` and ``V_{(m/2+1)}``, so the estimate need not be a member of the contrast
set.

**Changing scale.** Shifting and stretching the data does the same to the estimate: from
``a d + b`` with ``a > 0`` the estimator returns ``a \hat\theta + b``. Anything more than
that changes the answer. Take logs, estimate, exponentiate, and what comes back is a ratio,
a perfectly good estimate of a different quantity, but not the number the procedure returns
on the untransformed data.

The p-value behaves differently, which is easy to conflate. ``U`` depends on the pooled
data only through its ranks, so any increasing transformation of a two-sample data set
leaves the two-sided p-value bit-for-bit unchanged while moving the estimate. Choosing the
scale is therefore part of specifying the analysis, and it is a choice about the estimate,
not about the test.

## 6. Interval estimation

### 6.1 Form

For an integer ``k \in \{0, 1, \dots, \lfloor m/2 \rfloor\}``, the two-sided interval is

```math
\bigl(\, V_{(k+1)}, \ V_{(m-k)} \,\bigr) ,
```

a pair of order statistics of the contrast set of [§4](@ref "4. Contrast sets").
Equivalently ``(V_{(C_\alpha)}, V_{(m+1-C_\alpha)})`` with ``C_\alpha = k+1``, which is
the form used in most of the literature. Only the choice of ``k`` distinguishes the exact
construction ([§6.3](@ref "6.3 Exact index")) from the approximate one
([§6.4](@ref "6.4 Normal-approximation index")).

These intervals carry names. The two-sample one, a pair of order statistics of the
``n_x n_y`` cross-group differences, is the **Moses interval**, after the chapter
L. E. Moses contributed to Walker and Lev's *Statistical Inference* (1953); the one-sample
one, read off the Walsh averages, is usually credited to Tukey. Both are also called
distribution-free, or Hodges-Lehmann, confidence intervals, the latter because
[§5](@ref "5. Point estimation") sits inside them by construction. This page treats them as
one object because [§6.2](@ref "6.2 Inversion") derives both from the same counting
identity. [Hollander and Wolfe (1973)](@cite hollander1973) tabulates both, at pages 27–33
and 68–75.

### 6.2 Inversion

Take the one-sample case; the two-sample case is identical with ``U``, ``D`` and
``\Delta`` throughout. By [§4.2](@ref "4.2 The counting identity"), taking ``\theta`` not
itself a contrast so that the half-count vanishes, ``W^+(\theta) = \#\{A_{ij} > \theta\}``,
so ``\#\{A_{ij} \le \theta\} = m - W^+(\theta)`` and

```math
W^+(\theta) \ge k+1 \iff \theta < A_{(m-k)} , \qquad
W^+(\theta) \le m-k-1 \iff \theta \ge A_{(k+1)} .
```

The two-sided test with rejection region ``\{W^+ \le k\} \cup \{W^+ \ge m-k\}`` therefore
fails to reject exactly on ``\bigl[A_{(k+1)},\, A_{(m-k)}\bigr)``. The interval of
[§6.1](@ref "6.1 Form") is the closure of that set; under continuous ``F`` the endpoints
are attained with probability zero, so the distinction does not affect coverage. Since the
null distribution of ``W^+`` is symmetric about ``m/2``, the two rejection tails have
equal probability and

```math
P\bigl(\theta \in (V_{(k+1)}, V_{(m-k)})\bigr) = 1 - 2\,P(W^+ \le k) ,
```

decreasing in ``k``: larger ``k`` gives a narrower interval and less coverage. The
attainable coverages form a finite set, and no construction of this form can achieve a
value between two of them.

**That is an equality, and only under this section's assumptions.** It is exact, not a
bound, when ``F`` is continuous and symmetric about ``\theta``: continuity is what makes
``\theta`` almost surely not a contrast and the ``|d_i - \theta|`` almost surely untied, and
symmetry is what gives ``W^+(\theta)`` the null distribution whose tails appear on the
right. Drop either and the equality goes:

  - ties or zeros, which is to say discrete data, leave the achieved coverage above the
    figure computed here, since the untied null distribution is being inverted for data
    that does not have it. On ``d`` uniform on ``\{-3, \dots, 3\}`` at ``n = 15``, where
    every sample is tied, the nominal ``0.95`` interval covers about ``0.982`` of the time
    against the ``0.95209`` this formula gives. Conservative, but no longer exact;
    [§6.6](@ref "6.6 Zeros, ties, and degeneracy") returns to this.
  - asymmetric ``F`` breaks it in the other direction and by less. For
    ``\mathrm{Exponential}(1)`` at ``n = 15``, coverage of the pseudomedian is about
    ``0.947``.

Both figures are Monte Carlo over ``200\,000`` samples, against ``0.95226`` for the
continuous symmetric case that the equality describes.

### 6.3 Exact index

```math
k = \max\bigl\{\, j \in \{0,\dots,\lfloor m/2 \rfloor\} \;:\; P(W \le j) < \alpha/2 \,\bigr\} ,
```

taken as ``0`` when no such ``j`` exists, with ``P(W \le \cdot)`` the exact null CDF of
[§2.2](@ref "2.2 Exact null distribution, no ties") or
[§3.2](@ref "3.2 Exact null distribution, no ties").

By [§6.2](@ref "6.2 Inversion") the attained coverage is then
``1 - 2P(W \le k) > 1 - \alpha`` strictly, and the next narrower interval, at ``k+1``,
attains at most ``1-\alpha``. This ``k`` therefore gives the narrowest interval of this
form whose coverage still reaches the nominal level. The excess over ``1-\alpha`` is the
discreteness of the null distribution and is not removable.

Equivalently ``C_\alpha = k+1 = \min\{j : P(W \le j) \ge \alpha/2\}``, the ``\alpha/2``
quantile of the null distribution, whenever that minimum is at least ``1``. Where it is
``0``, which is the degenerate case of
[§6.6](@ref "6.6 Zeros, ties, and degeneracy"), the two disagree: the convention above
gives ``k = 0`` and so ``C_\alpha = 1``, which is the widest interval this form admits,
whereas the quantile would index outside the contrast set.

!!! note "This is a choice"
    The alternative is to take the attainable coverage *closest* to ``1-\alpha``, which
    may fall below it. That is not a conservative interval, and it is not what is
    specified here. The two rules coincide whenever the attainable coverage immediately
    below nominal is further from it than the one immediately above.

``P(W \le j)`` is monotone in ``j``, so the condition holds on an initial segment of
``\{0, \dots, \lfloor m/2 \rfloor\}`` and ``k`` is its last member: a binary search finds it
without evaluating the CDF at every index, which matters because each evaluation runs a
lattice recursion of its own.

### 6.4 Normal-approximation index

The target is the exact critical value ``C_\alpha = \min\{j : P(W \le j) \ge \alpha/2\}``.
The statistic is supported on a unit lattice, so with ``\mu_0`` the null **mean**, not
the centred statistic of [§2.4](@ref "2.4 Normal approximation"),

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

``\sigma`` is the tie-corrected standard deviation of
[§2.4](@ref "2.4 Normal approximation") or [§3.4](@ref "3.4 Normal approximation"), so
unlike the exact construction this one does respond to ties.

!!! note "The continuity correction is a choice, and both implementations make it"
    This package applies the ``1/2``, and so does R's `wilcox.test`, whose `correct`
    argument defaults to `TRUE`. The figures below say what dropping it would cost, not
    what either does.

    Without it the index is anticonservative. Across the 66 one-sample sizes ``n = 5:70``
    the interval comes out narrower than the exact one on 45 of them at
    ``1-\alpha = 0.90`` and on 7 at ``0.95``; with the correction, on 10 and on none.

The attained coverage of an approximate interval is not computed and is not guaranteed to
reach the nominal level.

### 6.5 One-sided intervals

A one-sided bound at level ``L`` is the corresponding endpoint of the two-sided interval
at level ``2L - 1``; that is, the two-sided ``\alpha`` used is ``2(1-L)`` rather than
``1-L``. The other endpoint is infinite:

```math
\bigl(V_{(k+1)},\, \infty\bigr) \quad\text{or}\quad \bigl(-\infty,\, V_{(m-k)}\bigr) .
```

!!! warning "This package is not internally consistent here"
    Which of the two a given tail selects is a naming convention, not mathematics, and
    this package does not apply one convention throughout. Its four rank tests return
    ``(V_{(k+1)}, \infty)``, a *lower* bound, for `tail = :left`, and the upper bound
    for `tail = :right`. Every other test here that accepts a `tail` does the opposite,
    returning an upper bound for `tail = :left`: the three t-tests, the z-tests,
    `BinomialTest` and `FisherExactTest`. That is also what R does, for `t.test`,
    `wilcox.test` and `binom.test` alike, under `alternative = "less"`. See
    [§7.1](@ref "7.1 One-sided intervals") of [The t-tests](@ref).

    The rank behaviour predates these specifications and is not changed by them.
    Reconciling the two is a breaking change to whichever side moves.

### 6.6 Zeros, ties, and degeneracy

**Zeros.** By [§2.1](@ref "2.1 Model, estimand, statistic") the one-sample statistic is
computed from the ``n`` non-zero differences. The contrast set must be formed from the
same ``n`` observations, or the p-value and the interval describe different samples. For a
20-point sample containing five zeros the contrast set has 120 members
(``m = n(n+1)/2`` with ``n = 15``), not 210 (which is ``N(N+1)/2`` with ``N = 20``, the
count obtained by retaining the zeros). If every difference is zero, every contrast is zero
and the interval degenerates to the point ``0``.

**Ties on the exact route.** [§6.3](@ref "6.3 Exact index") inverts the untied null
distribution of [§2.2](@ref "2.2 Exact null distribution, no ties") or
[§3.2](@ref "3.2 Exact null distribution, no ties"). Under ties the relevant null
distribution is the conditional one of
[§2.3](@ref "2.3 Exact null distribution, ties present") or
[§3.3](@ref "3.3 Exact null distribution, ties present"), so the attained coverage is
approximate rather than exact. The classical construction retains the untied distribution;
the alternative is to decline an exact interval under ties and fall back to
[§6.4](@ref "6.4 Normal-approximation index").

**Degeneracy.** If ``P(W \le 0) \ge \alpha/2`` then ``k = 0`` and the widest available
interval, ``(V_{(1)}, V_{(m)})``, does not attain the nominal level. Its coverage is
``1 - 2^{1-n}`` in the untied one-sample case: ``0.75`` at ``n = 3``, ``0.875`` at
``n = 4``, ``0.9375`` at ``n = 5``, so a ``0.95`` interval is unattainable below ``n = 6``
and a ``0.99`` interval below ``n = 8``.

Returning that interval as though it met the request would misstate the coverage, so this
package returns it and warns. On the exact route the warning names the coverage that is
attainable, as R's does.

The approximate route warns for a related but weaker reason. Its index rule
([§6.4](@ref "6.4 Normal-approximation index")) can ask for an order statistic outside the
contrast set, which is not the same as the level being out of reach: at ``n = 8`` and
``1-\alpha = 0.99`` the rule asks for ``k = -1`` while the exact route attains ``0.9922``
and so meets the request. That warning therefore reports what happened, and says the exact
route is the way on, without asserting anything about coverage the route cannot compute.

Either way the interval returned is the widest the form admits, since it is the best there
is.

## 7. Relation to other implementations

R's `wilcox.test(conf.int = TRUE)` is the usual reference. Where it takes its exact route
it implements [§6.3](@ref "6.3 Exact index"), via the algorithm of
[Bauer (1972)](@cite bauer1972), and agrees digit for digit with this specification.

Three deliberate differences. The first two concern its approximate route:

  - It does not return order statistics. It solves numerically, with `uniroot`, for the
    shift at which the statistic crosses its critical value, so its endpoints lie near a
    contrast without being one: `3.0500354` where
    [§6.4](@ref "6.4 Normal-approximation index") gives `3.05`.
  - It continuity-corrects the interval, as
    [§6.4](@ref "6.4 Normal-approximation index") does, but not the point estimate, which
    it also root-finds rather than taking as the median of the contrast set. Its reported
    estimate therefore drifts from ``\hat\theta``: `9.71184` against `9.675` on the sample
    of [§8](@ref "8. Worked values for the rank tests").
  - Under ties it declines an exact interval entirely and falls back to its approximate
    route, where [§6.6](@ref "6.6 Zeros, ties, and degeneracy") retains the classical
    construction instead.

It also warns and substitutes a lower-coverage interval in the degenerate case of
[§6.6](@ref "6.6 Zeros, ties, and degeneracy").

Its route-selection rule differs too: R takes the exact route when each sample is under 50
and there are no ties. Comparisons against R must set its `exact` argument explicitly, or
the two implementations may be running different constructions.

## 8. Worked values for the rank tests

Conformance vectors. Values are exact as printed unless a tolerance is implied by the
digits shown.

### 8.1 One sample, no ties and no zeros

``n = 15``, ``d`` =
`[-7.8, -6.9, -4.7, 3.7, 6.5, 8.7, 9.1, 10.1, 10.8, 13.6, 14.4, 16.6, 20.2, 22.4, 23.5]`,
``m = 120``.

| quantity | value |
|---|---|
| ``\hat\theta`` ([§5](@ref "5. Point estimation")) | `9.675` |
| median of ``d``, for contrast | `10.1` |
| exact, ``1-\alpha = 0.95`` ([§6.3](@ref "6.3 Exact index")) | ``k = 25``, ``C_\alpha = 26``, `(3.3, 15.5)` |
| attained coverage | `0.95209`; at ``k=26`` it is `0.94464` |
| approximate, ``1-\alpha = 0.95`` ([§6.4](@ref "6.4 Normal-approximation index")) | ``\sigma = 17.60682``, ``C_\alpha = 25``, `(3.05, 15.5)` |
| exact, ``1-\alpha = 0.90`` | `(4.45, 14.45)` |
| exact, one-sided lower, ``L = 0.95`` ([§6.5](@ref "6.5 One-sided intervals")) | `4.45` |

The last two rows illustrate [§6.5](@ref "6.5 One-sided intervals"): the one-sided bound
at ``0.95`` is the lower endpoint of the two-sided interval at ``0.90``.

### 8.2 One sample, five zeros and ties among the rest

``N = 20``, ``d`` = `[0, 0, 0, 0.5, 0.5, 1, -0.5, -1, 1.5, -1.5, 0.5, 0, 1, -0.5, 2, 0, 0.5, -1, 1, 0.5]`,
so ``n = 15`` and ``T(|d|) = 462``.

| quantity | value |
|---|---|
| ``\hat\theta`` | `0.5` |
| contrast set size ([§6.6](@ref "6.6 Zeros, ties, and degeneracy")) | `120` = ``15 \cdot 16 / 2``, against `210` = ``20 \cdot 21 / 2`` if zeros were retained |
| exact, ``1-\alpha = 0.90`` | `(-0.25, 0.75)` |
| two-sided p-value ([§2.5](@ref "2.5 p-values"), tied branch) | `0.30719` |

### 8.3 Two samples, no ties

``x`` = `1:10`, ``y`` = `2.1, 4.1, …, 20.1`; ``m = 100``, ``U = 20``.

| quantity | value |
|---|---|
| ``\hat\Delta`` | `-5.6` |
| exact, ``1-\alpha = 0.95`` | ``k = 23``, ``C_\alpha = 24``, `(-11.1, -0.1)` |
| ``P(U \le 23)``, ``P(U \le 24)`` | `0.021629`, `0.026213` |
| attained coverage | `0.95674` |

### 8.4 Two samples, ties

``x`` = `1:10`, ``y`` = `2, 4, …, 24`. Exact at ``0.95``: `(-14.0, -1.0)`.

## 9. The rank tests in this package

The one-sample procedure is [`SignedRankTest`](@ref), [`ExactSignedRankTest`](@ref) and
[`ApproximateSignedRankTest`](@ref); the two-sample procedure is
[`MannWhitneyUTest`](@ref), [`ExactMannWhitneyUTest`](@ref) and
[`ApproximateMannWhitneyUTest`](@ref). The `Exact*` types implement
[§6.3](@ref "6.3 Exact index"), the `Approximate*` types
[§6.4](@ref "6.4 Normal-approximation index"), and the dispatchers select between them by
sample size and tie pattern unless the `method` keyword says otherwise.

`pvalue` implements [§2.5](@ref "2.5 p-values") and [§3.5](@ref "3.5 p-values"), `confint`
implements [§6](@ref "6. Interval estimation"), and [`hodgeslehmann`](@ref) implements
[§5](@ref "5. Point estimation").

The exact null distributions of [§2.2](@ref "2.2 Exact null distribution, no ties") and
[§3.2](@ref "3.2 Exact null distribution, no ties") come from StatsFuns, whose recursions
accumulated lattice counts in `Int` and overflowed silently past exactly the bounds those
sections give, until JuliaStats/StatsFuns.jl#221 folded the normaliser into them. The
`[compat]` floor of StatsFuns 2.2.1 is what makes the numerical care of those two sections
a settled question here rather than a caveat. The tied routes are this package's own, and
are bounded rather than corrected: see
[`MAX_EXACT_ENUMERATION_N`](@ref HypothesisTests.MAX_EXACT_ENUMERATION_N) below.

Departures from the specification, recorded here because
[§7](@ref "7. Relation to other implementations") asks the same of other implementations.
Every test here carries a `median` field, the sample
median for the signed rank tests and the difference of sample medians for the Mann-Whitney
ones. Both are descriptive statistics rather than the estimand of
[§5](@ref "5. Point estimation"), and that field, rather than anything the procedure needs,
is what makes an empty sample throw instead of returning the degenerate p-value of
[§3.5](@ref "3.5 p-values"). And the contrast set is materialised as written in
[§4.1](@ref "4.1 Definitions") rather than computed by selection as that section
describes, which bounds the usable sample size; the
bound is [`MAX_RANK_CONTRASTS`](@ref HypothesisTests.MAX_RANK_CONTRASTS), and the tied
enumerations of [§2.3](@ref "2.3 Exact null distribution, ties present") and
[§3.3](@ref "3.3 Exact null distribution, ties present") are bounded by
[`MAX_EXACT_ENUMERATION_N`](@ref HypothesisTests.MAX_EXACT_ENUMERATION_N).

## 10. References

Both procedures originate with [Wilcoxon (1945)](@cite wilcoxon1945), which introduced the
signed rank and the rank sum tests together; [Mann and Whitney (1947)](@cite mann1947)
arrived at the two-sample test independently and in the counting form, which
[§3](@ref "3. The two-sample procedure (Wilcoxon rank sum, Mann-Whitney U)") reconciles
with the rank sum.
The estimator is due to [Hodges and Lehmann (1963)](@cite hodges1963). The two-sample
interval of [§6](@ref "6. Interval estimation") is commonly called the **Moses interval**,
after the chapter L. E. Moses contributed to Walker and Lev's *Statistical Inference*
(1953); the one-sample counterpart is usually credited to Tukey. Both constructions, with
tables, are in [Hollander and Wolfe (1973)](@cite hollander1973) at pages 27–33 and 68–75.
[Bauer (1972)](@cite bauer1972) gives the order-statistic algorithm.
