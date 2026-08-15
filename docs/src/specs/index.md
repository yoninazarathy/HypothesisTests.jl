# Appendix: mathematical specifications

These pages state, precisely and in one place, what a test computes: the statistic, its
null distribution, the p-value, and — where the test has them — the point estimate and
the confidence interval.

They are an appendix rather than part of the manual proper. The manual tells you what a
function is called and what its arguments mean; a specification tells you what it
computes, in terms that do not depend on this package or on Julia. The definitions are
the published ones, the derivations are given rather than asserted, and the worked values
closing each page are conformance vectors that any implementation can be held to. The
mapping onto this package's API is confined to a single closing section.

Sections are numbered so that they can be cited. §6.3 of a specification means what it
says, and will go on meaning it.

## Contents

  - [Rank-based location inference](@ref) — the Wilcoxon signed rank and rank sum
    (Mann-Whitney) procedures, the Hodges-Lehmann estimators, and the distribution-free
    intervals obtained by inverting either test.

More will be added as they are written.

## Conventions

Each page states its estimand before its statistic. A rank test and the parameter it
estimates are separate objects, and conflating them is the most common way to misread
one.

Derivations appear where a result would otherwise have to be taken on trust,
particularly for a confidence interval obtained by inverting a test.

Where a construction involves a genuine choice rather than a consequence — a continuity
correction, a tie-breaking convention, a conservative rather than a nearest-attainable
level — the choice is named as such and the alternative is given.

Differences between implementations are recorded as differences, not as errors, with the
reference implementation named.
