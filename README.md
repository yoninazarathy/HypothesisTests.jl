# Documentation preview — not the HypothesisTests.jl documentation

This branch of a **fork** hosts a rendered preview of the documentation as it
stands on the rank test pull request stack, so that reviewers can read the
mathematical specification without building the docs locally.

- Preview: <https://yoninazarathy.github.io/HypothesisTests.jl/>
- Specification appendix: <https://yoninazarathy.github.io/HypothesisTests.jl/specs/>

The real documentation for the package is at
<https://juliastats.org/HypothesisTests.jl/stable/>. Nothing here is official,
released, or automatically rebuilt; it is a hand-published snapshot and will be
stale the moment the branches move.

The pull requests it previews:

- <https://github.com/JuliaStats/HypothesisTests.jl/pull/367> — additive
- <https://github.com/yoninazarathy/HypothesisTests.jl/pull/2> — breaking fixes
- <https://github.com/yoninazarathy/HypothesisTests.jl/pull/3> — these docs

This branch previously held the copy of upstream's `gh-pages` inherited when the
fork was made, at commit `29fcd58b51ac3d9ff04400198cac955c58fd1a0d`, which is
still present upstream and can be restored from there.
