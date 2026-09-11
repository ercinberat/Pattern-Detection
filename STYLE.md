# STYLE.md — Rules for Writing Code in This Repo

These rules apply to anyone (or anything) writing code in this repository,
including Claude Code. They take priority over general best-practice
instincts when the two conflict — follow what's written here.

---

## 1. Never delete without asking

Do not delete files, functions, tests, or blocks of working code as part of
a change — even if they look unused, redundant, or superseded by new work.

- If something seems safe to remove, say so and ask first. Explain what it
  is and why you think it can go.
- This includes deleting via a rewrite — if a planned edit would drop
  existing logic rather than extend it, stop and ask before writing it.
- Commented-out old code, unused imports, dead functions: flag them, don't
  silently remove them.

## 2. Stick to the plan — ask before improving it

`PLAN.md` defines what gets built and in what order. When implementing a
stage:

- Build what the plan describes for that stage. Don't add extra scope,
  swap in a different approach, or "improve" the design mid-implementation
  without asking first — even if the improvement is genuinely better.
- If you notice a better way to do something, or a gap in the plan, raise
  it as a question or a suggested edit to `PLAN.md` — don't just implement
  the change and explain it after the fact.
- Small implementation details not specified in the plan (variable names,
  helper function structure) are fine to decide on your own. Anything that
  changes what the tool does, how a stage works, or what gets built next is
  not.

## 3. Write for an engineer, not a software engineer

The reader is technically capable and comfortable with logic, math, and
trading concepts — but is not a professional software developer. Code
should be readable by someone who knows *what* the code should do, without
requiring them to be fluent in software design patterns to follow *how* it
does it.

Concretely, this means:

- Prefer straightforward, linear code over clever abstractions. If a
  simpler, slightly more verbose version is easier to follow, use it.
- Avoid unnecessary layers of indirection (extra classes, decorators,
  metaprogramming, deep inheritance) unless the problem genuinely needs
  them.
- It's fine to repeat a few lines of code if the alternative is an
  abstraction that's harder to follow than the repetition itself.
- If a design pattern or library feature is used, add a short comment
  explaining what it does in plain terms — don't assume the reader already
  knows it.

## 4. Use clear, descriptive names

- Variable and function names should say what they hold or do, in plain
  language — not abbreviations or single letters, except for very
  short-lived loop counters or well-known math symbols (`i`, `x`, `y` in a
  tight local context is fine; `hp`, `flg_r`, `trg` is not).
- Prefer `pole_return_pct` over `pr`, `flag_volume_ratio` over `fvr`,
  `is_symmetrical_triangle` over `is_sym`.
- Function names should describe the action or result:
  `detect_bull_flags`, not `process` or `run`.
- If a name needs a comment to explain what it actually means, rename it
  instead.

## 5. Comment often, and comment clearly

- Every function should have a short docstring or comment explaining what
  it does, in plain terms — not just restating the code, but saying *why*
  it exists and what problem it solves.
- Comment non-obvious logic inline, especially: threshold choices (why
  this number), financial/trading concepts (what a squeeze, pole, or
  contraction ratio means), and anywhere the code diverges from the
  obvious/naive approach.
- Prefer a few extra comments over too few. If in doubt, explain it.
- Comments should stay in plain, non-jargon language wherever possible —
  same audience as rule 3.

---

## Quick checklist before committing

- [ ] Did I delete anything? If yes, did I ask first?
- [ ] Does this match what `PLAN.md` says for this stage? If I changed or
      improved on the plan, did I ask first?
- [ ] Could someone who isn't a software engineer follow this code?
- [ ] Are all names descriptive and unambiguous?
- [ ] Are functions and non-obvious logic commented clearly?
