# Learning guide (for me, Hai)

This file is my study guide for this repo. Goal: explain every part in an interview, change it
with confidence, and answer "why?" three levels deep. Work through it before pinning the repo.

Start with [docs/how-it-works.md](docs/how-it-works.md) (the code walkthrough and the maths) and
[docs/decisions.md](docs/decisions.md) (the reasons). This file is the practice plan.

## 0. Why this matters to me as a teacher

In my IELTS classes the pattern was always the same. Students "learned" 30 topic words on Monday,
got most of them right on Tuesday, and had lost half by the next week. The students who kept
their vocabulary were not the ones who studied longest the night before. They were the ones who
met each word again a few days later, then a week later, then a few weeks later.

This repo turns that classroom experience into something I can measure:

- **Forgetting is fast and then slows down.** The simulator uses `recall = 2^(-days / half-life)`.
  After a good review the half-life grows, so the next review can wait longer. That is why
  Leitner boxes and Anki intervals grow: 1 day, a few days, a week, a month.
- **The best moment to review is just before you forget** (the spacing effect). Reviewing a word
  a student knows perfectly feels good but teaches little. Reviewing a word they have almost
  lost costs effort, and that effort is what makes memory last. In the simulator the half-life
  gain is `sqrt(1 - recall)`: no gain at 100% recall, a large gain at 50%.
- **Students are different.** In class I gave weaker students more frequent review and let strong
  students move on. A fixed schedule (Leitner, SM-2) cannot do that on its own. A model that
  predicts recall *per student and per word* can.
- **Time is limited.** A student has maybe 15 minutes a day. The real question is not "should
  they review?" but "which 20 words today?". That is the scheduler with a budget.
- **Testing is learning.** Every review here is a recall test followed by the right answer, like
  flashcards or a quick quiz at the start of a lesson. Retrieval practice beats re-reading.

Things I must be careful about when I tell this story: the simulator was built with these
effects, so the simulation can only show that the code finds them and acts on them. It does not
prove the size of the gain with real learners. Only an experiment with real students can.

## 1. Run it and watch each step (1 hour)

```bash
make install
make check                       # lint, types, tests with coverage
ktrace generate --learners 50    # look at data/synthetic_log.csv in a notebook
ktrace train                     # metrics table + artifacts/model.json
cat artifacts/model.json         # the HLR weights: read them, explain each sign
ktrace report                    # docs/results.md and the two charts
ktrace serve                     # http://127.0.0.1:8000/docs, try /next-reviews
```

Then change one thing and see the effect: in `simulator.py` set `spacing_power` to 0 (no spacing
effect) and rerun `ktrace report`. Predict first what happens to the scheduler comparison.

## 2. Read the code in this order

| File | What to understand | Question to answer yourself |
| --- | --- | --- |
| `simulator.py` | half-life memory, spacing effect, lapses, guess/slip | Why does a review at 99% recall barely change the half-life? |
| `data/split.py` | split by learner and by time | Why are test learners' early answers kept but not scored? |
| `features.py` | `features()` before `update()` | What would leak if `update` ran first? |
| `models/baselines.py` | global mean, shrunk item mean | Why is the global-mean AUC exactly 0.5? |
| `models/bkt.py` | prediction/update, forward-backward, M-step | Why does EM never decrease the likelihood? Why cap guess and slip at 0.5? |
| `models/pfa.py` | counts as `log(1 + n)` | Why log counts instead of raw counts? |
| `models/hlr.py` | `h = 2^(w·x)`, `p = 2^(-Δ/h)`, the gradient | Derive `dp/dw` on paper. |
| `metrics.py` | AUC vs log loss vs calibration | Can a model have high AUC and bad calibration? (BKT per word.) |
| `benchmark.py` | fit, predict, score, ceiling | Why is the "true recall" row a ceiling and not 1.0? |
| `scheduler.py` | Leitner, SM-2, model-based, paired simulation | Why give every scheduler the same random stream? |
| `api.py` | stateless endpoints, "only events before the query" | How would you add learner state in a database? |
| `report.py` | every number comes from code | What would you change to report on real data? |

## 3. Key concepts in my own words (fill these in)

- **Knowledge tracing:** ...
- **Hidden Markov model / BKT:** ...
- **EM (expectation-maximisation):** guess the hidden states, re-fit the parameters, repeat ...
- **Forgetting curve and half-life:** ...
- **Spacing effect / desirable difficulty:** ...
- **AUC vs log loss vs calibration:** ...
- **Data leakage in sequence data:** ...
- **Paired comparison and confidence intervals:** ...

## 4. Interview questions this repo answers

1. *How do you evaluate a model that predicts student answers?* Split by learner and time,
   compare with baselines and a ceiling, look at AUC and calibration, check per topic.
2. *What is BKT and how do you fit it?* Two-state HMM, four parameters, EM with forward-backward.
   Walk through the hand example (0.41, 0.659, 0.693).
3. *Why does BKT do poorly here?* No forgetting, and one skill per topic. Show the per-word
   variant: better ranking, still poor calibration.
4. *How does Duolingo schedule reviews?* Half-life regression; explain `p = 2^(-Δ/h)` and what a
   positive weight on successes means.
5. *How do you turn a prediction into a decision?* A budget and a rule (just below 0.85), then
   measure the decision itself, not only the prediction metric.
6. *Is the gain real?* Paired CIs over 20 cohorts, the budget-10 check, and the honest caveat
   that the world was built with these effects. Next step: an A/B test with a delayed quiz.
7. *How would you serve this?* Stateless FastAPI, JSON model, same feature code as training.

## 5. Make it mine (do at least three)

- [ ] Download the Duolingo HLR traces, run `ktrace train --data ... --format duolingo`, and
      compare HLR with PFA on real data.
- [ ] Try the rule "lowest predicted recall first" and targets 0.7, 0.8, 0.9, 0.95; add a chart.
- [ ] Add isotonic calibration to HLR and see if the gap to the true-recall scheduler shrinks.
- [ ] Use my own IELTS topic word lists as the catalog (real words instead of `envi_000`).
- [ ] Install `.[deep]`, run the DKT test, and add DKT to the report with `ktrace report --deep`.
- [ ] Sketch a small pilot for HaiEduTech: which data the app must log, and how to run an A/B test.

## 6. Quality checks I should be able to explain

- `make check` = `ruff check`, `ruff format --check`, `mypy src`, `pytest --cov-fail-under=80`.
- Why the leakage test flips the *test* labels and expects the *fitted model* not to change.
- Why the BKT test checks that predictions do not change when rows are shuffled.

## 7. Honesty checklist before pushing

- [ ] I can explain every file without looking at it.
- [ ] Every number in the README is from `docs/results.md`, which I regenerated myself.
- [ ] The README says the data is synthetic and that the simulator favours HLR-style models.
- [ ] DKT is described as optional and not run; I run it myself before claiming any result.
- [ ] Anything I have not built is under "What I'd do next".

Note: the Docker image was not built in the environment this repo was first made in (no Docker
daemon there). CI builds it; run `docker build -t ktrace .` once yourself before a demo.
