# Design decisions

Short records of the main choices: what was decided, why, and what it costs. Format inspired by
Architecture Decision Records (ADRs).

## 1. A simulator as the ground truth

- **Decision:** a seeded learner simulator with half-life memory and a spacing effect; real
  datasets only through loaders.
- **Why:** CI has no internet and the real datasets need manual download. A simulator also gives
  something no real dataset has: the true recall probability. That lets the report show a
  ceiling for prediction and a "perfect model" scheduler.
- **Cost:** the simulator's memory has the same shape that HLR assumes, so HLR and the
  model-based scheduler start with an advantage. The README says so next to the results.

## 2. Split by learner and by time

- **Decision:** 30% of learners held out; training rows end at a cutoff time; only later answers
  of held-out learners are scored.
- **Why:** a product predicts new learners in the future. A random row split would let a model
  memorise each learner and read the future.
- **Cost:** fewer training rows (answers after the cutoff from training learners are dropped).

## 3. One feature class for training, simulation and serving

- **Decision:** `LearnerHistory` computes features before seeing the answer, and is reused by
  the scheduler and the API.
- **Why:** leakage-free by construction and no training/serving skew.
- **Cost:** a Python loop over rows (about one second for 120,000 answers). Fine at this size;
  a vectorised group-wise cumulative sum would be needed for millions of rows.

## 4. BKT per skill, and per word as an extra

- **Decision:** classic BKT per IELTS topic, plus a per-word variant.
- **Why:** per-topic BKT is the textbook setup. But for vocabulary, knowing one environment word
  says little about another, so a per-word KC ranks answers much better (AUC 0.761 vs 0.678).
- **Cost:** neither BKT models forgetting, so both are poorly calibrated on time-spaced reviews.
  The per-word version does not beat the item-mean baseline on log loss (0.662 vs 0.642); the
  report marks it "no" instead of hiding it.

## 5. HLR with log loss and bounded probabilities

- **Decision:** fit HLR on 0/1 answers with log loss; keep `p` in [0.01, 0.99].
- **Why:** the paper fits a per-session recall fraction with squared error; this log has one
  answer per row. Pure `2^(-Δ/h)` goes to 0 for long gaps, but a lucky guess is always possible,
  and one such answer would cost a lot of log loss.
- **Cost:** a small departure from the paper, written in the docstring.

## 6. Model-based rule: just below the target, closest first

- **Decision:** review words with predicted recall below 0.85, starting with those closest to
  0.85.
- **Why:** a word just below the target gains a lot from a review (the spacing effect) and will
  most likely be answered right. A word near zero needs re-learning, which costs the same review
  for a smaller gain.
- **Cost:** a learner who falls far behind may see some lost words late. "Lowest recall first"
  is a sensible alternative and is listed under next steps.

## 7. HLR drives the scheduler, not PFA

- **Decision:** the scheduler and API use HLR even though PFA has slightly better log loss.
- **Why:** a scheduler asks "what is recall at time t?" for many future times. HLR answers that
  by design (`p = 2^(-Δ/h)` falls smoothly with time); PFA's time effect is one learned weight on
  `log(1 + Δ)`. The two tie on AUC (0.845 vs 0.844).
- **Cost:** HLR predicts recall a little too low in the middle range, which makes it review
  some words early. That is part of its 0.06 gap to the true-recall scheduler.

## 8. Paired simulation with confidence intervals

- **Decision:** 20 cohorts; each scheduler meets the same learners with the same random stream;
  report mean, 95% t-interval and paired differences; repeat at budgets 10, 20 and 40.
- **Why:** a single run can mislead. Pairing removes most of the cohort-to-cohort noise, so
  small differences become measurable. The budget check shows if a gain only comes from using
  more reviews.
- **Cost:** about two minutes for `ktrace report`.

## 9. Expected final-test score, not a sampled one

- **Decision:** retention = mean true P(correct) over all course words at test time.
- **Why:** same meaning as giving the test, with less noise.

## 10. DKT only as an optional extra

- **Decision:** `models/dkt.py` needs `pip install ".[deep]"`; the test skips without torch,
  and the file is left out of coverage.
- **Why:** torch would make CI slow and heavy. On this data, a deep model is not needed to show
  the point.
- **Cost:** DKT was not run in the environment where this repo was built (torch could not be
  installed there), so it has no reported numbers. Run `pytest tests/test_dkt.py` after
  installing the extra.

## 11. Stateless API with a JSON model file

- **Decision:** the client sends the history; the model is a small JSON file.
- **Why:** no database, easy to test, the artifact can be read in a code review, no pickle.
- **Cost:** long histories mean larger requests; a real app would cache learner state.
