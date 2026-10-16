# How it works

This page follows the data from the simulator to a review list, and explains the maths of each
model on the way. File references point to `src/ktrace/`.

## 1. The world: a learner simulator (`simulator.py`)

The simulator is the ground truth. The models never see its hidden state, only the practice log.

- **Words:** 120 IELTS words in 8 topics (environment, education, health, technology, work,
  travel, crime, media). Each topic has its own mean difficulty, each word its own difficulty `b`.
- **Learners:** each has an ability `θ ~ N(0, 1)`.
- **Before studying:** a learner already knows a word with probability
  `sigmoid(θ - b - 1.5)`.
- **Memory:** after a word is seen, recall decays exponentially with time:
  `m = 2^(-Δ / h)`, where `Δ` is days since the last time the word was seen and `h` is the
  half-life. The first half-life is 1.5 days for an unknown word and 6 days for a known one,
  multiplied by `exp(0.4 (θ - b))`: able learners and easy words keep memories longer.
- **Successful review:** `h ← h · (1 + 3 · exp(0.4 (θ - b)) · sqrt(1 - m))`. The gain depends on
  `1 - m`: a word you remember perfectly gains almost nothing (no point reviewing it), a word
  that was half forgotten gains a lot. This is the **spacing effect**, planted on purpose.
- **Failed review (lapse):** `h ← max(0.4 · h, fresh half-life)`. The learner sees the answer,
  so the word is at least as strong as after first studying it.
- **Answers are noisy:** `P(correct) = 0.05 + 0.92 · m` (5% lucky guesses, 3% slips).

`generate_log` writes practice logs for 400 learners over 30 days. Each learner has their own
study habit (studies on 50-95% of days) and pace (3-7 new words a day). Reviews are chosen at
random, with recently seen words more likely. Random reviews matter: they create a wide spread of
gaps between reviews, from hours to weeks, and a model can only learn how memory decays if it
sees many different gaps. Each row also stores `p_true`, the hidden P(correct), used only to
score the "true recall" ceiling.

## 2. The split: by learner and by time (`data/split.py`)

- 30% of learners are **test learners**. Models never fit on any of their answers.
- A **cutoff time** (the 60% quantile of all timestamps, day 18.3) cuts training rows: training
  learners' answers after the cutoff are dropped.
- Test learners' answers **before** the cutoff are context: the models read them to predict the
  next answers (that is the task), but they are not scored. Answers **after** the cutoff are
  scored.

So a model is always predicting new people, later in time.

## 3. Features: only the past (`features.py`)

`LearnerHistory` keeps running counts for one learner. For every answer, `build_features` first
asks `features(item, skill, t)` and only then calls `update(...)` with the answer. A feature row
therefore cannot contain its own label or anything later.

| Feature | Meaning |
| --- | --- |
| `succ_item`, `fail_item` | earlier right and wrong answers on this word |
| `succ_skill`, `fail_skill` | earlier right and wrong answers on this topic |
| `delta_days` | days since this word was last seen (0 if never seen) |
| `first_seen` | 1 if the learner has never seen this word |
| `item_difficulty` | `-log((s+1)/(f+1))` from training learners only |
| `learner_rate` | `log((s+1)/(f+1))` over the learner's earlier answers |

The same class is used by the scheduler (`predictor.py → LearnerPredictor`) and by the API, so
training and serving cannot compute features differently.

Two tests protect this: `test_features_do_not_change_when_the_future_changes` flips every answer
after a cutoff and checks that features up to the cutoff do not move, and
`test_test_labels_never_reach_training` flips every scored answer and checks that the fitted HLR
weights and the difficulty map are identical.

## 4. The models

### Baselines (`models/baselines.py`)

- **Global mean:** always the training success rate. It is the majority-class baseline in
  probability form (a hard 0/1 vote would have infinite log loss). AUC 0.5 by definition.
- **Item mean:** each word's training success rate, shrunk towards the global rate with 5
  pseudo-answers.

### Bayesian Knowledge Tracing (`models/bkt.py`)

A hidden Markov model with two states per learner and knowledge component (KC): unknown (0) and
known (1). Parameters: `L0 = P(known at start)`, `T = P(learn)`, `G = P(guess)`, `S = P(slip)`.

**Prediction and update.** With `k = P(known)` before an answer:

```
P(correct)                = k (1 - S) + (1 - k) G
P(known | correct)        = k (1 - S) / P(correct)
P(known | wrong)          = k S / (k S + (1 - k)(1 - G))
P(known before next one)  = posterior + (1 - posterior) T
```

Hand example (tested): `k = 0.3, T = 0.1, G = 0.2, S = 0.1` and a correct answer:
`P(correct) = 0.27 + 0.14 = 0.41`, posterior `0.27 / 0.41 = 0.659`, next `0.659 + 0.341 · 0.1 = 0.693`.

**Fitting with EM (Baum-Welch).** All sequences of one KC are padded into a matrix
(`to_sequences`), so each time step is one vectorised numpy operation over all learners.

- *Forward pass* (`_forward`): `α_t(s) ∝ P(state_t = s, answers_1..t)`, normalised at every step.
  The normaliser `c_t = P(answer_t | answers before t)`, so `log-likelihood = Σ log c_t`.
  Padded cells get emission 1 and add `log 1 = 0`.
- *Backward pass* (`em_step`): `β_t(i) = Σ_j A(i,j) e_{t+1}(j) β_{t+1}(j) / c_{t+1}`, with the
  transition matrix `A = [[1-T, T], [0, 1]]`.
- *Posteriors:* `γ_t = α_t β_t` = P(state at t | all answers). Expected learning events:
  `ξ_t = α_t(0) · T · e_{t+1}(1) · β_{t+1}(1) / c_{t+1}`.
- *M-step:* `L0 = mean γ_1(known)`, `T = Σ ξ / Σ γ(unknown)` over steps that have a next answer,
  `G = Σ γ(unknown)·correct / Σ γ(unknown)`, `S = Σ γ(known)·wrong / Σ γ(known)`.

EM cannot lower the likelihood; `test_em_never_lowers_likelihood_and_recovers_parameters`
checks it on data generated from known parameters. Guess and slip are capped at 0.5, otherwise
"known" could start to mean "more likely wrong" (a known identifiability problem).

`BKT()` is one model per IELTS topic (classic BKT). `BKT(kc="item_id")` is one model per word.

### PFA-style logistic model (`models/pfa.py`)

Logistic regression on `log(1 + counts)` per word and topic, `log(1 + Δ)`, `first_seen`, item
difficulty and learner rate, standardised. It learns "time hurts" from the data, but the shape of
forgetting is whatever a straight line in `log(1 + Δ)` can express.

### Half-life regression (`models/hlr.py`)

Settles & Meeder (2016): `h = 2^(w · x)` and `p = 2^(-Δ / h)`, with
`x = [1, sqrt(1 + successes), sqrt(1 + failures), item difficulty, learner rate]`.

The loss is mean log loss on 0/1 answers plus `l2 · |w[1:]|²`. Its gradient:

```
ln p = -Δ ln2 · 2^(-w·x)    →    dp/dw = p · ln2² · (Δ / h) · x
dL/dp = (p - y) / (p (1 - p))
```

`test_gradient_matches_finite_differences` checks this, and SciPy's L-BFGS does the fitting.
The half-life is kept between 15 minutes and one year, and `p` in [0.01, 0.99]. First
encounters have no `Δ`, so a small logistic model on difficulty and learner rate covers them.

The learned weights are readable: in the latest run, each `sqrt(1 + successes)` step multiplies
the half-life by `2^2.34 ≈ 5`, and failures shorten it (see docs/results.md).

### DKT (optional, `models/dkt.py`)

A GRU over (word, right/wrong) tokens with inputs shifted by one step. Needs `pip install
".[deep]"`; the test skips without torch. It was **not run** in the environment this repo was
built in (no torch available there), so it has no numbers in the results.

## 5. Metrics (`metrics.py`)

AUC (ranking), log loss (probability quality), ECE and a reliability table (10 bins), and AUC per
topic. Log loss and calibration matter here because the scheduler acts on the probability itself.

## 6. Scheduling (`scheduler.py`)

Each simulated day at 18:00 a scheduler picks at most `budget` words to review, then the learner
studies 5 new words. All schedulers see only answers, never the hidden memory.

- **Leitner:** boxes with intervals 1, 2, 4, 8, 16 days. Right → next box, wrong → box 1. Due
  words, lowest box first.
- **SM-2:** intervals 1, 6, then `interval × ease`; ease starts at 2.5. A wrong answer restarts
  the word at 1 day and lowers the ease by 0.2 (minimum 1.3), as Anki does. Most overdue first.
- **Model-based:** predict P(recall) now for every word seen so far (HLR through
  `LearnerPredictor`), keep those below 0.85, closest to 0.85 first (`select_reviews`).
- **Model-based with true recall:** the same rule on the simulator's hidden P(correct). Not a
  real product: it shows how much better a perfect model would do.

`simulate_course` runs 30 new learners (a cohort) through 30 days, then measures the expected
score on a final test of all 120 words, 7 days after the last session. `compare_schedulers`
repeats this for 20 cohorts. Every scheduler gets the **same** cohorts: same abilities and the
same random stream per learner (`World.rngs`), so differences come from the schedule, and the
report gives paired differences with t-intervals.

## 7. Serving (`api.py`)

Stateless: the client sends the learner's history with each request.

- `POST /predict`: history + queries → P(recall) per query, using only events strictly before
  each query's time (tested by adding future events and checking nothing changes).
- `POST /next-reviews`: history + `now` + `budget` + `target` → the review list, using
  `select_reviews`, the same function as the simulation.

The model artifact is a small JSON file (`ModelBundle`): HLR weights and the difficulty map.
