# Results

Data: **Synthetic data** from `ktrace.simulator` (seeded). All numbers below come from `ktrace report`; rerun it to reproduce them.

## 1. Predicting the next answer

Log: 119,653 answers from 400 learners over 30 days, 120 words in 8 IELTS topics. Models fit on 280 training learners before day 18.3; scored on 14,695 later answers of 120 held-out learners.

| Model | AUC | Log loss | ECE | AUC (reviews only) | Beats both baselines |
| --- | --- | --- | --- | --- | --- |
| Global mean (majority class) | 0.500 | 0.690 | 0.060 | 0.500 | (baseline) |
| Item mean | 0.669 | 0.642 | 0.050 | 0.653 | (baseline) |
| BKT (per skill) | 0.678 | 0.634 | 0.016 | 0.645 | yes |
| BKT (per word) | 0.761 | 0.662 | 0.097 | 0.764 | no |
| PFA (logistic) | 0.844 | 0.490 | 0.050 | 0.855 | yes |
| HLR (half-life regression) | 0.845 | 0.497 | 0.056 | 0.855 | yes |
| True recall (simulator) | 0.864 | 0.452 | 0.005 | 0.874 | (ceiling) |

`True recall (simulator)` scores the simulator's hidden P(correct): no model can beat it on average, because answers are random draws from it. ECE = expected calibration error (10 bins).

AUC per IELTS topic:

| Topic | Item mean | BKT (per skill) | BKT (per word) | PFA (logistic) | HLR (half-life regression) | True recall (simulator) |
| --- | --- | --- | --- | --- | --- | --- |
| crime | 0.640 | 0.606 | 0.729 | 0.802 | 0.795 | 0.817 |
| education | 0.578 | 0.680 | 0.802 | 0.869 | 0.870 | 0.899 |
| environment | 0.674 | 0.709 | 0.863 | 0.893 | 0.897 | 0.919 |
| health | 0.624 | 0.650 | 0.734 | 0.852 | 0.855 | 0.872 |
| media | 0.586 | 0.612 | 0.606 | 0.760 | 0.758 | 0.777 |
| technology | 0.606 | 0.639 | 0.743 | 0.827 | 0.822 | 0.839 |
| travel | 0.596 | 0.617 | 0.696 | 0.799 | 0.800 | 0.819 |
| work | 0.639 | 0.636 | 0.730 | 0.827 | 0.826 | 0.834 |

Learned HLR weights (log2 half-life = w · x): `bias` -0.23, `sqrt_succ_item` +2.34, `sqrt_fail_item` -0.87, `item_difficulty` -0.59, `learner_rate` +0.61.

## 2. Scheduling reviews

Course: 30 new learners per cohort, 30 days, 5 new words a day, at most 20 reviews a day, final test 7 days after the last session. 20 cohorts (seeds); every scheduler meets the same cohorts. Final-test recall = mean true P(correct) over the course words.

| Scheduler | Final-test recall | 95% CI | Reviews used per day |
| --- | --- | --- | --- |
| Leitner boxes | 0.407 | 0.385 to 0.429 | 15.6 |
| SM-2 | 0.430 | 0.410 to 0.451 | 15.6 |
| Model-based (HLR) | 0.543 | 0.531 to 0.556 | 17.4 |
| Model-based (true recall) | 0.603 | 0.591 to 0.615 | 16.8 |

Paired difference, same cohorts:

- Model-based (HLR) minus Leitner boxes: +0.136 (95% CI +0.126 to +0.147)
- Model-based (HLR) minus SM-2: +0.113 (95% CI +0.104 to +0.122)
- Model-based (HLR) minus Model-based (true recall): -0.060 (95% CI -0.062 to -0.058)

Sensitivity to the daily review budget: mean final-test recall (reviews used per day).

| Budget per day | Leitner boxes | SM-2 | Model-based (HLR) | Model-based (true recall) |
| --- | --- | --- | --- | --- |
| 10 | 0.224 (8.8) | 0.250 (8.9) | 0.356 (9.3) | 0.417 (9.2) |
| 20 | 0.407 (15.6) | 0.430 (15.6) | 0.543 (17.4) | 0.603 (16.8) |
| 40 | 0.545 (20.7) | 0.536 (20.0) | 0.721 (26.9) | 0.718 (24.2) |
