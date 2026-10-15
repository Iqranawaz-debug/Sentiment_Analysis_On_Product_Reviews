# Short-review diagnosis

Reproduce with `python -m scripts.diagnose_short_reviews`. This only loads the existing
model and writes `reports/short_review_sanity.json`; it never trains or changes artifacts.
The JSON contains all 24 authored cases, expected labels, all probabilities, raw
coefficients, TF-IDF values, contributions, intercepts and the model SHA256.

## Findings

The exact input `Absolutely hate it` yields positive 0.6488919076, negative
0.3411401734, neutral 0.0099679190. This is a real model error, not an old prediction.
Before the fix, committed text changes recomputed automatically, but uncommitted
browser edits could leave the previous result on screen.

Live Review now clears results on submitted edits and asks the user to press
Read the mood. While the text area has focus, CSS hides the old verdict and mascot
and displays an editing notice. Browser testing confirmed the old mascot disappears
while typing. Streamlit may consume the first click to submit a focused text edit;
if the changed-review notice remains, press Read the mood after the edit is applied.
Sample buttons still populate and analyze with one click.

Below `SENTIX_UNSURE_THRESHOLD` (default 0.70), the live mascot shows a confused face,
question mark and unsure message, and the UI lists the two leading classes.
The label and all model probabilities remain unmodified. The threshold is a UI
choice, not a calibrated guarantee of correctness.

| Feature | Negative contribution | Neutral contribution | Positive contribution |
|---|---:|---:|---:|
| absolutely | 0.086706 | -0.474761 | 0.388054 |
| hate | 0.364727 | -0.358712 | -0.006015 |
| it | 0.015994 | -0.019993 | 0.003999 |
| hate it | 0.287840 | -0.177119 | -0.110722 |
| intercept | -0.004618 | -0.721979 | 0.726597 |
| total raw score | 0.750649 | -1.752563 | 1.001914 |

Contributions are coefficient times TF-IDF, not probability percentage points.
The phrase `absolutely hate` is absent from the fitted vocabulary. `Absolutely`
supports positive strongly, and the positive intercept outweighs the stronger
negative feature evidence. Sigmoid calibration then produces the probabilities.
This decomposition is exact for the linear scores, not causal evidence.

## Sanity evaluation

14/24 correct (58.33%); positive 7/8, negative 5/8, neutral 2/8. These are
hand-authored expectations, not an independently sampled benchmark. Short neutral
phrases, especially `mediocre`, may admit different human labels. No case was
removed after observing a prediction. Every failure:

| Review | Expected | Predicted | Predicted-class probability |
|---|---|---|---:|
| Not bad at all | positive | negative | 83.94% |
| Absolutely hate it | negative | positive | 64.89% |
| I hate it | negative | positive | 61.74% |
| It broke immediately | negative | positive | 49.81% |
| It is average | neutral | positive | 62.22% |
| Nothing special | neutral | negative | 57.06% |
| Neither good nor bad | neutral | negative | 74.93% |
| An ordinary product | neutral | positive | 61.59% |
| Average quality | neutral | positive | 76.14% |
| It is mediocre | neutral | positive | 79.24% |

`Not good`, `Does not work`, and `I do not recommend it` were correctly negative.
The unsure threshold catches six of the ten failures, but four remain confident
errors. It also marks one correct prediction unsure. It does not improve accuracy.

## Proposed experiments — approval required, none applied

1. Collect and adjudicate short real reviews, especially negative, neutral and
   negation examples; retain varied longer reviews. Use independent train,
   validation and test groups, separating phrase families and near duplicates.
   Expected benefit: better coverage of short phrases. Trade-off: annotation cost
   and risk of distorting the original domain distribution. Keep these 24 cases
   as an exposed regression set, not a final generalization benchmark.
2. Compare vocabulary coverage and regularization on training/validation only.
   The current model already uses word unigrams/bigrams and balanced class weights.
   Lower min_df or higher vocabulary caps may retain useful phrases, but also add
   noise, memory use and overfitting. Trying trigrams/negation-scope features may
   help compositional errors but increases sparsity and can misread negation scope.
3. Audit rating-derived neutral labels and evaluate by review length. Recalibrate
   using a separate representative short-review calibration set, and select any
   abstention threshold on validation data. Better uncertainty may reduce misleading
   displays but cannot repair missing semantic features and reduces coverage.

Do not change the intercept by hand or add a `hate` keyword override. Those can
hide this example while damaging phrases such as `I thought I would hate it, but
I love it`. No improvement percentage is known until controlled evaluation.
