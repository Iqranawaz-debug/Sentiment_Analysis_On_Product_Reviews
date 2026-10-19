# Approved sentiment experiments — 2026-10-16

Run started 2026-10-16; final artifact and routing verification completed 2026-10-19.

Production artifacts were read only. Every trained candidate is stored separately under `models/candidates/approved_20261016/`.

## Protocol and limitations

Excluded all 30,000 rows from the production sample (training, calibration, validation, and old test). Source checksums matched the original report.
New split sizes: {'fit': 10073, 'calibration': 2519, 'validation': 2701, 'test': 2698}. Removed 9 sampled rows in near-duplicate components connected to production.
Duplicate screen: nonnegative hashed unigram/bigram cosine >=0.92; connected components; no exhaustive semantic paraphrase guarantee. Connected components stay in one split. This does not establish product/user-level or semantic independence.
The 24 exposed cases were excluded by normalized text, never used for fitting, calibration, hyperparameter selection, or threshold choice. Their scores are regression evidence only.
The new pool contains 16,000 randomly sampled unused real reviews plus up to 2,000 unused short reviews. Its test distribution is deliberately short-review enriched; labels derive from stars (1–2 negative, 3 neutral, 4–5 positive). This is not a deployment prevalence estimate or independently human-labeled test benchmark.
Test class counts: {'positive': 2104, 'negative': 381, 'neutral': 213}; short (<=8 normalized whitespace words): 316; long: 2382.
Short-test class counts: {'positive': 277, 'negative': 25, 'neutral': 14}. Short accuracy must be read alongside this class imbalance.
The short-review audit is one assistant judgment on 36 training-only rows, not independent human adjudication. Ambiguous audited examples are excluded from fitting; no validation or test labels were edited. Remaining labels are rating proxies.
Candidate training sizes differ from production. Comparisons against production combine data and configuration effects; baseline_fresh versus extra_short isolates added sampled short rows more closely.

## Comparison

| Model | Test accuracy | Macro-F1 | Short accuracy | Long accuracy | Regression /24 | Model MB | ms/review |
|---|---:|---:|---:|---:|---:|---:|---:|
| production | 85.58% | 0.5878 | 89.87% | 85.01% | 14/24 | 1.70 | 0.288 |
| baseline_fresh | 85.84% | 0.6180 | 89.24% | 85.39% | 12/24 | 1.69 | 0.296 |
| extra_short | 85.99% | 0.6179 | 89.56% | 85.52% | 12/24 | 1.69 | 0.273 |
| coverage_c1 | 85.54% | 0.6030 | 89.87% | 84.97% | 12/24 | 2.67 | 0.386 |
| trigrams_c2 | 85.69% | 0.6017 | 89.56% | 85.18% | 12/24 | 2.68 | 0.408 |
| short_calibration | 83.43% | 0.5124 | 90.51% | 82.49% | 13/24 | 2.68 | 0.532 |
| transformer | 75.87% | 0.5798 | 72.47% | 76.32% | 17/24 | 243.51 | 256.331 |
| hybrid_production | 84.21% | 0.6489 | 87.34% | 83.80% | 20/24 | 245.22 | 62.151 |
| hybrid_trigrams_c2 | 83.73% | 0.6299 | 88.29% | 83.12% | 18/24 | 246.20 | 59.944 |

Timings are measured warm, CPU batch throughput divided by review count, not interactive single-request latency; exclude model loading. Transformer results use dynamic-int8 Linear layers, two CPU threads, batch 8, label-blind length sorting, and truncation to 256 tokens. Hybrid quality is scored by routing the stored model probabilities at the fixed threshold; conditional timing is measured separately by running TF-IDF on every row and the int8 transformer only on deferred rows. Dynamic quantization can produce small numerical differences with different batches. Model sizes exclude Python/dependency environments; the original downloaded full-precision weights are also retained separately.

## Validation-only selection and calibration

Selected candidate before test/regression evaluation: **trigrams_c2** (maximum validation macro-F1).

| Candidate | Validation macro-F1 | Fit rows | Calibration rows | Training seconds |
|---|---:|---:|---:|---:|
| baseline_fresh | 0.5920 | 8941 | 2252 | 15.66 |
| extra_short | 0.5949 | 10066 | 2519 | 16.41 |
| coverage_c1 | 0.5893 | 10066 | 2519 | 17.16 |
| trigrams_c2 | 0.6032 | 10066 | 2519 | 14.57 |
| short_calibration | 0.5129 | 10066 | 277 | 12.30 |

Validation abstention threshold: 0.6. Rule: lowest prespecified threshold achieving 90% accepted accuracy. This threshold is diagnostic only and was not applied to production or the fixed 0.80 hybrid.
```json
[
  {
    "threshold": 0.5,
    "coverage": 0.9711218067382451,
    "accuracy": 0.881433473122379
  },
  {
    "threshold": 0.6,
    "coverage": 0.9052202887819326,
    "accuracy": 0.9051124744376278
  },
  {
    "threshold": 0.7,
    "coverage": 0.8352462051092188,
    "accuracy": 0.9299645390070922
  },
  {
    "threshold": 0.8,
    "coverage": 0.7382450944094779,
    "accuracy": 0.958876629889669
  },
  {
    "threshold": 0.9,
    "coverage": 0.5264716771566087,
    "accuracy": 0.9845288326300985
  },
  {
    "threshold": 0.95,
    "coverage": 0.25509070714550164,
    "accuracy": 0.988388969521045
  }
]
```

Short-only calibration class counts: {'positive': 253, 'negative': 14, 'neutral': 10}. The small negative/neutral support limits this experiment; it is not evidence that all short-review calibration strategies fail.


## Calibration diagnostics

Lower log loss, multiclass Brier sum, and 10-bin expected calibration error (ECE) are better. ECE depends on binning and class mix.

| Model | Test log loss | Test Brier | Test ECE |
|---|---:|---:|---:|
| production | 0.3995 | 0.2074 | 0.0512 |
| baseline_fresh | 0.4177 | 0.2153 | 0.0319 |
| extra_short | 0.4154 | 0.2149 | 0.0225 |
| coverage_c1 | 0.4182 | 0.2178 | 0.0244 |
| trigrams_c2 | 0.4136 | 0.2149 | 0.0269 |
| short_calibration | 0.4433 | 0.2383 | 0.0602 |
| transformer | 0.5502 | 0.3129 | 0.0845 |
| hybrid_production | 0.4023 | 0.2127 | 0.0558 |
| hybrid_trigrams_c2 | 0.4145 | 0.2199 | 0.0422 |
## production

| Class | Test precision | Test recall | Support |
|---|---:|---:|---:|
| negative | 0.7250 | 0.6089 | 381 |
| neutral | 0.4510 | 0.1080 | 213 |
| positive | 0.8827 | 0.9762 | 2104 |

Confusion matrices: rows actual, columns predicted; order negative, neutral, positive.

Test:
```json
[[232, 13, 136], [53, 23, 137], [35, 15, 2054]]
```
Regression:
```json
[[5, 0, 3], [2, 2, 4], [1, 0, 7]]
```

Regression failures:
- Not bad at all: positive → negative
- Absolutely hate it: negative → positive
- I hate it: negative → positive
- It broke immediately: negative → positive
- It is average: neutral → positive
- Nothing special: neutral → negative
- Neither good nor bad: neutral → negative
- An ordinary product: neutral → positive
- Average quality: neutral → positive
- It is mediocre: neutral → positive
## baseline_fresh

| Class | Test precision | Test recall | Support |
|---|---:|---:|---:|
| negative | 0.7584 | 0.5932 | 381 |
| neutral | 0.5362 | 0.1737 | 213 |
| positive | 0.8807 | 0.9758 | 2104 |

Confusion matrices: rows actual, columns predicted; order negative, neutral, positive.

Test:
```json
[[226, 12, 143], [41, 37, 135], [31, 20, 2053]]
```
Regression:
```json
[[5, 0, 3], [0, 0, 8], [1, 0, 7]]
```

Regression failures:
- Not bad at all: positive → negative
- Absolutely hate it: negative → positive
- I hate it: negative → positive
- It broke immediately: negative → positive
- It is average: neutral → positive
- It is okay: neutral → positive
- Nothing special: neutral → positive
- Neither good nor bad: neutral → positive
- An ordinary product: neutral → positive
- It is just acceptable: neutral → positive
- Average quality: neutral → positive
- It is mediocre: neutral → positive
## extra_short

| Class | Test precision | Test recall | Support |
|---|---:|---:|---:|
| negative | 0.7592 | 0.5958 | 381 |
| neutral | 0.5538 | 0.1690 | 213 |
| positive | 0.8813 | 0.9777 | 2104 |

Confusion matrices: rows actual, columns predicted; order negative, neutral, positive.

Test:
```json
[[227, 14, 140], [40, 36, 137], [32, 15, 2057]]
```
Regression:
```json
[[5, 0, 3], [0, 0, 8], [1, 0, 7]]
```

Regression failures:
- Not bad at all: positive → negative
- Absolutely hate it: negative → positive
- I hate it: negative → positive
- It broke immediately: negative → positive
- It is average: neutral → positive
- It is okay: neutral → positive
- Nothing special: neutral → positive
- Neither good nor bad: neutral → positive
- An ordinary product: neutral → positive
- It is just acceptable: neutral → positive
- Average quality: neutral → positive
- It is mediocre: neutral → positive
## coverage_c1

| Class | Test precision | Test recall | Support |
|---|---:|---:|---:|
| negative | 0.7535 | 0.5696 | 381 |
| neutral | 0.5517 | 0.1502 | 213 |
| positive | 0.8754 | 0.9786 | 2104 |

Confusion matrices: rows actual, columns predicted; order negative, neutral, positive.

Test:
```json
[[217, 11, 153], [41, 32, 140], [30, 15, 2059]]
```
Regression:
```json
[[5, 0, 3], [0, 0, 8], [1, 0, 7]]
```

Regression failures:
- Not bad at all: positive → negative
- Absolutely hate it: negative → positive
- I hate it: negative → positive
- It broke immediately: negative → positive
- It is average: neutral → positive
- It is okay: neutral → positive
- Nothing special: neutral → positive
- Neither good nor bad: neutral → positive
- An ordinary product: neutral → positive
- It is just acceptable: neutral → positive
- Average quality: neutral → positive
- It is mediocre: neutral → positive
## trigrams_c2

| Class | Test precision | Test recall | Support |
|---|---:|---:|---:|
| negative | 0.7362 | 0.5932 | 381 |
| neutral | 0.5172 | 0.1408 | 213 |
| positive | 0.8813 | 0.9772 | 2104 |

Confusion matrices: rows actual, columns predicted; order negative, neutral, positive.

Test:
```json
[[226, 13, 142], [48, 30, 135], [33, 15, 2056]]
```
Regression:
```json
[[5, 0, 3], [0, 0, 8], [1, 0, 7]]
```

Regression failures:
- Not bad at all: positive → negative
- Absolutely hate it: negative → positive
- I hate it: negative → positive
- It broke immediately: negative → positive
- It is average: neutral → positive
- It is okay: neutral → positive
- Nothing special: neutral → positive
- Neither good nor bad: neutral → positive
- An ordinary product: neutral → positive
- It is just acceptable: neutral → positive
- Average quality: neutral → positive
- It is mediocre: neutral → positive
## short_calibration

| Class | Test precision | Test recall | Support |
|---|---:|---:|---:|
| negative | 0.8235 | 0.4042 | 381 |
| neutral | 0.6250 | 0.0469 | 213 |
| positive | 0.8365 | 0.9919 | 2104 |

Confusion matrices: rows actual, columns predicted; order negative, neutral, positive.

Test:
```json
[[154, 2, 225], [20, 10, 183], [13, 4, 2087]]
```
Regression:
```json
[[5, 0, 3], [0, 0, 8], [0, 0, 8]]
```

Regression failures:
- Absolutely hate it: negative → positive
- I hate it: negative → positive
- It broke immediately: negative → positive
- It is average: neutral → positive
- It is okay: neutral → positive
- Nothing special: neutral → positive
- Neither good nor bad: neutral → positive
- An ordinary product: neutral → positive
- It is just acceptable: neutral → positive
- Average quality: neutral → positive
- It is mediocre: neutral → positive
## transformer

| Class | Test precision | Test recall | Support |
|---|---:|---:|---:|
| negative | 0.8131 | 0.4226 | 381 |
| neutral | 0.1963 | 0.5915 | 213 |
| positive | 0.9473 | 0.8365 | 2104 |

Confusion matrices: rows actual, columns predicted; order negative, neutral, positive.

Test:
```json
[[161, 189, 31], [20, 126, 67], [17, 327, 1760]]
```
Regression:
```json
[[5, 3, 0], [1, 5, 2], [0, 1, 7]]
```

Regression failures:
- Not bad at all: positive → neutral
- It broke immediately: negative → neutral
- Not good: negative → neutral
- Does not work: negative → neutral
- It is okay: neutral → positive
- It is just acceptable: neutral → positive
- It is mediocre: neutral → negative
## hybrid_production

| Class | Test precision | Test recall | Support |
|---|---:|---:|---:|
| negative | 0.8292 | 0.5223 | 381 |
| neutral | 0.2936 | 0.4977 | 213 |
| positive | 0.9380 | 0.9349 | 2104 |

Confusion matrices: rows actual, columns predicted; order negative, neutral, positive.

Test:
```json
[[199, 133, 49], [26, 106, 81], [15, 122, 1967]]
```
Regression:
```json
[[7, 1, 0], [1, 6, 1], [1, 0, 7]]
```

Regression failures:
- Not bad at all: positive → negative
- It broke immediately: negative → neutral
- It is just acceptable: neutral → positive
- It is mediocre: neutral → negative

Deferred 26.80% of test reviews.
- accuracy: difference versus TF-IDF -0.0137; versus transformer +0.0834.
- macro_f1: difference versus TF-IDF +0.0611; versus transformer +0.0691.
## hybrid_trigrams_c2

| Class | Test precision | Test recall | Support |
|---|---:|---:|---:|
| negative | 0.8233 | 0.4646 | 381 |
| neutral | 0.2890 | 0.4789 | 213 |
| positive | 0.9296 | 0.9411 | 2104 |

Confusion matrices: rows actual, columns predicted; order negative, neutral, positive.

Test:
```json
[[177, 142, 62], [23, 102, 88], [15, 109, 1980]]
```
Regression:
```json
[[6, 1, 1], [1, 5, 2], [0, 1, 7]]
```

Regression failures:
- Not bad at all: positive → neutral
- Absolutely hate it: negative → positive
- It broke immediately: negative → neutral
- It is okay: neutral → positive
- It is just acceptable: neutral → positive
- It is mediocre: neutral → negative

Deferred 27.32% of test reviews.
- accuracy: difference versus TF-IDF -0.0196; versus transformer +0.0786.
- macro_f1: difference versus TF-IDF +0.0282; versus transformer +0.0501.

## Frozen transformer provenance

Model: cardiffnlp/twitter-roberta-base-sentiment-latest; revision: 3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7; license: cc-by-4.0.
[Pinned model card](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest/blob/3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7/README.md). No fine-tuning was performed here; the upstream model was sentiment-fine-tuned on TweetEval. Its pretraining data overlap with this corpus cannot be independently ruled out.
Downloaded 502,406,167 bytes; model load 14.18s; 143 test rows exceeded 256 tokens and were truncated.
The reported transformer is the separate dynamic-int8 CPU candidate. Full-precision inference was stopped for runtime reasons before the 200-review progress marker; no full-precision aggregate scores are available. Quantization can change predictions. The runtime change was made without inspecting transformer test/regression metrics; see transformer_runtime_adjustment.json.

## Artifact integrity

| Candidate | SHA256 |
|---|---|
| production | 85edf60258b04b218bde4d4f7c954176454ee7f42ddeb6816985e53fd72617a0 |
| baseline_fresh | f8a54c89a9d3a4d8a5ea0664ebaabdb26ac2fb0094aef98c718c58567b776634 |
| extra_short | 33d1ea9a73d99155be7716a5f8833e200d33eef2572ca34b9cc3a923f5df933a |
| coverage_c1 | d2db53e048b21a3f108ea25fb77b23efbaef84a0ef238744dd455815fdcb26ba |
| trigrams_c2 | bf26ba5fa44578ab2c31559eddf40f942af535fb49873d1848e25f25d0136dac |
| short_calibration | b23cd5af88778e25a2bc569b82c76a79d30feef46f93dff97f6879d61b873db8 |
| transformer | dec75d628a2e0d47b3b2dd65d9d695531d375cc297d6747e9406b1ff16e35e95 |
| hybrid_production | 28bb0ac617748618ce8094903c4472289a7e47c6023375f380fe10f8656fe633 |
| hybrid_trigrams_c2 | c005ac6a7c9e23dad06db990f4a1b7cabf0637e722cadbfa997720b32434af3d |

Transformer file hashes and pinned revision are in its training_config.json; hybrids reference both component hashes and their fixed routing threshold. Full per-row probabilities, split hashes, configurations and metrics accompany this report.

## Decision

No model has been promoted. Keep production unchanged for now. If choosing a candidate for the next separately approved pilot, prefer the production-based 0.80 hybrid when macro-F1 and neutral recall matter more than overall accuracy and footprint. It has the highest measured test macro-F1 (0.6489) and 20/24 exposed regression cases correct, but its test accuracy falls from 85.58% to 84.21%, CPU batch throughput costs about 62.15 ms/review versus 0.29 ms, and deployment model assets grow from 1.70 MB to 245.22 MB. These are observed point estimates, not a claim of statistical significance or improved real-world performance.
The hybrid beats production TF-IDF and the int8 transformer on macro-F1; it beats only the transformer on overall accuracy. Neutral recall rises from 10.80% in production to 49.77% in this hybrid, while other class trade-offs remain visible in the tables. The standalone transformer is not recommended on this rating-labeled benchmark. The added-short-review TF-IDF candidate has the highest overall test accuracy (85.99%) but only 12/24 regression cases correct; it also still misclassifies the original hate example.
The int8 transformer and production hybrid classify 'Absolutely hate it' as negative with 79.10% probability. This is exposed regression evidence, not a selection criterion. Sparse linear models offer exact feature-score explanations; deferred transformer decisions do not. The quantization API emitted deprecation warnings, adding a maintenance consideration. A new independently human-labeled review benchmark, especially for neutral and short reviews, should gate any later deployment decision. Do not tune again on this now-inspected test set.
Experiment 1 was a bounded pilot using previously unused real reviews from the existing sources and one assistant's 36-row training audit, not a completed independent human annotation effort. Experiment 3's short-only calibration had only 14 negative and 10 neutral examples. These limits prevent claiming that short-review augmentation or recalibration generally fails.
