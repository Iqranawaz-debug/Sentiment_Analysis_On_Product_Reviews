# Review Sentiment Lab model card

## Intended use
English product-review triage and educational exploration, with human review.
Not suitable for automated decisions about people or as a customer-satisfaction measure.

## Data
Amazon Fine Food Reviews and Datafiniti consumer-product reviews, supplied locally.
Ratings map to negative (low), neutral (middle), and positive (high). These proxy
labels can disagree with expressed sentiment. Source hashes and exclusions are
recorded in generated reports. Normalized duplicates are removed globally before
splitting; all conflicting-label duplicate groups are excluded.

## Protocol and metrics
See `reports/metrics.json` and `reports/comparison.csv` after running training.
No performance claim is made without generated evidence. Training, validation and
test partitions are stratified. A training-only calibration subset is kept separate
from fitting and selection. Feature vocabularies are fitted inside each CV fold.
Validation tunes regularization; test results never select a model.

## Known failure modes
Sarcasm, mixed aspects, ambiguous neutral labels, new vocabulary, spelling changes,
domain shift and non-English inputs can cause errors. Source-specific test metrics
describe within-mixture performance, not transfer to an unseen domain. Random splits
do not isolate products, authors or time; near duplicates may remain.

## Ethics and operation
Review corpora reflect selection and demographic biases. Confidence is not a guarantee
of correctness. Review text may contain personal information: do not redistribute raw
data or generated error samples without checking source terms and privacy. The app
processes uploads in memory. Only load trusted joblib artifacts. There is no operational
monitoring, authentication or automatic retraining.
