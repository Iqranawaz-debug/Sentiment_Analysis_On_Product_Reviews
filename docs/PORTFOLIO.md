# Portfolio review and LinkedIn material

## What different reviewers can assess

| Audience | Evidence to highlight |
|---|---|
| Data analysts | Distribution, exclusions, review lengths and a downloadable batch summary; explain that predictions are not population sentiment measurements |
| Data scientists | Training-only CV, imbalance-aware selection, dummy baseline, untouched holdout, neutral-class errors and proxy-label limitations |
| ML engineers | One serialized pipeline, version/checksum checks, offline training, bounded inference, configuration and reproducibility |
| Recruiters | Clear problem, functioning demo, concise architecture, reproducible commands and honest results |
| Hiring managers | Judgment: keeping useful team work, fixing correctness before styling, comparing suitable models, and acknowledging deployment/operational gaps |

Avoid claiming state of the art, business savings, production deployment, fairness guarantees or a performance improvement over the notebook. The evaluation protocols differ. Credit Sameen Fatima and Laiba Nadeem for the original university project, as recorded in the preserved report.

## Recommended screenshot sequence — seven maximum

1. **Overview:** project title, intended feedback-triage use, 24,000 training reviews and holdout macro F1. Keep the rating-proxy caveat visible.
2. **Single review:** an understandable example, predicted class, three class probabilities and the uncalibrated-probability caption. Do not cherry-pick this as proof of accuracy.
3. **Batch prediction:** the illustrative three-row CSV, predictions, sentiment distribution and download control. Use sample reviews without personal information.
4. **Data exploration:** original/clean/sample counts, duplicate exclusions and class imbalance. Explain that the positive majority motivated macro F1.
5. **Model comparison:** CV macro F1 alongside accuracy for all four candidates. Show why logistic regression was selected despite SVM's higher accuracy.
6. **Holdout evaluation:** confusion matrix plus per-class precision, recall and F1. Include neutral F1 = 0.36 rather than cropping it away.
7. **Explainability:** class-associated terms or local TF-IDF × coefficient contributions. Keep the association/not-causation caption visible.

The captured local overview is in `docs/screenshots/overview.png`. The other six are a capture plan, not claims that screenshots already exist. The current app supports these views. Use a normal desktop browser window when capturing the final LinkedIn carousel.

## Draft LinkedIn post

**My sentiment model with the highest accuracy wasn't the one I selected.**

I've been revisiting our university product-review sentiment project and turning it into a more reproducible ML portfolio application.

The problem is straightforward: when reviews pile up, it becomes harder to spot unhappy customers and mixed experiences. I built a Streamlit app to explore positive, negative and neutral feedback, one review or a CSV at a time.

Using a stratified sample of 30,000 reviews after data-quality checks, I compared a dummy baseline, Naive Bayes, logistic regression and Linear SVM. Logistic regression had the best five-fold cross-validation macro F1, even though Linear SVM had higher accuracy.

On 6,000 held-out reviews, the selected model achieved:

- Accuracy: **83.72%**
- Macro F1: **0.6535**
- Weighted F1: **0.8445**

Neutral sentiment is still the hardest class, with F1 of **0.36**. Three-star ratings are imperfect sentiment labels, so that's an important limitation rather than something to hide.

The engineering work included preserving negation, removing duplicate reviews before splitting, saving preprocessing and classification together, adding batch downloads and explanations, and writing automated tests. The stack is Python, scikit-learn, TF-IDF, pandas and Streamlit.

My biggest takeaway: an ML project becomes much more useful when the evaluation, prediction pipeline and documentation tell the same story.

The app runs locally, and the repository includes Streamlit Community Cloud deployment instructions. A public deployment is still pending.

Thanks to Sameen Fatima and Laiba Nadeem, my teammates on the original university project.

GitHub: [add repository link after pushing the reviewed changes]

Live app: [add verified URL after deployment]

#DataScience #MachineLearning #NLP #Python #Streamlit #MLPortfolio

## Before publishing

Replace the placeholders only after the updated code is pushed and a live deployment is verified. If mentioning test counts, use the latest actual run. Keep attribution and personal wording accurate to your own contribution and experience. The draft is material for review; nothing has been posted.
