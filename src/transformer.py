"""Optional CPU DistilBERT experiment; never imported by normal training or app."""

import importlib.metadata

import numpy as np
from sklearn.model_selection import train_test_split

from src.data import LABELS
from src.evaluate import evaluate_predictions


def train_transformer(fit, val, test, config, output):
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise ImportError("Install requirements-transformer.txt to use --transformer.") from error
    cfg = config["transformer"]
    set_seed(config["seed"])
    if cfg["max_samples"] < len(fit):
        fit, _ = train_test_split(
            fit, train_size=cfg["max_samples"], stratify=fit.label, random_state=config["seed"]
        )
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])

    class Reviews(torch.utils.data.Dataset):
        def __init__(self, frame):
            self.tokens = tokenizer(
                frame.review_text.tolist(), truncation=True, max_length=cfg["max_length"]
            )
            self.labels = [LABELS.index(label) for label in frame.label]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return {
                **{key: value[index] for key, value in self.tokens.items()},
                "labels": self.labels[index],
            }

    counts = fit.label.value_counts()
    weights = torch.tensor(
        [len(fit) / (len(LABELS) * counts[label]) for label in LABELS], dtype=torch.float
    )

    class BalancedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                outputs.logits, labels, weight=weights.to(outputs.logits.device)
            )
            return (loss, outputs) if return_outputs else loss

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"],
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    )
    args = TrainingArguments(
        output_dir=str(output),
        use_cpu=True,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=config["seed"],
        data_seed=config["seed"],
        report_to="none",
        dataloader_pin_memory=False,
        save_total_limit=1,
    )

    def compute_metrics(prediction):
        result = evaluate_predictions(
            np.array(LABELS)[prediction.label_ids],
            np.array(LABELS)[prediction.predictions.argmax(axis=1)],
        )
        return {key: result[key] for key in ("accuracy", "macro_f1")}

    trainer = BalancedTrainer(
        model=model,
        args=args,
        train_dataset=Reviews(fit),
        eval_dataset=Reviews(val),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    # One optional-model test prediction pass, after validation checkpoint selection.
    predictions = np.array(LABELS)[trainer.predict(Reviews(test)).predictions.argmax(axis=1)]
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    return {
        "model": cfg["model"],
        "train_rows": len(fit),
        "test": evaluate_predictions(test.label, predictions),
        "per_source": {
            source: evaluate_predictions(test.label[mask], predictions[mask])
            for source in sorted(test.source.unique())
            for mask in [test.source.to_numpy() == source]
        },
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("torch", "transformers", "accelerate")
        },
        "note": "Optional subset experiment; no CV score; not the deployed calibrated sklearn pipeline.",
    }
