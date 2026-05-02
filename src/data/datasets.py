"""Load and preprocess the 5 continual-learning benchmark datasets."""

from functools import partial

from datasets import load_dataset

# Dataset configs: (hf_name, text_fields, label_field, label_names or None)
DATASET_REGISTRY = {
    "ag_news": {
        "path": "ag_news",
        "text_fields": ["text"],
        "label_field": "label",
        "label_names": ["World", "Sports", "Business", "Technology"],
    },
    "yelp_review_full": {
        "path": "yelp_review_full",
        "text_fields": ["text"],
        "label_field": "label",
        "label_names": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
    },
    "amazon_polarity": {
        "path": "amazon_polarity",
        "text_fields": ["title", "content"],
        "label_field": "label",
        "label_names": ["Negative", "Positive"],
    },
    "dbpedia_14": {
        "path": "dbpedia_14",
        "text_fields": ["title", "content"],
        "label_field": "label",
        "label_names": [
            "Company", "Educational Institution", "Artist", "Athlete",
            "Office Holder", "Mean of Transportation", "Building", "Natural Place",
            "Village", "Animal", "Plant", "Album", "Film", "Written Work",
        ],
    },
    "yahoo_answers_topics": {
        "path": "yahoo_answers_topics",
        "text_fields": ["question_title", "question_content", "best_answer"],
        "label_field": "topic",
        "label_names": [
            "Society & Culture", "Science & Mathematics", "Health",
            "Education & Reference", "Computers & Internet", "Sports",
            "Business & Finance", "Entertainment & Music",
            "Family & Relationships", "Politics & Government",
        ],
    },
}


def _format_example(example, text_fields, label_field, label_names, tokenizer, max_length):
    """Format a single example as 'Classify: {text}\nLabel: {label_name}'."""
    # Build text from fields
    parts = [str(example[f]) for f in text_fields if example.get(f)]
    text = " ".join(parts)

    label_idx = example[label_field]
    label_name = label_names[label_idx]

    target = f" {label_name}"
    # Reserve space for the target tokens so they survive truncation
    target_token_len = len(tokenizer(target, add_special_tokens=False)["input_ids"])
    prompt_budget = max(8, max_length - target_token_len - 2)

    prompt = f"Classify: {text}\nLabel:"
    prompt_tokens = tokenizer(
        prompt, truncation=True, max_length=prompt_budget, add_special_tokens=True
    )
    # Rebuild prompt from truncated tokens to keep prompt+target within max_length
    prompt_truncated = tokenizer.decode(prompt_tokens["input_ids"], skip_special_tokens=True)
    full_text = prompt_truncated + target

    full_tokens = tokenizer(
        full_text, truncation=True, max_length=max_length, add_special_tokens=True,
        padding="max_length",
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    # Labels: mask prompt tokens with -100, keep target tokens
    labels = list(input_ids)
    # Re-tokenize the truncated prompt under the same conditions as full_tokens
    # so prompt_len aligns with full_tokens' leading positions.
    prompt_only_tokens = tokenizer(
        prompt_truncated, truncation=True, max_length=max_length, add_special_tokens=True
    )
    prompt_len = len(prompt_only_tokens["input_ids"])
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100
    # Also mask padding
    for i in range(len(labels)):
        if attention_mask[i] == 0:
            labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "label_idx": label_idx,
    }


def load_task_dataset(
    task_name,
    tokenizer,
    max_length=512,
    max_samples=20000,
    max_eval_samples=2000,
    dataset_fraction=1.0,
    eval_fraction=1.0,
):
    """Load a single task dataset, return (train_dataset, eval_dataset).

    Sample-size control: an effective limit is computed as
    min(fraction * split_size, hard_cap). Fraction is the primary knob,
    hard caps are optional ceilings for memory/compute safety.
    """
    cfg = DATASET_REGISTRY[task_name]

    ds = load_dataset(cfg["path"])

    train_ds = ds["train"]
    eval_ds = ds["test"]

    train_target = int(len(train_ds) * dataset_fraction)
    if max_samples:
        train_target = min(train_target, max_samples)
    train_target = max(1, min(train_target, len(train_ds)))
    train_ds = train_ds.shuffle(seed=42).select(range(train_target))

    eval_target = int(len(eval_ds) * eval_fraction)
    if max_eval_samples:
        eval_target = min(eval_target, max_eval_samples)
    eval_target = max(1, min(eval_target, len(eval_ds)))
    eval_ds = eval_ds.shuffle(seed=42).select(range(eval_target))

    map_fn = partial(
        _format_example,
        text_fields=cfg["text_fields"],
        label_field=cfg["label_field"],
        label_names=cfg["label_names"],
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_ds = train_ds.map(map_fn, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(map_fn, remove_columns=eval_ds.column_names)

    train_ds.set_format("torch")
    eval_ds.set_format("torch")

    return train_ds, eval_ds


class LazyDatasetLoader:
    """Loads datasets on-demand when first accessed, not all upfront."""

    def __init__(
        self,
        task_order,
        tokenizer,
        max_length=512,
        max_samples=20000,
        max_eval_samples=2000,
        dataset_fraction=1.0,
        eval_fraction=1.0,
    ):
        self.task_order = task_order
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples
        self.max_eval_samples = max_eval_samples
        self.dataset_fraction = dataset_fraction
        self.eval_fraction = eval_fraction
        self._cache = {}

    def __getitem__(self, task_name):
        if task_name not in self._cache:
            print(f"Loading dataset: {task_name}")
            max_samples = (
                self.max_samples.get(task_name, self.max_samples.get("default"))
                if isinstance(self.max_samples, dict)
                else self.max_samples
            )
            self._cache[task_name] = load_task_dataset(
                task_name,
                self.tokenizer,
                self.max_length,
                max_samples,
                self.max_eval_samples,
                self.dataset_fraction,
                self.eval_fraction,
            )
        return self._cache[task_name]


def load_all_datasets(
    task_order,
    tokenizer,
    max_length=512,
    max_samples=20000,
    max_eval_samples=2000,
    dataset_fraction=1.0,
    eval_fraction=1.0,
):
    """Return a lazy loader that loads each dataset on first access."""
    return LazyDatasetLoader(
        task_order,
        tokenizer,
        max_length,
        max_samples,
        max_eval_samples,
        dataset_fraction,
        eval_fraction,
    )
