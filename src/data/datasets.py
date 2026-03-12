"""Load and preprocess the 5 continual-learning benchmark datasets."""

from functools import partial

from datasets import load_dataset
from sklearn.model_selection import train_test_split

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

    prompt = f"Classify: {text}\nLabel:"
    target = f" {label_name}"
    full_text = prompt + target

    # Tokenize prompt (for masking) and full text
    prompt_tokens = tokenizer(
        prompt, truncation=True, max_length=max_length, add_special_tokens=True
    )
    full_tokens = tokenizer(
        full_text, truncation=True, max_length=max_length, add_special_tokens=True,
        padding="max_length",
    )

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    # Labels: mask prompt tokens with -100, keep target tokens
    labels = list(input_ids)
    prompt_len = len(prompt_tokens["input_ids"])
    for i in range(prompt_len):
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


def load_task_dataset(task_name, tokenizer, max_length=512, max_samples=20000):
    """Load a single task dataset, return (train_dataset, dev_dataset, eval_dataset)."""
    cfg = DATASET_REGISTRY[task_name]

    ds = load_dataset(cfg["path"])

    train_ds = ds["train"]
    eval_ds = ds["test"]

    # Subsample if needed
    if max_samples and len(train_ds) > max_samples:
        train_ds = train_ds.shuffle(seed=42).select(range(max_samples))
    eval_max = min(2000, len(eval_ds))
    eval_ds = eval_ds.shuffle(seed=42).select(range(eval_max))

    # 90/10 train/dev split
    indices = list(range(len(train_ds)))
    train_indices, dev_indices = train_test_split(indices, test_size=0.1, random_state=42)
    dev_ds = train_ds.select(dev_indices)
    train_ds = train_ds.select(train_indices)

    map_fn = partial(
        _format_example,
        text_fields=cfg["text_fields"],
        label_field=cfg["label_field"],
        label_names=cfg["label_names"],
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_ds = train_ds.map(map_fn, remove_columns=train_ds.column_names)
    dev_ds = dev_ds.map(map_fn, remove_columns=dev_ds.column_names)
    eval_ds = eval_ds.map(map_fn, remove_columns=eval_ds.column_names)

    train_ds.set_format("torch")
    dev_ds.set_format("torch")
    eval_ds.set_format("torch")

    return train_ds, dev_ds, eval_ds


class LazyDatasetLoader:
    """Loads datasets on-demand when first accessed, not all upfront."""

    def __init__(self, task_order, tokenizer, max_length=512, max_samples=20000):
        self.task_order = task_order
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_samples = max_samples
        self._cache = {}

    def __getitem__(self, task_name):
        if task_name not in self._cache:
            print(f"Loading dataset: {task_name}")
            self._cache[task_name] = load_task_dataset(
                task_name, self.tokenizer, self.max_length, self.max_samples
            )
        return self._cache[task_name]

    def get_split(self, task_name, split="eval"):
        """Get a specific split: 'train', 'dev', or 'eval'."""
        train_ds, dev_ds, eval_ds = self[task_name]
        if split == "train":
            return train_ds
        elif split == "dev":
            return dev_ds
        return eval_ds


def load_all_datasets(task_order, tokenizer, max_length=512, max_samples=20000):
    """Return a lazy loader that loads each dataset on first access."""
    return LazyDatasetLoader(task_order, tokenizer, max_length, max_samples)
