"""Load and preprocess the continual-learning benchmark datasets.

Prompt schema follows O-LoRA (Wang et al., 2023) Section 3.1:
  {Task Definition}
  Option: {opt1}, {opt2}, ...
  {Text}
  Answer: {label}
"""

from functools import partial

from datasets import load_dataset

DATASET_REGISTRY = {
    "ag_news": {
        "path": "ag_news",
        "text_fields": ["text"],
        "label_field": "label",
        "label_names": ["World", "Sports", "Business", "Technology"],
        "instruction": "What is the topic of the following paragraph? Choose one from the option.",
    },
    "yelp_review_full": {
        "path": "yelp_review_full",
        "text_fields": ["text"],
        "label_field": "label",
        "label_names": ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"],
        "instruction": "What is the sentiment of the following paragraph? Choose one from the option.",
    },
    "amazon_polarity": {
        "path": "amazon_polarity",
        "text_fields": ["title", "content"],
        "label_field": "label",
        "label_names": ["Negative", "Positive"],
        "instruction": "What is the sentiment of the following paragraph? Choose one from the option.",
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
        "instruction": "What is the topic of the following paragraph? Choose one from the option.",
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
        "instruction": "What is the topic of the following paragraph? Choose one from the option.",
    },
}


def _format_example(example, text_fields, label_field, label_names, instruction,
                    tokenizer, max_length):
    """Format a single example using the O-LoRA instruction tuning schema:

        {Task Definition}
        Option: {opt1}, {opt2}, ...
        {Text}
        Answer: {label}
    """
    parts = [str(example[f]) for f in text_fields if example.get(f)]
    text = " ".join(parts)

    label_idx  = example[label_field]
    label_name = label_names[label_idx]

    options_str = ", ".join(label_names)
    target      = f" {label_name}"

    # Reserve space for the target token(s) so they survive truncation
    target_token_len = len(tokenizer(target, add_special_tokens=False)["input_ids"])

    # The prefix (instruction + options + "Answer:") is fixed-length — compute it
    prefix = f"{instruction}\nOption: {options_str}\n"
    suffix = "\nAnswer:"
    prefix_len = len(tokenizer(prefix + suffix, add_special_tokens=False)["input_ids"]) + 1  # +1 for BOS
    text_budget = max(8, max_length - prefix_len - target_token_len - 1)

    # Truncate only the article text, keep prefix/suffix intact
    text_tokens = tokenizer(text, truncation=True, max_length=text_budget,
                            add_special_tokens=False)
    text_truncated = tokenizer.decode(text_tokens["input_ids"], skip_special_tokens=True)

    prompt     = f"{prefix}{text_truncated}{suffix}"
    full_text  = prompt + target

    full_tokens = tokenizer(
        full_text, truncation=True, max_length=max_length, add_special_tokens=True,
        padding="max_length",
    )

    input_ids      = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]

    # Labels: mask prompt with -100, keep target token(s)
    labels = list(input_ids)
    prompt_only = tokenizer(prompt, truncation=True, max_length=max_length,
                            add_special_tokens=True)
    prompt_len = len(prompt_only["input_ids"])
    for i in range(min(prompt_len, len(labels))):
        labels[i] = -100
    for i in range(len(labels)):
        if attention_mask[i] == 0:
            labels[i] = -100

    return {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
        "label_idx":      label_idx,
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
    """Load a single task dataset, return (train_dataset, eval_dataset)."""
    cfg = DATASET_REGISTRY[task_name]

    ds = load_dataset(cfg["path"])

    train_ds = ds["train"]
    eval_ds  = ds["test"]

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
        instruction=cfg["instruction"],
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_ds = train_ds.map(map_fn, remove_columns=train_ds.column_names)
    eval_ds  = eval_ds.map(map_fn,  remove_columns=eval_ds.column_names)

    train_ds.set_format("torch")
    eval_ds.set_format("torch")

    return train_ds, eval_ds


class LazyDatasetLoader:
    """Loads datasets on-demand when first accessed, not all upfront."""

    def __init__(self, task_order, tokenizer, max_length=512, max_samples=20000,
                 max_eval_samples=2000, dataset_fraction=1.0, eval_fraction=1.0):
        self.task_order       = task_order
        self.tokenizer        = tokenizer
        self.max_length       = max_length
        self.max_samples      = max_samples
        self.max_eval_samples = max_eval_samples
        self.dataset_fraction = dataset_fraction
        self.eval_fraction    = eval_fraction
        self._cache           = {}

    def __getitem__(self, task_name):
        if task_name not in self._cache:
            print(f"Loading dataset: {task_name}")
            max_samples = (
                self.max_samples.get(task_name, self.max_samples.get("default"))
                if isinstance(self.max_samples, dict)
                else self.max_samples
            )
            self._cache[task_name] = load_task_dataset(
                task_name, self.tokenizer, self.max_length,
                max_samples, self.max_eval_samples,
                self.dataset_fraction, self.eval_fraction,
            )
        return self._cache[task_name]


def load_all_datasets(task_order, tokenizer, max_length=512, max_samples=20000,
                      max_eval_samples=2000, dataset_fraction=1.0, eval_fraction=1.0):
    """Return a lazy loader that loads each dataset on first access."""
    return LazyDatasetLoader(
        task_order, tokenizer, max_length, max_samples,
        max_eval_samples, dataset_fraction, eval_fraction,
    )
