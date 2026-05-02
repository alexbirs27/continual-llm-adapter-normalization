"""Count train/test samples in the O-LoRA CL_Benchmark data files."""

import json
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).parent.parent / "olora_orig/O-LoRA/CL_Benchmark"

TASKS = {
    "ag_news":          BENCHMARK_ROOT / "TC/agnews",
    "amazon_polarity":  BENCHMARK_ROOT / "SC/amazon",
    "dbpedia_14":       BENCHMARK_ROOT / "TC/dbpedia",
    "yahoo_answers":    BENCHMARK_ROOT / "TC/yahoo",
}


def count(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return len(json.load(f))


if __name__ == "__main__":
    print(f"{'Task':<20} {'Train':>8} {'Test':>8}")
    print("-" * 38)
    for task, folder in TASKS.items():
        train = count(folder / "train.json")
        test  = count(folder / "test.json")
        print(f"{task:<20} {train:>8} {test:>8}")
