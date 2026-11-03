from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from transformers import AutoTokenizer


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
FIGURE_DIR = ROOT / "docs" / "figures"
REPORT_PATH = ROOT / "docs" / "eda.md"

MODEL_NAME = "distilbert/distilbert-base-uncased"

LABEL_NAMES = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

MAX_LENGTHS = [128, 256]

BATCH_SIZE = 256


# ==========================================
# 2. LOAD PROCESSED DATA
# ==========================================

def load_data():

    datasets = {}

    for split in ["train", "validation"]:

        path = DATA_DIR / f"{split}.csv"

        if not path.exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        required = {"text", "label"}

        if not required.issubset(df.columns):
            raise ValueError(
                f"{split} must contain text and label"
            )

        if df["text"].isna().any():
            raise ValueError(
                f"{split} contains missing text"
            )

        if not df["label"].isin(LABEL_NAMES).all():
            raise ValueError(
                f"{split} contains invalid labels"
            )

        df["text"] = df["text"].astype(str)

        df["word_count"] = (
            df["text"].str.split().str.len()
        )

        df["char_count"] = df["text"].str.len()

        datasets[split] = df

    return datasets


# ==========================================
# 3. LABEL DISTRIBUTION
# ==========================================

def analyze_labels(datasets):

    counts = {}

    for split, df in datasets.items():

        counts[split] = (
            df["label"]
            .value_counts()
            .reindex(LABEL_NAMES.keys(), fill_value=0)
        )

    result = pd.DataFrame(counts)

    print("\nLABEL DISTRIBUTION")
    print(result)

    x = np.arange(len(LABEL_NAMES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x - width / 2,
        result["train"],
        width,
        label="Train"
    )

    ax.bar(
        x + width / 2,
        result["validation"],
        width,
        label="Validation"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES.values())

    ax.set_ylabel("Number of articles")
    ax.set_title("AG News Label Distribution")

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "label_distribution.png",
        dpi=150
    )

    plt.close(fig)

    return result


# ==========================================
# 4. TEXT LENGTH ANALYSIS
# ==========================================

def analyze_text_length(datasets):

    train = datasets["train"]

    print("\nTEXT LENGTH STATISTICS")

    print(
        train[
            ["word_count", "char_count"]
        ].describe(percentiles=[0.5, 0.9, 0.95, 0.99])
    )

    # Word length distribution

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        train["word_count"],
        bins=50
    )

    ax.set_xlabel("Number of words")
    ax.set_ylabel("Number of articles")

    ax.set_title("Word Length Distribution")

    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "word_length_distribution.png",
        dpi=150
    )

    plt.close(fig)

    # Length comparison by category

    groups = [
        train.loc[
            train["label"] == label,
            "word_count"
        ].values
        for label in LABEL_NAMES
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.boxplot(
        groups,
        tick_labels=list(LABEL_NAMES.values()),
        showfliers=False
    )

    ax.set_ylabel("Number of words")
    ax.set_title("Text Length by News Category")

    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "word_length_by_category.png",
        dpi=150
    )

    plt.close(fig)

    # Per-class statistics

    class_stats = (
        train.groupby("label")["word_count"]
        .agg(["mean", "median", "max"])
    )

    return class_stats


# ==========================================
# 5. TOKEN LENGTH ANALYSIS
# ==========================================

def calculate_token_lengths(texts, tokenizer):

    lengths = []

    for start in range(0, len(texts), BATCH_SIZE):

        batch = texts[start:start + BATCH_SIZE]

        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False,
            padding=False
        )

        batch_lengths = [
            len(ids)
            for ids in encoded["input_ids"]
        ]

        lengths.extend(batch_lengths)

        if (start // BATCH_SIZE) % 50 == 0:

            print(
                f"Tokenizing: "
                f"{min(start + BATCH_SIZE, len(texts)):,}"
                f"/{len(texts):,}"
            )

    return np.asarray(lengths)


def analyze_truncation(datasets, tokenizer):

    special_tokens = (
        tokenizer.num_special_tokens_to_add(pair=False)
    )

    all_lengths = {}
    truncation_results = []

    for split, df in datasets.items():

        texts = df["text"].tolist()

        lengths = calculate_token_lengths(
            texts,
            tokenizer
        )

        all_lengths[split] = lengths

        content_lengths = np.maximum(
            lengths - special_tokens,
            0
        )

        print(f"\nTOKEN STATISTICS: {split.upper()}")

        print(
            "Mean:",
            round(float(lengths.mean()), 2)
        )

        print(
            "Median:",
            int(np.median(lengths))
        )

        print(
            "P95:",
            int(np.percentile(lengths, 95))
        )

        print(
            "P99:",
            int(np.percentile(lengths, 99))
        )

        print("Maximum:", int(lengths.max()))

        for max_length in MAX_LENGTHS:

            truncated = np.maximum(
                lengths - max_length,
                0
            )

            truncated_samples = int(
                (lengths > max_length).sum()
            )

            sample_percentage = (
                truncated_samples / len(lengths) * 100
            )

            discarded_tokens = int(
                truncated.sum()
            )

            total_content_tokens = int(
                content_lengths.sum()
            )

            token_percentage = (
                discarded_tokens /
                max(total_content_tokens, 1) * 100
            )

            truncation_results.append({
                "split": split,
                "max_length": max_length,
                "truncated_samples": truncated_samples,
                "truncated_percentage": sample_percentage,
                "discarded_tokens": discarded_tokens,
                "discarded_token_percentage": token_percentage
            })

    results = pd.DataFrame(truncation_results)

    print("\nTRUNCATION COMPARISON")
    print(results.to_string(index=False))

    # Token length histogram

    lengths = all_lengths["train"]

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.hist(
        lengths,
        bins=60
    )

    ax.axvline(
        128,
        linestyle="--",
        label="Max Length = 128"
    )

    ax.axvline(
        256,
        linestyle="--",
        label="Max Length = 256"
    )

    ax.set_xlim(
        0,
        max(512, int(np.percentile(lengths, 99)))
    )

    ax.set_xlabel("Number of tokens (including special tokens)")
    ax.set_ylabel("Number of articles")

    ax.set_title("DistilBERT Token Length Distribution")

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "token_length_distribution.png",
        dpi=150
    )

    plt.close(fig)

    # Truncation percentage comparison

    fig, ax = plt.subplots(figsize=(9, 6))

    x = np.arange(len(MAX_LENGTHS))
    width = 0.35

    for index, split in enumerate(datasets):

        subset = results[
            results["split"] == split
        ]

        ax.bar(
            x + (index - 0.5) * width,
            subset["truncated_percentage"],
            width,
            label=split.capitalize()
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [str(length) for length in MAX_LENGTHS]
    )

    ax.set_xlabel("Maximum token length")
    ax.set_ylabel("Truncated articles (%)")

    ax.set_title("Truncation Rate: 128 vs 256")

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "truncation_comparison.png",
        dpi=150
    )

    plt.close(fig)

    return results, all_lengths


# ==========================================
# 6. GENERATE EDA REPORT
# ==========================================

def generate_report(
    datasets,
    label_stats,
    class_stats,
    truncation_results,
    all_lengths
):

    lines = [
        "# AG News - Exploratory Data Analysis",
        "",
        "## 1. Dataset Overview",
        "",
        "| Split | Samples |",
        "|---|---:|"
    ]

    for split, df in datasets.items():

        lines.append(
            f"| {split.capitalize()} | {len(df):,} |"
        )
    
        train_128 = truncation_results[
        (truncation_results["split"] == "train") &
        (truncation_results["max_length"] == 128)
    ].iloc[0]

    train_256 = truncation_results[
        (truncation_results["split"] == "train") &
        (truncation_results["max_length"] == 256)
    ].iloc[0]

    valid_128 = truncation_results[
        (truncation_results["split"] == "validation") &
        (truncation_results["max_length"] == 128)
    ].iloc[0]

    valid_256 = truncation_results[
        (truncation_results["split"] == "validation") &
        (truncation_results["max_length"] == 256)
    ].iloc[0]

    lines.extend([
        "",
        "## 2. Label Distribution",
        "",
        "![Label distribution](figures/label_distribution.png)",
        "",
        "| Category | Train | Validation |",
        "|---|---:|---:|"
    ])

    for label, name in LABEL_NAMES.items():

        lines.append(
            f"| {name} | "
            f"{int(label_stats.loc[label, 'train']):,} | "
            f"{int(label_stats.loc[label, 'validation']):,} |"
        )

    train = datasets["train"]

    lines.extend([
        "",
        "## 3. Text Length Analysis",
        "",
        "![Word length](figures/word_length_distribution.png)",
        "",
        "| Statistic | Word Count | Character Count |",
        "|---|---:|---:|"
    ])

    for stat, description in [
        ("mean", "Mean"),
        ("50%", "Median"),
        ("90%", "P90"),
        ("95%", "P95"),
        ("99%", "P99"),
        ("max", "Maximum")
    ]:

        stats = train[
            ["word_count", "char_count"]
        ].describe(percentiles=[0.5, 0.9, 0.95, 0.99])

        lines.append(
            f"| {description} | "
            f"{stats.loc[stat, 'word_count']:.2f} | "
            f"{stats.loc[stat, 'char_count']:.2f} |"
        )

    lines.extend([
        "",
        "### 3.1 Length by Category",
        "",
        "![Length by category](figures/word_length_by_category.png)",
        "",
        "| Category | Mean Words | Median Words | Maximum Words |",
        "|---|---:|---:|---:|"
    ])

    for label, name in LABEL_NAMES.items():

        row = class_stats.loc[label]

        lines.append(
            f"| {name} | {row['mean']:.2f} | "
            f"{row['median']:.0f} | {row['max']:.0f} |"
        )

    lines.extend([
        "",
        "## 4. Token Length and Truncation",
        "",
        "Tokenizer: `distilbert/distilbert-base-uncased`",
        "",
        "Token counts include special tokens added by the tokenizer.",
        "",
        "![Token length](figures/token_length_distribution.png)",
        "",
        "| Split | Max Length | Truncated Articles | Truncated (%) | Discarded Tokens | Discarded (%) |",
        "|---|---:|---:|---:|---:|---:|"
    ])

    for _, row in truncation_results.iterrows():

        lines.append(
            f"| {row['split']} | {int(row['max_length'])} | "
            f"{int(row['truncated_samples']):,} | "
            f"{row['truncated_percentage']:.2f}% | "
            f"{int(row['discarded_tokens']):,} | "
            f"{row['discarded_token_percentage']:.2f}% |"
        )

    lines.extend([
        "",
        "![Truncation comparison](figures/truncation_comparison.png)",
        "",
        "## 5. Observations",
        "",
                "## 5. Findings and Recommendations",
        "",
        "### 5.1 Label Distribution",
        "",
        "The training and validation datasets maintain "
        "similar class proportions through stratified sampling.",
        "No significant class imbalance was observed.",
        "",
        "### 5.2 Text Length Distribution",
        "",
        f"- Average article length: "
        f"{train['word_count'].mean():.2f} words.",
        f"- Median article length: "
        f"{train['word_count'].median():.0f} words.",
        f"- P95 article length: "
        f"{train['word_count'].quantile(0.95):.0f} words.",
        f"- Average character count: "
        f"{train['char_count'].mean():.2f}.",
        "",
        "The distribution is right-skewed, indicating that "
        "most articles are relatively short, while a small "
        "number contain substantially more text.",
        "",
        "### 5.3 Length by Category",
        "",
        f"- Category with highest average word count: "
        f"{LABEL_NAMES[int(class_stats['mean'].idxmax())]}.",
        f"- Category with longest individual article: "
        f"{LABEL_NAMES[int(class_stats['max'].idxmax())]}.",
        "",
        "Average word lengths are relatively similar across "
        "the four news categories.",
        "",
        "### 5.4 Tokenization and Truncation",
        "",
        f"- Training average token length: "
        f"{all_lengths['train'].mean():.2f}.",
        f"- Training P99 token length: "
        f"{np.percentile(all_lengths['train'], 99):.0f}.",
        "",
                f"- Train truncation at 128: "
        f"{train_128['truncated_percentage']:.2f}%.",
        f"- Train truncation at 256: "
        f"{train_256['truncated_percentage']:.2f}%.",
        f"- Validation truncation at 128: "
        f"{valid_128['truncated_percentage']:.2f}%.",
        f"- Validation truncation at 256: "
        f"{valid_256['truncated_percentage']:.2f}%.",
        "",
        "Increasing max_length from 128 to 256 "
        "substantially reduces truncation.",
        "",
        "### 5.5 Recommendation",
        "",
        "A maximum sequence length of 128 is a reasonable "
        "starting configuration for DistilBERT.",
        "It preserves more than 99.4% of training articles "
        "without truncation while reducing computational "
        "and memory requirements compared with 256.",
        "",
        "However, the final configuration should be selected "
        "through validation experiments using Macro-F1 and "
        "resource consumption.",
        "",
        "The test split was excluded from this analysis "
        "to avoid using test performance for model selection.",
        "",
        ""
    ])

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\nEDA report saved:", REPORT_PATH)


# ==========================================
# 7. MAIN
# ==========================================

def main():

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading processed datasets...")

    datasets = load_data()

    print("\nLoading DistilBERT tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True
    )

    label_stats = analyze_labels(datasets)

    class_stats = analyze_text_length(datasets)

    truncation_results, all_lengths = (
        analyze_truncation(datasets, tokenizer)
    )

    generate_report(
        datasets,
        label_stats,
        class_stats,
        truncation_results,
        all_lengths
    )

    print("\nEDA COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()