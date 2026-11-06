import html
import re
from pathlib import Path

import pandas as pd


# Project configuration
ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
OUTPUT_FILE = ROOT / "docs" / "data_audit.md"

REQUIRED_COLUMNS = [
    "Class Index",
    "Title",
    "Description"
]

LABEL_NAMES = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}


def normalize_text(df):
    """
    Combine Title and Description,
    decode HTML entities and normalize whitespace.
    """

    title = df["Title"].fillna("").astype(str)
    description = df["Description"].fillna("").astype(str)

    text = title + " " + description

    return text.map(
        lambda value: re.sub(
            r"\s+", " ", html.unescape(value)
        ).strip()
    )


def audit_dataset(name, df):
    """
    Analyze one raw dataset without modifying it.
    """

    total_rows = len(df)

    text = normalize_text(df)

    # 1. Dataset information
    print(f"\n{'=' * 50}")
    print(f"DATASET: {name.upper()}")
    print(f"{'=' * 50}")

    print("Shape:", df.shape)
    print("\nData types:")
    print(df.dtypes)

    # 2. Missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # 3. Label distribution
    print("\nLabel distribution:")

    print(
        df["Class Index"]
        .value_counts(dropna=False)
        .sort_index()
    )

    # 4. Invalid labels
    invalid_labels = df[
        ~df["Class Index"].isin([1, 2, 3, 4])
    ]

    print("\nInvalid labels:", len(invalid_labels))

    # 5. Exact duplicate rows
    duplicate_rows = df.duplicated().sum()

    print("Duplicate rows:", duplicate_rows)

    # 6. Duplicate articles
    nonempty_text = text[text.ne("")]

    duplicate_text_rows = nonempty_text.duplicated().sum()

    duplicate_text_groups = (
        nonempty_text[
            nonempty_text.duplicated(keep=False)
        ].nunique()
    )

    print("Duplicate text rows:", duplicate_text_rows)
    print("Duplicate text groups:", duplicate_text_groups)

    # 7. Empty articles
    empty_articles = text.eq("").sum()

    print("Empty articles:", empty_articles)

    # 8. Same text but different labels
    temp = pd.DataFrame({
        "text": text,
        "label": df["Class Index"]
    })

    conflicts = (
        temp[temp["text"].ne("")]
        .groupby("text")["label"]
        .nunique()
    )

    conflicting_groups = (conflicts > 1).sum()

    print("Conflicting label groups:", conflicting_groups)

    return {
        "name": name,
        "rows": total_rows,
        "columns": len(df.columns),
        "missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(duplicate_rows),
        "duplicate_text_rows": int(duplicate_text_rows),
        "duplicate_text_groups": int(duplicate_text_groups),
        "empty_articles": int(empty_articles),
        "invalid_labels": len(invalid_labels),
        "conflicting_groups": int(conflicting_groups),
        "text": text,
        "labels": df["Class Index"]
    }


def load_dataset(name):
    path = RAW_DIR / f"{name}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"{name}: Missing columns {missing_columns}"
        )

    return df


def main():

    train = load_dataset("train")
    test = load_dataset("test")

    train_report = audit_dataset("train", train)
    test_report = audit_dataset("test", test)

    # Cross-dataset duplicate analysis
    train_texts = set(
        train_report["text"][train_report["text"].ne("")]
    )

    test_texts = set(
        test_report["text"][test_report["text"].ne("")]
    )

    overlap = train_texts & test_texts

    print("\n" + "=" * 50)
    print("CROSS DATASET ANALYSIS")
    print("=" * 50)

    print("Train unique texts:", len(train_texts))
    print("Test unique texts:", len(test_texts))
    print("Train-Test overlapping texts:", len(overlap))

    # Generate report
    reports = [train_report, test_report]

    lines = [
        "# AG News - Data Audit Report",
        "",
        "## 1. Dataset Overview",
        "",
        "| Dataset | Rows | Columns |",
        "|---|---:|---:|"
    ]

    for item in reports:
        lines.append(
            f"| {item['name']} | {item['rows']:,} "
            f"| {item['columns']} |"
        )

    for item, df in zip(reports, [train, test]):

        lines.extend([
            "",
            f"## 2. {item['name'].upper()} Dataset",
            "",
            "### 2.1 Missing Values",
            "",
            "| Column | Missing |",
            "|---|---:|"
        ])

        for column, count in df.isnull().sum().items():
            lines.append(f"| {column} | {count:,} |")

        lines.extend([
            "",
            "### 2.2 Label Distribution",
            "",
            "| Label | Category | Count | Percentage |",
            "|---:|---|---:|---:|"
        ])

        for label, category in LABEL_NAMES.items():
            count = int((df["Class Index"] == label).sum())
            percentage = count / max(len(df), 1) * 100

            lines.append(
                f"| {label} | {category} "
                f"| {count:,} | {percentage:.2f}% |"
            )

        lines.extend([
            "",
            "### 2.3 Duplicate and Invalid Data",
            "",
            f"- Exact duplicate rows: {item['duplicate_rows']:,}",
            f"- Duplicate text rows: {item['duplicate_text_rows']:,}",
            f"- Duplicate text groups: {item['duplicate_text_groups']:,}",
            f"- Empty articles: {item['empty_articles']:,}",
            f"- Invalid labels: {item['invalid_labels']:,}",
            f"- Conflicting label groups: {item['conflicting_groups']:,}",
        ])

    lines.extend([
        "",
        "## 3. Cross Dataset Analysis",
        "",
        f"- Train unique texts: {len(train_texts):,}",
        f"- Test unique texts: {len(test_texts):,}",
        f"- Overlapping texts: {len(overlap):,}",
        "",
        "## 4. Audit Notes",
        "",
        "- This audit does not modify or remove any raw records.",
        "- Duplicate and conflicting-label records require review "
        "before preprocessing.",
        "- Original labels use values 1 to 4.",
        ""
    ])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
        # ==========================================
    # 5. DUPLICATE AND DATA QUALITY FINDINGS
    # ==========================================

    train_audit = pd.DataFrame({
        "text": train_report["text"].values,
        "label": train_report["labels"].values
    })

    test_audit = pd.DataFrame({
        "text": test_report["text"].values,
        "label": test_report["labels"].values
    })

    train_audit = train_audit[train_audit["text"].ne("")]

    # Analyze duplicate groups
    duplicate_stats = (
        train_audit.groupby("text")["label"]
        .agg(["size", "nunique"])
    )

    duplicate_groups = duplicate_stats[
        duplicate_stats["size"] > 1
    ]

    consistent_groups = duplicate_groups[
        duplicate_groups["nunique"] == 1
    ]

    conflicting_groups = duplicate_groups[
        duplicate_groups["nunique"] > 1
    ]

    # Count conflicting label combinations
    pair_counts = {}

    for text in conflicting_groups.index:

        labels = sorted(
            train_audit.loc[
                train_audit["text"] == text,
                "label"
            ].unique()
        )

        pair = " vs ".join(
            LABEL_NAMES[int(label)]
            for label in labels
        )

        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    # Analyze Train-Test label consistency
    same_label_overlap = 0
    conflicting_overlap = 0

    for text in overlap:

        train_labels = set(
            train_audit.loc[
                train_audit["text"] == text,
                "label"
            ]
        )

        test_labels = set(
            test_audit.loc[
                test_audit["text"] == text,
                "label"
            ]
        )

        if train_labels == test_labels:
            same_label_overlap += 1
        else:
            conflicting_overlap += 1

    lines.extend([
        "",
        "## 5. Duplicate and Data Quality Findings",
        "",
        "### 5.1 Duplicate Text Analysis",
        "",
        f"The training dataset contains "
        f"{len(duplicate_groups)} duplicate text groups.",
        "",
        f"- Consistent-label groups: {len(consistent_groups)}",
        f"- Conflicting-label groups: {len(conflicting_groups)}",
        "",
        "The conflicting label distribution is:",
        "",
        "| Conflicting Labels | Number of Groups |",
        "|---|---:|"
    ])

    for pair, count in sorted(pair_counts.items()):

        lines.append(f"| {pair} | {count} |")

    lines.extend([
        f"| Total | {len(conflicting_groups)} |",
        "",
        "These inconsistencies indicate potential annotation "
        "ambiguity because identical articles are assigned "
        "different categories.",
        "",
        "Original labels should not be manually changed "
        "without a reliable source of ground truth.",
        "",
        "### 5.2 Train-Test Overlap",
        "",
        f"- Overlapping articles: {len(overlap)}",
        f"- Same-label overlaps: {same_label_overlap}",
        f"- Conflicting-label overlaps: {conflicting_overlap}",
        "",
        "Overlapping records may introduce data leakage "
        "if retained in both training and testing datasets.",
        "",
        "The preprocessing pipeline follows the project policy: "
        "retain testing records and remove corresponding "
        "training records.",
        "",
        "### 5.3 Data Quality Conclusion",
        "",
        "The AG News dataset is well-balanced, contains no "
        "missing values or empty articles, and has valid labels.",
        "",
        "Duplicate articles, conflicting labels, and "
        "train-test overlap are handled through the "
        "preprocessing policy.",
        "",
        "The original raw datasets remain unchanged. "
        "Cleaning and deduplication are performed on "
        "derived datasets.",
        ""
    ])

    OUTPUT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\nAudit report saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()