# AG News - Data Audit Report

## 1. Dataset Overview

| Dataset | Rows | Columns |
|---|---:|---:|
| train | 120,000 | 3 |
| test | 7,600 | 3 |

## 2. TRAIN Dataset

### 2.1 Missing Values

| Column | Missing |
|---|---:|
| Class Index | 0 |
| Title | 0 |
| Description | 0 |

### 2.2 Label Distribution

| Label | Category | Count | Percentage |
|---:|---|---:|---:|
| 1 | World | 30,000 | 25.00% |
| 2 | Sports | 30,000 | 25.00% |
| 3 | Business | 30,000 | 25.00% |
| 4 | Sci/Tech | 30,000 | 25.00% |

### 2.3 Duplicate and Invalid Data

- Exact duplicate rows: 0
- Duplicate text rows: 79
- Duplicate text groups: 79
- Empty articles: 0
- Invalid labels: 0
- Conflicting label groups: 15

## 2. TEST Dataset

### 2.1 Missing Values

| Column | Missing |
|---|---:|
| Class Index | 0 |
| Title | 0 |
| Description | 0 |

### 2.2 Label Distribution

| Label | Category | Count | Percentage |
|---:|---|---:|---:|
| 1 | World | 1,900 | 25.00% |
| 2 | Sports | 1,900 | 25.00% |
| 3 | Business | 1,900 | 25.00% |
| 4 | Sci/Tech | 1,900 | 25.00% |

### 2.3 Duplicate and Invalid Data

- Exact duplicate rows: 0
- Duplicate text rows: 0
- Duplicate text groups: 0
- Empty articles: 0
- Invalid labels: 0
- Conflicting label groups: 0

## 3. Cross Dataset Analysis

- Train unique texts: 119,921
- Test unique texts: 7,600
- Overlapping texts: 4

## 4. Audit Notes

- This audit does not modify or remove any raw records.
- Duplicate and conflicting-label records require review before preprocessing.
- Original labels use values 1 to 4.


## 5. Duplicate and Data Quality Findings

### 5.1 Duplicate Text Analysis

The training dataset contains 79 duplicate text groups.

- Consistent-label groups: 64
- Conflicting-label groups: 15

The conflicting label distribution is:

| Conflicting Labels | Number of Groups |
|---|---:|
| Business vs Sci/Tech | 12 |
| World vs Business | 2 |
| World vs Sci/Tech | 1 |
| Total | 15 |

These inconsistencies indicate potential annotation ambiguity because identical articles are assigned different categories.

Original labels should not be manually changed without a reliable source of ground truth.

### 5.2 Train-Test Overlap

- Overlapping articles: 4
- Same-label overlaps: 4
- Conflicting-label overlaps: 0

Overlapping records may introduce data leakage if retained in both training and testing datasets.

The preprocessing pipeline follows the project policy: retain testing records and remove corresponding training records.

### 5.3 Data Quality Conclusion

The AG News dataset is well-balanced, contains no missing values or empty articles, and has valid labels.

Duplicate articles, conflicting labels, and train-test overlap are handled through the preprocessing policy.

The original raw datasets remain unchanged. Cleaning and deduplication are performed on derived datasets.
