# TSMC-Depression-Anxiety

**A New Thai Social Media Corpus for Depression and Anxiety Detection with Fine-grained Emotion Intensity Annotations**

---

## Overview

This corpus is part of the PhD research *"Exploring Emotion Timeline Patterns in Social Media for Automatic Identification of Depression and Anxiety."* It contains chronological tweets from Thai users, where each user is labelled for the presence or absence of depression and anxiety. Each tweet is further annotated with emotion-intensity scores across 26 fine-grained emotion categories, labelled manually by human experts.

### Dataset Files

| File | Description |
|------|-------------|
| `Dataset/users.csv` | One row per user with depression and anxiety labels |
| `Dataset/timelines_all.csv` | Chronological tweets per user with emotion-intensity labels with all emotions annotated by annotator |
| `Dataset/timelines_ds.csv` | Chronological tweets per user with emotion-intensity labels aggregated using Dawid-Skene algorithm |

**Label encoding** — the users file contains both gold string labels (`gold_depression`: `"depressed"`/`"none"`, `gold_anxiety`: `"anxious"`/`"none"`) and binary labels (`depression`/`anxiety`: `0` = absent, `1` = present).

**Privacy** — all personally identifiable information has been replaced with pseudonyms, including names, addresses, dates, places of work or study, job titles, salaries, phone numbers, email addresses, social media handles, user mentions, URLs, IP addresses, and other personal details.

---

## Usage

### Requirements

```bash
pip install pandas
```

### 1. Import the Dataset

```python
import pandas as pd

# Load user-level labels
users = pd.read_csv("Dataset/users.csv", index_col=0)

# Load tweet timelines with emotion-intensity labels
# created_date has mixed UTC offsets (e.g. +00:00, +01:00) — read as string first,
# then convert with utc=True to normalise everything to a single UTC-aware dtype
timelines = pd.read_csv("Dataset/timelines_ds.csv", index_col=0)
timelines["created_date"] = pd.to_datetime(timelines["created_date"], utc=True)

print(users.head())
print(timelines.head())
```

**Users file columns:** `participant_id`, `gold_depression`, `gold_anxiety`, `depression`, `anxiety`
- `participant_id`: string identifier in the format `participant_0`, `participant_1`, …
- `gold_depression` / `gold_anxiety`: string labels — `"depressed"` / `"none"` and `"anxious"` / `"none"`
- `depression` / `anxiety`: binary labels — `0` = absent, `1` = present

**Timelines file columns:** `participant_id`, `created_date`, `anonymised_text`, followed by 26 emotion-intensity columns (scores range from **0.0 to 1.0**): `admiration`, `amusement`, `anger`, `annoyance`, `anticipation`, `approval`, `confusion`, `curiosity`, `desire`, `disapproval`, `disgust`, `disappointed`, `excitement`, `fear`, `gratitude`, `grief`, `joy`, `love`, `nervousness`, `optimism`, `pride`, `realisation`, `remorse`, `sadness`, `surprise`, `trust`
- `participant_id`: matches the format in the users file (e.g. `participant_0`)
- `created_date`: timezone-aware timestamp with mixed UTC offsets (e.g. `2023-03-09 05:17:54+00:00`, `2023-04-20 04:24:13+01:00`); normalised to UTC after loading

### 2. Select a User and Their Timeline

```python
# Select a specific user by participant_id
target_id = users["participant_id"].iloc[0]  # or set any participant_id directly

# Get all tweets for that user, sorted chronologically
user_timeline = (
    timelines[timelines["participant_id"] == target_id]
    .sort_values("created_date")
)

print(f"Tweets for participant {target_id}:")
print(user_timeline[["participant_id", "created_date", "anonymised_text"]])
```

You can also filter users by condition label:

```python
# Using gold labels (string)
depressed_users = users[users["gold_depression"] == "depressed"]
anxious_users   = users[users["gold_anxiety"] == "anxious"]
comorbid_users  = users[(users["gold_depression"] == "depressed") & (users["gold_anxiety"] == "anxious")]

# Using binary labels
depressed_users = users[users["depression"] == 1]

# Get timelines for all depressed users
depressed_timelines = timelines[timelines["participant_id"].isin(depressed_users["participant_id"])]
```

### 3. Select a Specific Emotion from the Timeline

```python
# All 26 emotion columns
emotion_columns = [
    "admiration", "amusement", "anger", "annoyance", "anticipation",
    "approval", "confusion", "curiosity", "desire", "disapproval",
    "disgust", "disappointed", "excitement", "fear", "gratitude",
    "grief", "joy", "love", "nervousness", "optimism", "pride",
    "realisation", "remorse", "sadness", "surprise", "trust"
]

# Select a single emotion
emotion_data = timelines[["participant_id", "created_date", "sadness"]]
print(emotion_data.head())
```

Select multiple emotions at once:

```python
target_emotions = ["sadness", "fear", "grief"]
emotion_subset = timelines[["participant_id", "created_date"] + target_emotions]
print(emotion_subset.head())
```

Filter tweets where a specific emotion exceeds a threshold:

```python
# Tweets with high sadness intensity (emotion scores range from 0.0 to 1.0)
high_sadness = timelines[timelines["sadness"] > 0.5][["participant_id", "created_date", "sadness"]]
print(high_sadness)
```

Aggregate emotion intensity per user over their full timeline:

```python
# Mean emotion intensity per user across all their tweets
user_emotion_profile = (
    timelines.groupby("participant_id")[emotion_columns]
    .mean()
    .reset_index()
)

# Merge with user labels for downstream analysis
user_emotion_profile = user_emotion_profile.merge(
    users[["participant_id", "gold_depression", "gold_anxiety", "depression", "anxiety"]],
    on="participant_id"
)
print(user_emotion_profile.head())
```

---

<!--
## Citation

If you use this dataset in your research, please cite:

```bibtex
@phdthesis{3f4692eaab864083970aeb2cf903fb57,
title = "Exploring Emotion Timeline Patterns in Social Media for Automatic Identification of Depression and Anxiety",
abstract = "Over the past years, public awareness of mental health issues has increased, and people are paying more attention to their mental well-being. Depressive and anxiety disorders are among the key causes of the global disease burden. These two mental conditions can disrupt the daily lives of people affected by them and, in severe cases, depression may even lead to suicide. To prevent suicide and alleviate the burden on the affected individuals, researchers have increasingly focused on automatically detecting these mental conditions using social media data, as people with mental health issues often seek supports and share their feelings on social media.Since depression and anxiety are chronic issues and related to emotions, past studies have utilised emotion information to identify social media users with depression and anxiety. However, these emotion features used in the past studies are often static, lacking the temporal fluctuation patterns that can effectively reflect the characteristics of these mental conditions. To address this issue, I propose a novel method that applies the timeline Shapelet classification method, a time series analysis method, to the automatic detection of depression and anxiety based on emotion intensity timeline patterns.This study involves two languages. I first created a new Thai dataset for depression and anxiety detection. Next, I selected and modified an existing English dataset for depression detection. Both datasets contain users' tweeting histories in X. Tweets in the Thai dataset were labelled with 26 emotion intensities by both human annotators and a Large Language Model (LLM). In contrast, tweets in the English dataset were automatically labelled with intensities of four emotions using an LLM. Based on the manually annotated Thai dataset, depression and anxiety are found to be associated with 14 and 10 emotions respectively. My study demonstrates that Shapelet classifiers can identify emotion intensity timeline patterns associated with depression and anxiety. Using these patterns, the classifiers detected depression with a precision of 0.7292 on the English dataset, and achieved maximum precision of 1.00 in detecting both depression and anxiety on the Thai dataset. These classifiers also outperformed benchmark models on the Thai dataset in detecting the two conditions, although they produced relatively low recall on both datasets. Finally, this study also investigates the potential of porting emotion intensity timeline patterns for cross-lingual depression detection, from English to Thai in this case. In my experiment, the Shapelet classifiers trained on English dataset could effectively detect depression in the Thai dataset, showing the potential of this approach for cross-lingual application, particularly for under-resourced languages.",
keywords = "Thai NLP, Emotion, Depression, Anxiety, Automatic detection, Social media, LLM, shapelets, Time series analysis, NLP",
author = "Ratchakrit Arreerard",
year = "2026",
doi = "10.17635/lancaster/thesis/3121",
language = "English",
publisher = "Lancaster University",
school = "Computing and Communications, Lancaster University",
}
```
--!>
