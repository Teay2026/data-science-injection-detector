# Prompt injection detection in emails

Data science project with Pandas and PySpark. We detect prompt injections in emails with classic ML models, and measure how text cleaning changes the result, including on hidden injections.

## Questions

1. Does cleaning the text improve detection?
2. Which text representation resists hidden injections best: words, characters, LSA or Word2Vec?
3. Does a detector trained on phase 1 attacks still catch phase 2 attacks, made after defenses changed? Real life: like a spam filter trained last year, facing this year's spam.
4. Does classic ML detect injections better than LLMs, and at what cost?

## Data

- Injections: [LLMail-Inject](https://huggingface.co/datasets/microsoft/llmail-inject-challenge) (Microsoft, MIT licence), texts labelled `true` by the challenge.
- Normal emails: LLMail texts labelled `false`, the challenge's legitimate emails, and a sample of [Enron](https://huggingface.co/datasets/corbt/enron-emails).

`notebooks/01_data.ipynb` downloads everything (pinned revisions) and builds `data/processed/emails.parquet`: 30 000 emails per class, duplicates removed, fixed 80/20 train/test split.

27 LLMail texts labelled normal hide an attack in invisible characters, missed by the challenge's judge (found in notebook 5). Notebook 1 relabels them as injections after the sampling (`reason = hidden_text`): 30 027 injections, 29 973 normal emails.

Most LLMail labels come from the challenge's LLM judge. Only `api_triggered` labels are sure: the attack made the LLM call the email API. The `reason` column keeps this, so ML and LLMs can also be compared on sure labels only.

## Setup

Python 3.10 and Java 17.

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

On Windows, Spark needs `winutils.exe` and `hadoop.dll` in `%HADOOP_HOME%\bin` to write Parquet.

## Plan

- [x] 1. Data: one balanced dataset, duplicates removed, fixed train/test split
- [x] 2. Exploration
- [x] 3. Source bias check: a model that only sees the shape of the text
- [x] 4. Iteration 1: raw text, word TF-IDF, pruning, logistic regression and Naive Bayes
- [x] 5. Iteration 2: cleaned text
- [x] 6. Iteration 3: character n-grams
- [ ] 7. Iteration 4: LSA and Word2Vec, then SVM and XGBoost
- [ ] 8. Evaluation: obfuscation test, phase 1 to phase 2, errors by attack type
- [ ] 9. ML vs LLM on 1 000 test emails: local LLM, small dedicated model, then an LLM through an API. Quality, speed and cost.
