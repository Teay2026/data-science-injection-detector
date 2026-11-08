# Report

The path we followed and what we learned. Details are in the notebooks.

## Goal

Detect prompt injections in emails (instructions hidden for the AI assistant that reads them) with classic ML, and answer the four questions of the README.

## Our path

1. **Data**: 60 000 emails, half attacks from the LLMail-Inject challenge, half normal emails (LLMail and Enron). Duplicates removed before a fixed train/test split.
2. **Exploration**: many clues reveal the source, not the attack. Enron emails are shorter and have no accents, and almost all attacks target `contact@contact.com`.
3. **Bias check**: a model that reads no word, only the shape of the text, already reaches AUC 0.93. This became our bar.
4. **Words** (TF-IDF + logistic regression): AUC 0.97, but partly through shortcuts.
5. **Cleaning**: no gain (still 0.97), even without the shortcuts. It revealed 27 attacks hidden in invisible characters that the challenge had labelled normal.
6. **Characters**: AUC 0.98. We moved from Spark ML to scikit-learn, which handles this on a laptop.
7. **LSA and Word2Vec + XGBoost**: AUC up to 0.99, our best score.
8. **Evaluation**: we tested the models harder (below).

## Our results

| Model | AUC | New attacks detected | Look-alike letters detected |
|---|---|---|---|
| shape only, no words | 0.93 | – | – |
| words + logistic regression | 0.97 | 57 % | 58 % |
| characters + logistic regression | 0.98 | 60 % | 78 % |
| Word2Vec + XGBoost | 0.98 | 63 % | 93 % |
| LSA + XGBoost | 0.99 | 63 % | 88 % |

AUC on the random test split. New attacks: phase 2 attacks, with models trained on phase 1 only. Look-alike letters: test attacks rewritten with Cyrillic letters. All models except the first use the cleaned text B.

## Main takeaways

- **High scores are partly inflated**: by the source of the emails, by the challenge's target, and by near copies (half of the test attacks have one in training).
- **New attacks are the real problem**: trained on phase 1, the models catch about 60 % of the phase 2 attacks, against about 90 % on a random split. A detector must be retrained as attacks change.
- **Cleaning matters under attack, not on natural data**: invisible tricks cost up to 27 points without it, nothing with it. Look-alike letters still get through (word models drop under 60 %).
- **Word2Vec + XGBoost resists hidden attacks best**; word TF-IDF is the most fragile.
- **In a real mailbox attacks are rare**: with 1 attack in 1 000 emails, fewer than 1 alert in 4 is right. The alert threshold matters more than the AUC.

## Limits

- All attacks come from one challenge with one goal: we detect this kind of attack, not every prompt injection.
- Most labels come from the challenge's LLM judge, and most social engineering was labelled unclear and dropped.

## Next

Compare classic ML with LLMs on quality, speed and cost.
