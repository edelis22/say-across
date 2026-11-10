# Automatic emotion detection from text

Recorded speech uses the separate [voice-only emotion model](VOICE_MODEL.md).

The target is affect expressed or implied by a written utterance as perceived by readers. Keep reader interpretation, a speaker’s stated intention, and any private self-report as separate annotations. The classifier receives only the source sentence and optional surrounding text. It runs before the translation adapter and can be evaluated by itself. Inferences are hypotheses about the wording and can affect translation only as weak hints when the speaker has supplied no intent annotation.

## Detector

The pinned checkpoint is Horizon Labs’ Apache-2.0 multilingual emotions model, exported as quantized ONNX. Its training draws on human-annotated English GoEmotions comments and machine-translated comments with copied labels, including Spanish, Greek and Japanese. Its 28 independent outputs represent 27 GoEmotions labels and neutral. It has limited exposure to native non-English dialogue. [Model card](https://huggingface.co/Horizon-Labs/multilingual-emotions-small) · [Training dataset card](https://huggingface.co/datasets/Horizon-Labs/go-emotions-multilingual) · [Original GoEmotions paper](https://aclanthology.org/2020.acl-main.372/)

The model outputs logits. The application takes the sigmoid independently for each label. These scores describe model activation and need calibration on the intended domain. When surrounding sentences fit within 256 tokens together with the source, the detector computes three views. It scores the source alone, the joined text, and the context alone. A label qualifies at its published GoEmotions validation threshold. The context-aware policy uses scores from the joined text and requires a qualifying label to also qualify from the source alone or gain at least 0.15 over the context-only score. This is a heuristic to reduce attribution errors. A qualifying neutral score that is within 0.10 of, or exceeds, the strongest eligible non-neutral score yields uncertainty. Uncertainty is distinct from neutral wording and from a runtime failure. The source sentence stays intact if the context exceeds the token limit. Adaptation receives detected labels only if their raw score is at least 0.55 and the speaker supplied no emotion annotation. This extra threshold is provisional and does not imply a calibrated probability.

## Questions and measures

1. Can the detector distinguish emotions when no user label or context is available? Evaluate per-language and per-label precision, recall and F1. Treat an abstention as an empty prediction for F1 on clearly labeled items. Report exact set matches and coverage alongside F1.
2. Does contextual evidence improve the interpretation of the target sentence? Compare the same items with and without context. Report paired results by language and by stratum, including both sides correct for matched-context scenarios. Include a context-only condition when investigating attribution. Report errors on quoted emotion, another speaker’s emotion, mixed affect, irony, politeness and brief ambiguous replies.
3. Is uncertainty expressed appropriately? Distinguish the neutral category from unresolved cases. Report the fraction of ambiguous cases flagged uncertain. Report coverage and conditional error at different score thresholds, and per-label Brier scores and reliability diagrams on clear items. Document failures to return scores. Fit calibration or tune thresholds on a separate development set and freeze them before testing.
4. Do the labels improve translation? Compare blinded ratings for the full arm and the no-emotion-signal arm with the same context and preferences. Analyze an additional subset where the detector actually supplied a hint, alongside the comparison across all items. Track cases where a label changes the apparent emotion, escalates tone or invents a motive.

## Build the held-out set

Have fluent reviewers author native examples within each language and locale, with everyday and difficult cases. Before model scoring, collect at least three independent judgments per item. Ask reviewers to label the **target sentence**, select multiple emotions where they appear to co-occur, or mark neutral wording or insufficient evidence. Have them identify the words or contextual cues that informed the judgment and tag any emotion attributed to someone other than the target speaker. Gather judgments for the text-only and context-visible conditions separately. Use separate annotator pools or counterbalanced order with a washout period to reduce recall.

Keep annotators blind to model predictions and proposed author labels. Preserve their individual judgments, report per-label agreement and aggregate agreement, and predefine a consensus rule. For example, apply strict majority per label and retain an unresolved state when no clear majority emerges or neutral and affective judgments conflict. Do not force a single emotion on ambiguous wording. Track test languages separately and resample scenario groups for confidence intervals. Keep every variant of a scenario in the same split. Document provenance, participant consent where applicable, and domain, locale and scenario coverage.

The bundled pilot is a diagnostic set with provisional author labels. A null gold value marks a proposed ambiguous case. Both text-only and context-visible predictions are scored against the same context-visible author labels to measure the change when context is available. A larger study should report both this diagnostic comparison and agreement with separately elicited text-only labels. Pilot results provide an engineering check and can guide design. Independent estimates require the held-out annotation process above.

## Observed pilot results

I ran the pinned ONNX checkpoint and the full 44-case synthetic pilot locally. Forty cases have proposed gold labels. Four have proposed ambiguous readings. The observations below use the published per-label thresholds for display and the documented context policy.

| Measure | Sentence alone | Context when supplied |
|---|---:|---:|
| Exact label-set matches on clear cases | 18 / 40 | 18 / 40 |
| Micro-F1 on clear cases | 0.595 | 0.602 |
| Clear cases with a label, including neutral | 35 / 40 | 33 / 40 |
| Ambiguous cases marked uncertain | 1 / 4 | 1 / 4 |
| Matched-context pairs with both labels correct | 0 / 4 | 0 / 4 |

The model correctly identified gratitude from the wording in all four languages, fear in one Spanish item and sadness in one Japanese item. It correctly kept the four neutral targets neutral despite emotional distractors. It missed every complete matched-context pair and sometimes treated short ambiguous replies as neutral. The 0.55 minimum score was also tried as a display threshold on the same pilot and reduced exact matches to 15 out of 40. The app uses it only for adaptation hints.

I also tried Horizon Labs' larger ONNX checkpoint at revision `4f95ad28e89def7cce33b0589de4ed369b970deb` with the same context policy and its published per-label thresholds. With context it matched 16 of 40 exact label sets, had micro-F1 of 0.633, and got both sides correct on zero pairs. The smaller checkpoint stays the default to limit the download. These are exploratory comparisons on the same authored examples. The separate bilingual study is needed to estimate real-world accuracy and to determine whether the hints improve translation. [Larger checkpoint model card](https://huggingface.co/Horizon-Labs/multilingual-emotions-base)
