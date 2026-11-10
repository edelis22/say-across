# Study protocol for reviewer discussion

## Questions and decision rules

The experimental unit is an independently authored source scenario with one designated source and target direction. The paired contrast uses the same sentence, scenario and eligible raters for each system. Treat paraphrases, counterfactual context pairs and multiple target versions of one scenario as a single cluster when resampling. Use independent scenarios in the supplied analysis script, which resamples individual item IDs.

1. **Meaning preservation gate.** The full system should be noninferior in adequacy to raw NLLB and Google NMT. Proposed margin is 0.25 on a 1–5 scale. Confirm the margin with bilingual reviewers before seeing confirmatory outputs. For each baseline, require the lower confidence bound of full minus baseline adequacy to exceed −0.25.
2. **Primary benefit.** After the adequacy gate passes, test whether full improves tone preservation relative to each baseline. Use the same paired contrasts. Require both lower confidence bounds to exceed zero for a joint claim. Cultural appropriateness is a key secondary endpoint. Report effect sizes and intervals rather than ranking systems on a combined score.
3. **Safety and operational trade-offs.** Estimate the safety pass rate, critical failure rate, fallback rate, and added latency. For a joint positive claim, have reviewers predefine a tolerable safety degradation before collection and require its uncertainty bound to meet that margin. Report severity categories with representative failures.
4. **Mechanism.** Full versus no context estimates the combined contribution of neighboring sentences, relationship, locale, requested tone and stated intent. Full versus no surface guards replays the same model response and isolates the checks for number and URL changes. It retains schema, abstention and control-character checks. The no-emotion-signal condition tests the contribution of the detector's hint at the same context and user preferences. All ablations are exploratory. [Independent emotion evaluation](EMOTION_PROTOCOL.md)

Use interval-based gates on the joint primary claims, with both baseline comparisons required for each gate. Treat individual direction, stratum, ablation and secondary intervals as exploratory unless separately preregistered with multiplicity control. Report failures to pass the gates as inconclusive or contrary evidence as appropriate. The pilot provides no efficacy estimate.

## Conditions

| Arm | Inputs available to system | Output rated |
|---|---|---|
| NLLB | Source and explicit language codes | Raw NLLB sentence |
| Google NMT | Same source, mapped language codes, plain text | Google Cloud Translation Basic v2 general NMT |
| Full | Source, raw NLLB, optional neighboring sentences and user preferences, plus eligible local emotion labels | Accepted revision or the NLLB fallback |
| No context | Source and raw NLLB with the same adapter, default preferences and empty context. The detector sees the source alone | Accepted revision or fallback |
| No surface guards | Same response captured from the full condition | Candidate when the only rejection is a number or URL mismatch |
| No emotion signal | Same context and preferences as full, with the detector hint omitted | Accepted revision or fallback |

Google Cloud Translation is the specific Google Translate baseline for this study. Record collection dates and archive raw outputs, as the provider model can change. The baseline APIs receive the same information they normally accept. The no-emotion-signal arm reuses the full response when there was no detector hint. With all ablations enabled, the collector makes up to three OpenAI calls per item and one Google call if enabled. Comparisons against them measure the complete workflows, and the matched ablation investigates the contribution of context. A future separately frozen arm can test direct LLM translation from the same inputs with the NLLB draft removed. [Google Cloud translation documentation](https://docs.cloud.google.com/translate/docs/translate-text)

## Sampling and provenance

- Recruit authors and raters familiar with the target locales and fluent in both languages of their assigned directions. Record locale familiarity and direction-level qualifications without identifying individuals in the released dataset.
- Develop separate authoring, rubric training, and confirmatory sets. Keep scenario families together and seal the confirmatory set before prompt tuning. Document origin, author consent or license, edits, intended audience, ambiguity, intended tone, and why each context field matters. Keep identifying information out of the dataset. Obtain appropriate ethics review before collecting participant data.
- Balance the 12 directions and sample ordinary prose as well as indirect speech, register shifts, idioms, ambiguous emotion, sarcasm, loss or conflict, quantity and negation, safety-relevant imperatives and instruction-like content. Author matched context pairs for diagnostic evaluation. Avoid treating countries as uniform language communities. Include realistic variance in age, region and register through scenario authoring, with sensitivity review.
- After a separate pilot, estimate per-item paired variance and rater variability and simulate power for the adequacy margin and a prespecified practically meaningful tone gain. Freeze target sample size, planned exclusions, hypotheses and analysis before confirmatory collection. Allocate more than one rater per item and target at least three independent eligible raters. Keep calibration training items distinct.

## Blinding and adjudication

Generate outputs and lock the run manifest. A coordinator distributes only the matching anonymous packet and CSV, plus the rater guide. Each packet independently randomizes all trials. Ratings include the source and the same context for every arm, even where a baseline could not receive it. Exclude explanations, scores, raw candidates, system names and fallback metadata from packets. Identical outputs and distinctive wording can still reveal clues. Record suspected unblinding and report it.

The bundled packer sends the full set to each pseudonymous rater. For a larger study, partition generation and packet creation by eligible language direction, use distinct packet directories, seeds and `--rater-prefix` values, and combine the resulting maps and output records before pooling. Keep the seed and key private until scoring is locked. Have raters score independently, with breaks and no access to translation services. Treat absence of a usable translation as an operational failure. Report its count and run a sensitivity analysis assigning unavailable outputs the worst eligible outcome. Keep accepted and fallback outputs in the full arm. Report analysis on completed same-rater pairs alongside missingness by arm. Use independent second-pass adjudication for severe errors and publish both raw agreement and adjudication outcomes.

## Analysis and reproducibility

- Compute mean rater scores per item, paired same-rater differences per item and 95% percentile bootstrap intervals over independent items. The script performs these calculations and reports arm-level counts. For stratified or clustered confirmatory designs, stratify resampling by direction and resample scenario clusters. For inference across a wider population of raters, use a crossed item/rater model or a crossed bootstrap. Treat pooled script intervals as conditional on recruited raters.
- Report item-macro results and paired differences by direction and stratum. Include null and negative effects, fallback reasons, exact duplicates, missing rows and severity categories. Scores use 1–5 for adequacy, tone and culture. Safety is a binary pass. Compute Krippendorff’s alpha per dimension on independent, pre-adjudication ratings of each item-arm output, using squared distance on the 1–5 scale and binary distance for safety. Report item counts and bootstrap agreement intervals for confirmatory subsets. Alpha can be undefined if ratings never vary.
- For adaptation calibration, define success as a strict majority of at least three independent raters assigning adequacy ≥4. Report Brier score and reliability bins on displayed accepted adaptations, with sample counts and coverage. Report coverage against all full-arm items, including fallbacks, and bin uncertainty in the confirmatory study. Model self-estimates are conditional on acceptance. Reserve a distinct calibration split if fitting any calibration model.
- Record warm median and p95 generation latency with sample counts and separately report the first item as a cold-start observation. The study runner serializes arm calls, measures shared NLLB once per sentence and adds its time to each timed adapted arm. It records local detection time separately. Timings exclude the Gradio queue and browser, and replay timing is omitted. Run a separate repeated latency benchmark with fixed hardware, region, cache state, concurrency and interleaved service order if comparing operational latency.
- Archive the frozen source tree or commit, prompt version and hash, package freeze, translation and classifier model revisions, detector policy version, returned API identifiers, hardware, raw artifacts, dataset checksum, seeds and analysis version. Temperature zero and pinned weights still allow platform and provider variation. Avoid retaining sensitive content in logs.

## Review milestones

1. Verify installation, GPU or CPU selection, live API shape and all initial directions on the research machine. Check that consent, credentials and spend caps match the pilot.
2. Have bilingual reviewers revise the pilot prompts and rubric. Use the pilot to estimate annotation time and tune error categories. Freeze the confirmatory protocol before collecting held-out ratings.
3. Run the blinded study, publish uncertainty and failure cases, and review licensing for any broader use. Extend language and locale coverage in cohorts with direction-specific validation.

## Voice evaluation

For speech-to-speech trials, retain the same consented recording across conditions. Record the source language, microphone setup, background noise and speaker variety. Use separate speakers across development and held-out sets. Collect a human transcript with a fixed normalization policy. Report word error rate for English, Spanish and Greek and character error rate for Japanese, with separate counts for dropped or changed names, numbers and negation.

Compare translation from the automatic transcript with translation from the human transcript to estimate errors introduced by speech recognition. Have native listeners rate the synthesized translation for intelligibility, meaning and tone. Use the same synthesis configuration across conditions when isolating translation quality. Report median and p95 elapsed time from recording stop to playable output, and separate model startup, transcription, translation and synthesis times. Synthetic audio checks verify that the software can complete a turn. Held-out human recordings are needed to estimate speech recognition and playback quality.

For two-way interpretation, test alternating speakers on one microphone, short replies, overlapping speech, language pairs with similar sounds, and out-of-pair audio. Report correct direction, wrong direction, and abstention separately. Measure time from the last voiced frame to the first caption and to the first audible output. Count dropped or clipped turns and unintended re-translation of the app's playback. Compare fast system playback and voice matching on the same equipment.

For cultural phrasing, hold the words fixed and vary the situation or relationship. Include literal controls where a phrase resembles an idiom and held-out idioms absent from the local catalogue. Score the intended sense, factual adequacy, register and naturalness independently. For example, “table the motion” can be paired with explicitly supplied U.S. Senate and U.K. Parliament contexts. Record whether an item used the local rule, NLLB, or optional revision. [U.S. Senate glossary](https://www.senate.gov/glossary/) · [U.K. Parliament glossary](https://www.parliament.uk/site-information/glossary/table/)
