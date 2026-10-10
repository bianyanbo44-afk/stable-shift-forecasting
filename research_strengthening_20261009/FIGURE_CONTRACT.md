# Figure and writing contract

The central question is whether bounding the effect of matured feedback on a separate current query remains useful after stronger tuning and query-independent adaptive controls. Actual complete results determine the wording; no favorable direction is assumed.

Canonical terminology: query-constrained calibration (QNC); independently tuned NLMS; historical-feedback clipping; controlled COSA; original benchmark re-evaluation; unused-entity transfer. The latter refers to new scored entity IDs with archived source checkpoints and fixed positional mapping on the same historical calendar. The old EIA period is a locked temporal extension only for its original configurations; new configurations are an exploratory re-evaluation of that period.

| Figure | Question | Panel roles | Sources and statistical units |
|---|---|---|---|
| 1, retained workflow | Which prediction is constrained? | Source workflow, maturity separation, proposal-ray control | Original deterministic schematic, no quantitative data |
| 2, expanded benchmark | Does the comparison survive stronger tuning? | Frozen gains; stronger output adapter; independently tuned normalized updates | All 12 original settings, five seeds, phase=test paired_comparisons, seven-day blocks, pointwise intervals and Holm family 72 |
| 3, mechanism and cost | What does current-query dependence add? | Matched-step NLMS accuracy; matched historical clipping accuracy; realized current-query displacement; separately timed scalar implementation | All 12 setting comparisons, 48-test matched family; all active test updates; all 12 seed-11 timing streams, warm-up plus three repetitions |
| 4, budget | How does the declared budget change accuracy and realized displacement? | All five budgets at fixed primary eta; all five with validation-selected eta; actual displacement across budgets | All original settings and seeds, descriptive curves; no test-best budget; inactive-policy displacement reported separately |
| 5, unused entities | Do fixed old-cohort configurations transfer to new scored entities? | Frozen comparison; independently tuned NLMS; independently tuned historical clipping | All eight cohort/source/horizon settings, five archived source seeds, paired intervals; primary family 48, historical family 16 |

All forest plots map relative_difference/relative_ci_low/relative_ci_high to percent error difference. Negative values favor QNC. Filled markers denote the specified Holm-adjusted threshold, without treating unfilled markers as evidence of equivalence. Error bars use exact interval endpoints; no assumption that the point must lie inside a percentile interval. No setting is removed for unfavorable outcomes.

Budget plots use five-seed pooled observed-entry errors at every declared rho. Thin gray curves show each setting; color curves give equal-setting dataset means. Geometry is descriptive, pooled over the actual active updates and explicitly marked when an entire configuration is disabled. No bootstrap is added post hoc to select a budget.

Use Python matplotlib throughout, inherit only the original project's palette/typography and forest-plot visual convention, rebuild data/statistical mapping explicitly. Width 183 mm, editable PDF/SVG and 600 dpi PNG/TIFF, minimum text 7 pt. Rendered panel alignment at 1.5 pt, source validation, PDF text audit and PDF collision audit are required. Every WARN must be inspected and documented. Complete data and inference go to Source Data/SI; the main text uses MAE plus the necessary uncertainty and conclusion-changing boundaries.

Writing allocation: expanded benchmark results replace the original small-grid main accuracy paragraphs. Mechanism controls replace the old single constraint-ablation claim. Budget and new-entity evidence are central. The original locked EIA experiment and initialization intervention remain explicitly labeled historical evidence in SI, preserving their original provenance and results. The author-led AI disclosure remains accurate about all actual assistance.
