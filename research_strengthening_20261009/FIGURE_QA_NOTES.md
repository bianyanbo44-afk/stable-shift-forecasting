# Rendered figure QA

The uncertainty source validator reports one heuristic warning: forest intervals
are drawn as exact endpoint lines/caps with separate point markers, rather than
Matplotlib errorbar(). Each forest row reads the recorded paired percentile
interval, checks finite ordered endpoints and retains all settings. This is an
intentional representation; it does not impose that a point lie inside a
percentile interval. The actual rendered intervals were visually inspected.

Budget curves and update CDFs are explicitly descriptive. They retain every
declared budget/setting in the data; no test-best budget is selected. Disabled
policies have zero corrections, with conditional active-update quantiles marked
undefined rather than invented. Panel c's dataset quantile means therefore use
enabled settings; accuracy means include the disabled zero-change settings.

Runtime points are seed-11 median synchronized durations from three technical
repetitions after warm-up, excluding common source inference. They are not an
aggregate over source seeds or a population confidence statement. All three
repetitions and inactive policies remain in Source Data. Only pairs with both
QNC and COSA active enter the comparison plot.

Alignment tolerance is 1.5 pt; final PDF text and collision audits accompany each
figure. The budget legend was raised after the first render to add bottom-page
clearance; its final export was audited again. Source widths are 183 mm and
rasters 600 dpi. Colors/typography inherit only the author's original plotting
style; all result mappings and statistical families use the new evidence.

Before manuscript compilation, inspect the four complete final figures and
their panels again. PDF audits complement the visual and statistical checks;
they do not certify substantive author review.
