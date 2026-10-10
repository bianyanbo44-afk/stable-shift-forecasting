# Third-party notices

## Time-Series-Library

- Upstream: https://github.com/thuml/Time-Series-Library
- Recorded commit: `4e938a1767106324dd753b2a44832bf870a0252e`
- Location: `vendor_cache/Time-Series-Library/`
- License: MIT, copyright 2021 THUML @ Tsinghua University.

The upstream license is preserved in that directory. StableShift's wrapper in `src/stable_shift_forecasting/upstream_baselines.py` records the integration, including the TiDE scalar-output normalization treatment described in the supplement.

Original data are obtained from EIA, UCI ECL, Ausgrid and the recorded archival PJM distribution. Their respective provider terms apply; the repository's software license does not replace those terms.

## COSA output adapter

- Authors: Jeonghwan Im and Hyuk-Yoon Kwon.
- Upstream: https://github.com/bigbases/COSA_ICLR2026
- Pinned commit: `527c0feb9e997dd85af485ee027616b446e4ae77`.
- Location: `literature/kbs_extension_20260924/cosa/`.
- Actual pinned license: Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International. The full original `LICENSE` is preserved, including its noncommercial and share-alike conditions.
- The upstream README mentions MIT; this conflicts with the pinned license file. This archive follows and reports the actual license file instead of relabeling COSA as MIT.

The controlled comparison uses the released linear output module with a common daily matured-feedback schedule. Its vectorized forward path and scheduling wrapper are described in Supplement S3. Native period scheduling, partial-horizon updates and retrospective forecast replacement are disabled. Original file notices, including TAFAS-derived notices, remain preserved. The original project's root MIT license does not override these third-party terms.
