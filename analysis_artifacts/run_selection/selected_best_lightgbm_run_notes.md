# Selected best LightGBM run

- `run_key`: `W72_H24`
- `window_hours`: `72`
- `horizon_hours`: `24`
- `experiment`: `weighted_thresholded`
- `macro_f1_5c`: `0.364946`

Selected by the Section 6 LightGBM ranking logic: among runs marked usable_for_comparison, choose the highest best 5-class test Macro-F1. The notebook explicitly states that W72_H24 is the strongest balanced 5-class LightGBM baseline.

Notebook Section 6 also explicitly states that `W72_H24` is the strongest overall LightGBM baseline by 5-class test Macro-F1.
