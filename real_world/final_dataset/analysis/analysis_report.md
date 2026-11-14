# Final dataset analysis

## Core scale and validity

- **50 events and 150 articles**, with exactly three human-accepted, distinct-domain articles per event.
- **59,358 final English word tokens**. Article length: median 288.5, mean 395.7, range 60--1504 words.
- All selected articles passed the five human criteria: same event, complete text, correct time/place, independent source, and useful information.

## Event coverage

- Disaster types: EQ 13, TC 12, FL 9, WF 7, VO 5, DR 4.
- Years: 2023 11, 2024 15, 2025 13, 2026 11.
- GDACS severity: 17 Red and 33 Orange alerts.
- The metadata contains 69 distinct country/territory labels. Coordinates span multiple world regions; multi-country events are counted once as events but in every involved country in the location chart.

## News-source diversity

- **84 publisher domains**; 59 occur only once.
- The three most frequent sources account for 26.0% of articles; the top ten account for 40.7%.
- Source HHI is 0.0313 (effective source count 32.0). This is diverse overall, although English Xinhua, Devdiscourse, and MENAFN are comparatively frequent.

## Human cleaning

- **46 / 150 (30.7%)** selected articles were manually edited.
- Editors removed 2,346 words in total. Among edited articles, the median reduction was 9.4% and the mean was 15.1%.
- The public release retains only cleaned `content`, the original-content hash, and derived removal statistics; raw original text remains private.

## Important interpretation

- Drought articles have much longer event-to-publication lags than sudden-onset hazards; this is expected because GDACS drought events cover long periods. Report lag by hazard type rather than only as one global average.
- The final set is balanced across years globally, but not every hazard occurs in every year (for example, selected tropical cyclones are concentrated in 2023--2024).
- All 150 selected articles are English. Describe the released real-world subset as English-language even though the collection pipeline can retain Chinese.

## Recommended paper presentation

Use Figure 1 and a compact dataset-statistics table in the main paper. Figures 2--4 are suitable for the appendix. The most important reported numbers are: 50 events, 150 articles, six hazard types, four years, 17 Red/33 Orange alerts, 84 publishers, 30.7% manually cleaned articles, and three independent sources per event.
