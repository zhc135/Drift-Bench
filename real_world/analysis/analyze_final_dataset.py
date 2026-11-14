from __future__ import annotations

import csv
import json
import math
import re
import shutil
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "final_dataset"
OUTPUT = DATASET / "analysis"
TYPE_ORDER = ["DR", "EQ", "FL", "TC", "VO", "WF"]
TYPE_NAMES = {
    "DR": "Drought", "EQ": "Earthquake", "FL": "Flood",
    "TC": "Tropical cyclone", "VO": "Volcanic eruption", "WF": "Wildfire",
}
TYPE_COLORS = {
    "DR": "CDr", "EQ": "CEq", "FL": "CFl",
    "TC": "CTc", "VO": "CVo", "WF": "CWf",
}
COLOR_DEFINITIONS = r"""
\definecolor{CDr}{HTML}{CC79A7}
\definecolor{CEq}{HTML}{D55E00}
\definecolor{CFl}{HTML}{0072B2}
\definecolor{CTc}{HTML}{56B4E9}
\definecolor{CVo}{HTML}{E69F00}
\definecolor{CWf}{HTML}{009E73}
\definecolor{COrange}{HTML}{E69F00}
\definecolor{CRed}{HTML}{D55E00}
\definecolor{CBlue}{HTML}{0072B2}
"""


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text or "", flags=re.UNICODE))


def parse_date(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def percentile(values: Sequence[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def five_number(values: Sequence[float]) -> Dict[str, float]:
    return {
        "min": min(values), "q1": percentile(values, 0.25),
        "median": percentile(values, 0.5), "q3": percentile(values, 0.75),
        "max": max(values),
    }


def tex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
        "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{",
        "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def coordinates(pairs: Iterable[tuple[Any, Any]]) -> str:
    return " ".join(f"({x},{y})" for x, y in pairs)


def base_tex(body: str) -> str:
    return rf"""\documentclass[tikz,border=4pt]{{standalone}}
\usepackage{{pgfplots}}
\usepgfplotslibrary{{groupplots,statistics}}
\pgfplotsset{{compat=1.18}}
{COLOR_DEFINITIONS}
\begin{{document}}
{body}
\end{{document}}
"""


def compile_figure(output: Path, stem: str, tex: str) -> None:
    tex_path = output / f"{stem}.tex"
    tex_path.write_text(tex, encoding="utf-8")
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        return
    for _ in range(2):
        result = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=output, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        if result.returncode:
            raise RuntimeError(f"Failed to compile {tex_path}:\n{result.stdout[-5000:]}")
    gs = shutil.which("gs")
    if gs:
        png_path = output / f"{stem}.png"
        subprocess.run([
            gs, "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=pngalpha",
            "-r300", "-dTextAlphaBits=4", "-dGraphicsAlphaBits=4",
            f"-sOutputFile={png_path}", str(output / f"{stem}.pdf"),
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for suffix in (".aux", ".log", ".synctex.gz"):
        generated = output / f"{stem}{suffix}"
        if generated.exists():
            generated.unlink()


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (DATASET / "events.csv").open(encoding="utf-8-sig") as handle:
        events = list(csv.DictReader(handle))
    articles = [
        json.loads(line) for line in (DATASET / "articles.jsonl").read_text(
            encoding="utf-8"
        ).splitlines() if line.strip()
    ]
    event_by_id = {event["event_id"]: event for event in events}
    article_rows: List[Dict[str, Any]] = []
    by_event: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for article in articles:
        event = event_by_id[article["event_id"]]
        final_words = word_count(article["content"])
        original_words = int(article.get("original_word_count") or 0)
        if not original_words and article.get("original_content"):
            # Backward-compatible fallback for pre-release dataset builds.
            original_words = word_count(article["original_content"])
        lag_days = (
            parse_date(article["published_at"]) - parse_date(event["start_date"])
        ).total_seconds() / 86400
        row = {
            "event_id": article["event_id"],
            "article_id": article["article_id"],
            "disaster_type": event["disaster_type_code"],
            "event_year": event["start_date"][:4],
            "alert_level": event["gdacs_alert_level"],
            "source_domain": article["source_domain"],
            "final_word_count": final_words,
            "original_word_count": original_words,
            "content_edited": bool(article["content_edited"]),
            "removed_word_count": max(0, original_words - final_words),
            "removed_word_ratio": (
                max(0, original_words - final_words) / original_words
                if original_words else 0.0
            ),
            "publication_lag_days": lag_days,
        }
        article_rows.append(row)
        by_event[article["event_id"]].append(row)

    event_rows: List[Dict[str, Any]] = []
    for event in events:
        rows = by_event[event["event_id"]]
        event_rows.append({
            "event_id": event["event_id"],
            "event_name": event["event_name"],
            "disaster_type": event["disaster_type_code"],
            "year": event["start_date"][:4],
            "alert_level": event["gdacs_alert_level"],
            "countries": event["countries"],
            "latitude": event["latitude"],
            "longitude": event["longitude"],
            "article_count": len(rows),
            "mean_article_words": round(statistics.mean(
                row["final_word_count"] for row in rows
            ), 2),
            "median_publication_lag_days": round(statistics.median(
                row["publication_lag_days"] for row in rows
            ), 3),
            "edited_article_count": sum(row["content_edited"] for row in rows),
        })

    domain_counts = Counter(row["source_domain"] for row in article_rows)
    domain_events: Dict[str, set[str]] = defaultdict(set)
    for row in article_rows:
        domain_events[row["source_domain"]].add(row["event_id"])
    source_rows = [{
        "source_domain": domain,
        "article_count": count,
        "article_share": round(count / len(article_rows), 6),
        "event_count": len(domain_events[domain]),
    } for domain, count in domain_counts.most_common()]

    type_counts = Counter(event["disaster_type_code"] for event in events)
    year_counts = Counter(event["start_date"][:4] for event in events)
    alert_counts = Counter(event["gdacs_alert_level"] for event in events)
    language_counts = Counter(article["original_language"] for article in articles)
    country_counts: Counter[str] = Counter()
    for event in events:
        country_counts.update(
            value.strip() for value in re.split(r"[;,]", event["countries"])
            if value.strip()
        )
    words = [row["final_word_count"] for row in article_rows]
    edited_rows = [row for row in article_rows if row["content_edited"]]
    edit_ratios = [row["removed_word_ratio"] for row in edited_rows]
    hhi = sum((count / len(article_rows)) ** 2 for count in domain_counts.values())

    word_stats_by_type = {
        code: five_number([
            row["final_word_count"] for row in article_rows
            if row["disaster_type"] == code
        ]) for code in TYPE_ORDER
    }
    lag_stats_by_type = {
        code: five_number([
            row["publication_lag_days"] for row in article_rows
            if row["disaster_type"] == code
        ]) for code in TYPE_ORDER
    }
    summary = {
        "events": len(events),
        "articles": len(articles),
        "articles_per_event": sorted(Counter(
            row["event_id"] for row in article_rows
        ).values()),
        "disaster_type_distribution": dict(sorted(type_counts.items())),
        "year_distribution": dict(sorted(year_counts.items())),
        "alert_distribution": dict(sorted(alert_counts.items())),
        "language_distribution": dict(sorted(language_counts.items())),
        "country_or_territory_labels": len(country_counts),
        "top_country_involvement": country_counts.most_common(15),
        "unique_source_domains": len(domain_counts),
        "singleton_source_domains": sum(count == 1 for count in domain_counts.values()),
        "top_3_source_share": round(sum(
            count for _, count in domain_counts.most_common(3)
        ) / len(article_rows), 4),
        "top_10_source_share": round(sum(
            count for _, count in domain_counts.most_common(10)
        ) / len(article_rows), 4),
        "source_hhi": round(hhi, 6),
        "effective_source_count": round(1 / hhi, 2),
        "total_final_words": sum(words),
        "article_words": {
            "min": min(words), "median": statistics.median(words),
            "mean": round(statistics.mean(words), 2), "max": max(words),
        },
        "edited_articles": len(edited_rows),
        "edited_article_share": round(len(edited_rows) / len(article_rows), 4),
        "words_removed_by_editing": sum(row["removed_word_count"] for row in edited_rows),
        "edited_article_removal_ratio": {
            "median": round(statistics.median(edit_ratios), 4),
            "mean": round(statistics.mean(edit_ratios), 4),
            "max": round(max(edit_ratios), 4),
        },
        "word_count_by_type": word_stats_by_type,
        "publication_lag_days_by_type": lag_stats_by_type,
    }
    (OUTPUT / "summary_statistics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(OUTPUT / "article_statistics.csv", article_rows)
    write_csv(OUTPUT / "event_statistics.csv", event_rows)
    write_csv(OUTPUT / "source_statistics.csv", source_rows)

    make_composition_figure(events)
    make_geography_figure(events, country_counts)
    make_text_figure(word_stats_by_type, lag_stats_by_type)
    make_sources_editing_figure(domain_counts, edit_ratios)
    write_report(summary)


def make_composition_figure(events: List[Dict[str, Any]]) -> None:
    years = ["2023", "2024", "2025", "2026"]
    by_year_type = Counter(
        (event["start_date"][:4], event["disaster_type_code"]) for event in events
    )
    by_type_alert = Counter(
        (event["disaster_type_code"], event["gdacs_alert_level"]) for event in events
    )
    year_plots = "\n".join(
        rf"\addplot+[ybar,fill={TYPE_COLORS[code]},draw=none] coordinates {{{coordinates((year, by_year_type[(year, code)]) for year in years)}}};"
        for code in TYPE_ORDER
    )
    body = rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.55cm}},
  width=8.0cm,height=6.2cm,tick label style={{font=\small}},label style={{font=\small}},
  title style={{font=\small}},legend style={{font=\scriptsize,draw=none,fill=none}}]
\nextgroupplot[ybar stacked,bar width=13pt,symbolic x coords={{2023,2024,2025,2026}},
  xtick=data,ymin=0,ymax=17,ytick distance=5,ymajorgrids,grid style={{gray!20}},
  xlabel={{Event year}},ylabel={{Number of events}},title={{(a) Temporal and hazard composition}},
  legend columns=3,legend to name=typelegend]
{year_plots}
\legend{{DR,EQ,FL,TC,VO,WF}}
\nextgroupplot[ybar stacked,bar width=13pt,symbolic x coords={{DR,EQ,FL,TC,VO,WF}},
  xtick=data,ymin=0,ymax=15,ytick distance=5,ymajorgrids,grid style={{gray!20}},
  xlabel={{Disaster type}},ylabel={{Number of events}},title={{(b) GDACS alert severity}},
  legend pos=north east]
\addplot+[ybar,fill=COrange,draw=none] coordinates {{{coordinates((code, by_type_alert[(code, 'Orange')]) for code in TYPE_ORDER)}}};
\addplot+[ybar,fill=CRed,draw=none] coordinates {{{coordinates((code, by_type_alert[(code, 'Red')]) for code in TYPE_ORDER)}}};
\legend{{Orange,Red}}
\end{{groupplot}}
\node at ($(group c1r1.south)!0.5!(group c2r1.south)+(0,-1.05cm)$) {{\ref{{typelegend}}}};
\end{{tikzpicture}}"""
    compile_figure(OUTPUT, "figure_1_dataset_composition", base_tex(body))


def make_geography_figure(
    events: List[Dict[str, Any]], country_counts: Counter[str],
) -> None:
    scatter = []
    for code in TYPE_ORDER:
        points = []
        for event in events:
            if event["disaster_type_code"] != code:
                continue
            try:
                points.append((float(event["longitude"]), float(event["latitude"])))
            except (TypeError, ValueError):
                continue
        scatter.append(
            rf"\addplot+[only marks,mark=*,mark size=2.2pt,{TYPE_COLORS[code]}] coordinates {{{coordinates(points)}}};"
        )
    top = country_counts.most_common(12)
    labels = [tex_escape(country) for country, _ in reversed(top)]
    country_coords = " ".join(
        f"({count},{{{tex_escape(country)}}})" for country, count in reversed(top)
    )
    body = rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.7cm}},
  width=9.0cm,height=6.8cm,tick label style={{font=\scriptsize}},label style={{font=\small}},
  title style={{font=\small}},legend style={{font=\scriptsize,draw=none,fill=none}}]
\nextgroupplot[xmin=-180,xmax=180,ymin=-60,ymax=80,xtick={{-180,-120,-60,0,60,120,180}},
  ytick={{-60,-30,0,30,60}},grid=both,grid style={{gray!18}},
  xlabel={{Longitude}},ylabel={{Latitude}},title={{(a) Event-coordinate coverage}},
  legend columns=3,legend pos=south east]
{chr(10).join(scatter)}
\legend{{DR,EQ,FL,TC,VO,WF}}
\nextgroupplot[xbar,bar width=7pt,symbolic y coords={{{','.join(labels)}}},ytick=data,
  xmin=0,xmax=6,xtick distance=1,xmajorgrids,grid style={{gray!18}},
  xlabel={{Events involving country/territory}},title={{(b) Most represented locations}},
  nodes near coords,nodes near coords style={{font=\scriptsize}},
  enlarge y limits=0.06]
\addplot+[xbar,fill=CBlue,draw=none] coordinates {{{country_coords}}};
\end{{groupplot}}
\end{{tikzpicture}}"""
    compile_figure(OUTPUT, "figure_2_geographic_coverage", base_tex(body))


def boxplots(stats: Dict[str, Dict[str, float]]) -> str:
    plots = []
    for index, code in enumerate(TYPE_ORDER, 1):
        value = stats[code]
        plots.append(rf"""\addplot[boxplot prepared={{draw position={index},
lower whisker={value['min']:.4f},lower quartile={value['q1']:.4f},
median={value['median']:.4f},upper quartile={value['q3']:.4f},
upper whisker={value['max']:.4f}}},draw={TYPE_COLORS[code]},fill={TYPE_COLORS[code]}!25,
thick] coordinates {{}};""")
    return "\n".join(plots)


def make_text_figure(
    word_stats: Dict[str, Dict[str, float]], lag_stats: Dict[str, Dict[str, float]],
) -> None:
    body = rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.55cm}},
  width=8.2cm,height=6.4cm,tick label style={{font=\small}},label style={{font=\small}},
  title style={{font=\small}},boxplot/draw direction=y,
  xtick={{1,2,3,4,5,6}},xticklabels={{DR,EQ,FL,TC,VO,WF}},xmin=0.5,xmax=6.5]
\nextgroupplot[ymin=0,ymax=1600,ytick distance=400,ymajorgrids,grid style={{gray!18}},
  xlabel={{Disaster type}},ylabel={{Final article length (words)}},
  title={{(a) Cleaned-text length distribution}}]
{boxplots(word_stats)}
\nextgroupplot[ymin=-5,ymax=135,ytick distance=25,ymajorgrids,grid style={{gray!18}},
  xlabel={{Disaster type}},ylabel={{Publication lag from event start (days)}},
  title={{(b) Event--article temporal alignment}}]
{boxplots(lag_stats)}
\end{{groupplot}}
\end{{tikzpicture}}"""
    compile_figure(OUTPUT, "figure_3_text_and_temporal_statistics", base_tex(body))


def make_sources_editing_figure(
    domain_counts: Counter[str], edit_ratios: List[float],
) -> None:
    top = domain_counts.most_common(12)
    labels = [tex_escape(domain) for domain, _ in reversed(top)]
    source_coords = " ".join(
        f"({count},{{{tex_escape(domain)}}})" for domain, count in reversed(top)
    )
    bins = [(0, .05), (.05, .10), (.10, .25), (.25, .50), (.50, 1.01)]
    bin_labels = ["0--5", "5--10", "10--25", "25--50", "50+"]
    bin_counts = [
        sum((low <= value < high) for value in edit_ratios) for low, high in bins
    ]
    body = rf"""\begin{{tikzpicture}}
\begin{{groupplot}}[group style={{group size=2 by 1,horizontal sep=1.85cm}},
  width=9.0cm,height=6.9cm,tick label style={{font=\scriptsize}},label style={{font=\small}},
  title style={{font=\small}}]
\nextgroupplot[xbar,bar width=7pt,symbolic y coords={{{','.join(labels)}}},ytick=data,
  xmin=0,xmax=16,xtick distance=4,xmajorgrids,grid style={{gray!18}},
  xlabel={{Articles}},title={{(a) Most frequent publisher domains}},
  nodes near coords,nodes near coords style={{font=\scriptsize}},enlarge y limits=0.06]
\addplot+[xbar,fill=CBlue,draw=none] coordinates {{{source_coords}}};
\nextgroupplot[ybar,bar width=16pt,symbolic x coords={{{','.join(bin_labels)}}},xtick=data,
  ymin=0,ymax=22,ytick distance=5,ymajorgrids,grid style={{gray!18}},
  xlabel={{Words removed from edited article (\%)}},ylabel={{Edited articles}},
  title={{(b) Human text-cleaning magnitude}},
  nodes near coords,nodes near coords style={{font=\small}}]
\addplot+[ybar,fill=COrange,draw=none] coordinates {{{coordinates(zip(bin_labels, bin_counts))}}};
\end{{groupplot}}
\end{{tikzpicture}}"""
    compile_figure(OUTPUT, "figure_4_sources_and_human_editing", base_tex(body))


def write_report(summary: Dict[str, Any]) -> None:
    words = summary["article_words"]
    edit = summary["edited_article_removal_ratio"]
    report = f"""# Final dataset analysis

## Core scale and validity

- **50 events and 150 articles**, with exactly three human-accepted, distinct-domain articles per event.
- **{summary['total_final_words']:,} final English word tokens**. Article length: median {words['median']:.1f}, mean {words['mean']:.1f}, range {words['min']}--{words['max']} words.
- All selected articles passed the five human criteria: same event, complete text, correct time/place, independent source, and useful information.

## Event coverage

- Disaster types: EQ 13, TC 12, FL 9, WF 7, VO 5, DR 4.
- Years: 2023 11, 2024 15, 2025 13, 2026 11.
- GDACS severity: 17 Red and 33 Orange alerts.
- The metadata contains {summary['country_or_territory_labels']} distinct country/territory labels. Coordinates span multiple world regions; multi-country events are counted once as events but in every involved country in the location chart.

## News-source diversity

- **{summary['unique_source_domains']} publisher domains**; {summary['singleton_source_domains']} occur only once.
- The three most frequent sources account for {summary['top_3_source_share'] * 100:.1f}% of articles; the top ten account for {summary['top_10_source_share'] * 100:.1f}%.
- Source HHI is {summary['source_hhi']:.4f} (effective source count {summary['effective_source_count']:.1f}). This is diverse overall, although English Xinhua, Devdiscourse, and MENAFN are comparatively frequent.

## Human cleaning

- **{summary['edited_articles']} / {summary['articles']} ({summary['edited_article_share'] * 100:.1f}%)** selected articles were manually edited.
- Editors removed {summary['words_removed_by_editing']:,} words in total. Among edited articles, the median reduction was {edit['median'] * 100:.1f}% and the mean was {edit['mean'] * 100:.1f}%.
- The public release retains only cleaned `content`, the original-content hash, and derived removal statistics; raw original text remains private.

## Important interpretation

- Drought articles have much longer event-to-publication lags than sudden-onset hazards; this is expected because GDACS drought events cover long periods. Report lag by hazard type rather than only as one global average.
- The final set is balanced across years globally, but not every hazard occurs in every year (for example, selected tropical cyclones are concentrated in 2023--2024).
- All 150 selected articles are English. Describe the released real-world subset as English-language even though the collection pipeline can retain Chinese.

## Recommended paper presentation

Use Figure 1 and a compact dataset-statistics table in the main paper. Figures 2--4 are suitable for the appendix. The most important reported numbers are: 50 events, 150 articles, six hazard types, four years, 17 Red/33 Orange alerts, 84 publishers, 30.7% manually cleaned articles, and three independent sources per event.
"""
    (OUTPUT / "analysis_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
