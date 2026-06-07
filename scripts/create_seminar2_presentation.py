from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
EXPORT_ROOT = ROOT / "exports" / "latest_results_20260607_165644" / "exports"
OUT_DIR = ROOT / "seminar"
ASSET_DIR = OUT_DIR / "wystapienie2_assets"
PPTX_PATH = OUT_DIR / "wystapienie2_latent_framsticks.pptx"
NOTES_PATH = OUT_DIR / "wystapienie2_tekst_do_slajdow.md"


BG = RGBColor(15, 23, 42)
BG2 = RGBColor(22, 33, 54)
PAPER = RGBColor(245, 241, 232)
PAPER2 = RGBColor(36, 51, 79)
TEXT = RGBColor(245, 241, 232)
MUTED = RGBColor(170, 181, 199)
INK = RGBColor(24, 31, 45)
CYAN = RGBColor(57, 208, 200)
AMBER = RGBColor(240, 184, 75)
RED = RGBColor(231, 111, 81)
GREEN = RGBColor(119, 178, 85)
VIOLET = RGBColor(161, 127, 224)
LINE = RGBColor(63, 81, 112)


@dataclass
class SlideText:
    title: str
    speaker: str


def read_csv(name: str) -> list[dict[str, str]]:
    with (EXPORT_ROOT / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(value: Any) -> float:
    return float(value)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def method_stats(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(f(row["best_score"]))
    out: dict[str, dict[str, float]] = {}
    for method, values in grouped.items():
        arr = np.asarray(values, dtype=float)
        out[method] = {
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "min": float(arr.min()),
        }
    return out


def compute_summary() -> dict[str, Any]:
    compare = read_csv("latent_operator_ea_compare_generators_cap1p2_20260530_151817.csv")
    transformer = read_csv("latent_operator_ea_transformer_deep_cap1p2_20260530_151822.csv")
    tree = read_csv("latent_operator_ea_tree_deep_cap1p2_20260530_151825.csv")
    evolution = read_csv("latent_evolution_generators_deep_no_slp_cap1p2_20260530_151818.csv")

    pairs: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in compare:
        pairs[(row["label"], row["seed_rank"])][row["method"]] = f(row["best_score"])
    wins = Counter()
    diffs = []
    for values in pairs.values():
        if "latent" in values and "frams" in values:
            wins["latent" if values["latent"] > values["frams"] else "frams"] += 1
            diffs.append(values["latent"] - values["frams"])

    all_rows = compare + transformer + tree + evolution
    best = max(all_rows, key=lambda row: f(row["best_score"]))

    return {
        "compare": compare,
        "transformer": transformer,
        "tree": tree,
        "evolution": evolution,
        "compare_stats": method_stats(compare),
        "wins": wins,
        "diffs": diffs,
        "best": best,
    }


def plot_compare(summary: dict[str, Any]) -> Path:
    rows = summary["compare"]
    labels = []
    frams = []
    latent = []
    pairs: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        pairs[(row["label"], row["seed_rank"])][row["method"]] = f(row["best_score"])
    for (label, seed), values in sorted(pairs.items()):
        labels.append(("Tree" if label.startswith("09") else "Trans") + f" s{seed}")
        frams.append(values.get("frams", np.nan))
        latent.append(values.get("latent", np.nan))

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.bar(x - width / 2, frams, width, label="Framsticks mutate/crossOver", color="#f0b84b")
    ax.bar(x + width / 2, latent, width, label="Operatory w przestrzeni ukrytej", color="#39d0c8")
    ax.axhline(1.2, color="#64748b", linewidth=1.0, linestyle="--", label="limit fitness seeda")
    ax.set_ylabel("Najlepszy vertpos", color="#f5f1e8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", color="#f5f1e8")
    ax.tick_params(axis="y", colors="#f5f1e8")
    ax.grid(axis="y", alpha=0.18, color="#f5f1e8")
    for spine in ax.spines.values():
        spine.set_visible(False)
    leg = ax.legend(frameon=False, loc="upper left")
    for text in leg.get_texts():
        text.set_color("#f5f1e8")
    ax.set_title("Ten sam budżet: 5000 ewaluacji true fitness na parę", color="#f5f1e8", pad=14)
    fig.tight_layout()
    path = ASSET_DIR / "compare_frams_latent.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_deep(summary: dict[str, Any]) -> Path:
    transformer = sorted(summary["transformer"], key=lambda row: int(row["seed_rank"]))
    tree = sorted(summary["tree"], key=lambda row: int(row["seed_rank"]))
    seeds = [int(row["seed_rank"]) for row in transformer]
    trans_values = [f(row["best_score"]) for row in transformer]
    tree_values = [f(row["best_score"]) for row in tree]
    x = np.arange(len(seeds))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5.2), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    ax.bar(x - width / 2, trans_values, width, label="TransformerVAE", color="#39d0c8")
    ax.bar(x + width / 2, tree_values, width, label="TreeVAE", color="#a17fe0")
    ax.axhline(1.2, color="#64748b", linewidth=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {seed}" for seed in seeds], color="#f5f1e8")
    ax.tick_params(axis="y", colors="#f5f1e8")
    ax.set_ylabel("Najlepszy vertpos", color="#f5f1e8")
    ax.grid(axis="y", alpha=0.18, color="#f5f1e8")
    for spine in ax.spines.values():
        spine.set_visible(False)
    leg = ax.legend(frameon=False, loc="upper left")
    for text in leg.get_texts():
        text.set_color("#f5f1e8")
    ax.set_title("Głębsze uruchomienia tylko w przestrzeni ukrytej: 20 000 ewaluacji na seed", color="#f5f1e8", pad=14)
    fig.tight_layout()
    path = ASSET_DIR / "deep_latent_results.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_convergence() -> Path:
    compare_hist = read_csv("latent_operator_ea_compare_generators_cap1p2_20260530_151817.history.csv")
    deep_hist = read_csv("latent_operator_ea_transformer_deep_cap1p2_20260530_151822.history.csv")
    selected = [
        (compare_hist, "10_transformer_vae_epoch90", "2", "latent", "EA w przestrzeni ukrytej, Transformer s2", "#39d0c8"),
        (compare_hist, "10_transformer_vae_epoch90", "2", "frams", "Framsticks EA, Transformer s2", "#f0b84b"),
        (deep_hist, "10_transformer_vae_epoch90", "1", "latent", "Deep search w przestrzeni ukrytej, s1", "#77b255"),
        (deep_hist, "10_transformer_vae_epoch90", "3", "latent", "Deep search w przestrzeni ukrytej, s3", "#a17fe0"),
    ]
    fig, ax = plt.subplots(figsize=(10.8, 5.4), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    for rows, label, seed, method, legend, color in selected:
        points = [
            row for row in rows
            if row["label"] == label and row["seed_rank"] == seed and row["method"] == method
        ]
        points.sort(key=lambda row: int(row["generation"]))
        ax.plot(
            [f(row["evaluated_unique"]) for row in points],
            [f(row["best_score"]) for row in points],
            linewidth=2.0,
            label=legend,
            color=color,
        )
    ax.set_xlabel("Liczba unikalnych ewaluacji", color="#f5f1e8")
    ax.set_ylabel("Najlepszy vertpos", color="#f5f1e8")
    ax.tick_params(colors="#f5f1e8")
    ax.grid(alpha=0.18, color="#f5f1e8")
    for spine in ax.spines.values():
        spine.set_visible(False)
    leg = ax.legend(frameon=False, loc="lower right")
    for text in leg.get_texts():
        text.set_color("#f5f1e8")
    ax.set_title("Przykładowe przebiegi optymalizacji", color="#f5f1e8", pad=14)
    fig.tight_layout()
    path = ASSET_DIR / "convergence.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def add_bg(slide, accent: RGBColor = CYAN) -> None:
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = BG
    shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.06))
    shape.fill.solid()
    shape.fill.fore_color.rgb = accent
    shape.line.fill.background()


def add_textbox(slide, text: str, x: float, y: float, w: float, h: float, size: int = 24, color: RGBColor = TEXT,
                bold: bool = False, align: PP_ALIGN | None = None) -> Any:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    p = tf.paragraphs[0]
    run = p.add_run()
    run.text = text
    run.font.name = "Aptos"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    if align is not None:
        p.alignment = align
    return box


def add_title(slide, title: str, kicker: str | None = None, accent: RGBColor = CYAN) -> None:
    if kicker:
        add_textbox(slide, kicker.upper(), 0.68, 0.36, 6.5, 0.32, 9, accent, True)
    add_textbox(slide, title, 0.65, 0.67, 11.2, 0.62, 28, TEXT, True)


def add_card(slide, x: float, y: float, w: float, h: float, title: str, body: str, accent: RGBColor = CYAN) -> None:
    line = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(x), Inches(y), Inches(0.04), Inches(h))
    line.fill.solid()
    line.fill.fore_color.rgb = accent
    line.line.fill.background()
    add_textbox(slide, title, x + 0.18, y, w - 0.24, 0.34, 15, TEXT, True)
    add_textbox(slide, body, x + 0.18, y + 0.48, w - 0.24, h - 0.5, 12, MUTED, False)


def add_metric(slide, x: float, y: float, w: float, value: str, label: str, accent: RGBColor) -> None:
    add_textbox(slide, value, x, y + 0.04, w, 0.42, 25, accent, True, PP_ALIGN.CENTER)
    rule = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(x + 0.35), Inches(y + 0.55), Inches(w - 0.7), Inches(0.02))
    rule.fill.solid()
    rule.fill.fore_color.rgb = LINE
    rule.line.fill.background()
    add_textbox(slide, label, x + 0.05, y + 0.68, w - 0.1, 0.32, 9, MUTED, False, PP_ALIGN.CENTER)


def add_bullets(slide, items: list[str], x: float, y: float, w: float, h: float, size: int = 15, color: RGBColor = TEXT) -> None:
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.clear()
    for idx, item in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.text = item
        p.font.name = "Aptos"
        p.font.size = Pt(size)
        p.font.color.rgb = color
        p.space_after = Pt(7)
        p.level = 0
        p.margin_left = Inches(0.18)
        p.text = f"• {item}"


def add_table(slide, rows: list[list[str]], x: float, y: float, w: float, h: float, widths: list[float] | None = None) -> None:
    table_shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h))
    table = table_shape.table
    if widths:
        for idx, width in enumerate(widths):
            table.columns[idx].width = Inches(width)
    for r_idx, row in enumerate(rows):
        for c_idx, value in enumerate(row):
            cell = table.cell(r_idx, c_idx)
            cell.text = value
            cell.fill.solid()
            cell.fill.fore_color.rgb = PAPER2 if r_idx == 0 else BG2
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.name = "Aptos"
                paragraph.font.size = Pt(10 if r_idx else 9)
                paragraph.font.bold = bool(r_idx == 0)
                paragraph.font.color.rgb = TEXT


def add_picture(slide, path: Path, x: float, y: float, w: float, h: float | None = None) -> None:
    if h is None:
        slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w))
    else:
        slide.shapes.add_picture(str(path), Inches(x), Inches(y), width=Inches(w), height=Inches(h))


def create_presentation(summary: dict[str, Any], charts: dict[str, Path]) -> list[SlideText]:
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]
    notes: list[SlideText] = []

    # 1
    slide = prs.slides.add_slide(blank)
    add_bg(slide, CYAN)
    add_textbox(slide, "Wystąpienie nr 2", 0.75, 0.48, 4.8, 0.35, 12, CYAN, True)
    add_textbox(
        slide,
        "Porównanie optymalizacji\nkompatybilnych podmian i optymalizacji\nw przestrzeni ukrytej dla genotypów Framsticks F1",
        0.75,
        1.08,
        11.2,
        1.8,
        30,
        TEXT,
        True,
    )
    add_textbox(slide, "Wystąpienie 2: wyniki części autoenkoderowej i operatorów w przestrzeni ukrytej", 0.78, 3.08, 10.6, 0.45, 17, MUTED)
    add_metric(slide, 0.8, 4.25, 2.25, "1.965", "najlepszy vertpos w przestrzeni ukrytej", CYAN)
    add_metric(slide, 3.35, 4.25, 2.25, "5 / 10", "wygrane vs Framsticks", AMBER)
    add_metric(slide, 5.9, 4.25, 2.25, "20k", "ewaluacji w deep run", VIOLET)
    add_textbox(slide, "Filip Rosiak", 0.82, 6.38, 3.0, 0.3, 13, TEXT, True)
    add_textbox(slide, "Promotor: dr hab. inż. Maciej Komosiński", 3.25, 6.38, 5.1, 0.3, 13, MUTED)
    add_textbox(slide, "Seminarium dyplomowe, 2026", 9.65, 6.38, 2.7, 0.3, 13, MUTED)
    notes.append(SlideText("Metryczka", "Witam, nazywam się Filip Rosiak. Temat całej pracy to porównanie CoSO i optymalizacji w przestrzeni ukrytej dla genotypów Framsticks F1. W tym wystąpieniu skupiam się głównie na części autoenkoderowej: pokazuję, co udało się zbudować i czy operatory wykonywane w przestrzeni ukrytej mogą zastępować standardową mutację i krzyżowanie Framsticks."))

    # 2
    slide = prs.slides.add_slide(blank)
    add_bg(slide, AMBER)
    add_title(slide, "Problem wspólny dla CoSO i przestrzeni ukrytej", "kontekst", AMBER)
    add_card(slide, 0.75, 1.65, 3.7, 3.65, "Problem", "Genotyp F1 ma zmienną długość i strukturę drzewiastą. Znaczenie fragmentu zależy od kontekstu, więc lokalna zmiana może zniszczyć dobrą podstrukturę.", AMBER)
    add_card(slide, 4.75, 1.65, 3.7, 3.65, "Pomysł", "Zamiast bezpośrednio mutować napis F1, uczymy autoenkoder i wykonujemy operacje na wektorach z w przestrzeni ukrytej.", CYAN)
    add_card(slide, 8.75, 1.65, 3.7, 3.65, "Weryfikacja", "Każdy kandydat po dekodowaniu jest oceniany tym samym prawdziwym kryterium Framsticks: vertpos. Porównanie jest wykonywane przy kontrolowanym budżecie ewaluacji.", VIOLET)
    notes.append(SlideText("Problem i teza", "Punktem wyjścia jest trudność reprezentacji F1. Nie jest to zwykły wektor parametrów, tylko opis strukturalny. CoSO i optymalizacja w przestrzeni ukrytej odpowiadają na ten sam problem z dwóch stron: CoSO przez zgodność fragmentów genotypu, a część autoenkoderowa przez reprezentację uczoną. W tej prezentacji weryfikuję głównie drugą część: czy mutacja i krzyżowanie w przestrzeni z mogą generować lepsze warianty niż operacje bezpośrednio na genotypie."))

    # 3
    slide = prs.slides.add_slide(blank)
    add_bg(slide, CYAN)
    add_title(slide, "Co udało się wykonać", "implementacja", CYAN)
    add_card(slide, 0.7, 1.55, 3.85, 4.55, "Modele", "VAE gramatyczne i drzewiaste, warianty LHS/depth/conditional, TransformerVAE, modele fitness-aware oraz modele trenowane z biasem wysokiego fitness.", CYAN)
    add_card(slide, 4.75, 1.55, 3.85, 4.55, "Algorytmy", "CEM, CMA-ES, ewolucja w przestrzeni ukrytej, EA z operatorami ukrytymi oraz warianty kontrolne: standardowe operatory Framsticks, grammar CEM i mutation search.", AMBER)
    add_card(slide, 8.8, 1.55, 3.85, 4.55, "Infrastruktura", "Wrapper Framsticks vertpos, wspólny format CSV/history/details, skrypty Slurm, eksport wyników, smoke testy oraz porównywalne protokoły seedów.", VIOLET)
    notes.append(SlideText("Co wykonano", "Wykonałem trzy warstwy pracy. Pierwsza to modele autoenkoderów. Druga to algorytmy optymalizacji i porównania. Trzecia to infrastruktura eksperymentalna, bez której wyniki nie byłyby porównywalne: wrapper do Framsticks, wspólne eksporty, historia przebiegu, skrypty Slurm i kontrolowane seedy."))

    # 4
    slide = prs.slides.add_slide(blank)
    add_bg(slide, VIOLET)
    add_title(slide, "Pipeline eksperymentalny", "metoda", VIOLET)
    labels = ["dane F1", "encoder", "wektor z", "operatory z", "decoder", "Framsticks"]
    xs = [0.8, 2.7, 4.65, 6.55, 8.55, 10.55]
    colors = [PAPER, PAPER, PAPER, CYAN, PAPER, AMBER]
    for x, label, color in zip(xs, labels, colors):
        shape = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Inches(x), Inches(2.55), Inches(1.45), Inches(0.9))
        shape.fill.solid()
        shape.fill.fore_color.rgb = color
        shape.line.fill.background()
        add_textbox(slide, label, x + 0.05, 2.82, 1.35, 0.25, 12, INK, True, PP_ALIGN.CENTER)
    for x in xs[:-1]:
        add_textbox(slide, "→", x + 1.48, 2.73, 0.45, 0.3, 24, MUTED, True, PP_ALIGN.CENTER)
    add_card(slide, 0.9, 4.35, 5.25, 1.35, "Badana metoda", "Operatory ewolucyjne wykonują mutację, krzyżowanie, interpolację i ekstrapolację na wektorach z w przestrzeni ukrytej.", CYAN)
    add_card(slide, 6.65, 4.35, 5.25, 1.35, "Baseline", "Ten sam protokół uruchamia standardowe Framsticks mutate/crossOver na tych samych seedach i budżetach.", AMBER)
    notes.append(SlideText("Pipeline", "Schemat jest prosty: dane F1 są kodowane do wektora z, w tej przestrzeni wykonujemy operacje ewolucyjne, potem dekodujemy wynik do genotypu i oceniamy prawdziwym Framsticks vertpos. Ważne jest to, że baseline używa tej samej puli seedów i tego samego budżetu, tylko operatory wykonuje bezpośrednio przez Framsticks."))

    # 5
    slide = prs.slides.add_slide(blank)
    add_bg(slide, RED)
    add_title(slide, "Protokół i ograniczenia", "warunki", RED)
    add_metric(slide, 0.75, 1.55, 2.25, "10 893", "genotypy w zbiorze F1", CYAN)
    add_metric(slide, 3.25, 1.55, 2.25, "≤ 1.2", "fitness seedów startowych", AMBER)
    add_metric(slide, 5.75, 1.55, 2.25, "vertpos", "prawdziwa funkcja celu", VIOLET)
    add_metric(slide, 8.25, 1.55, 2.25, "5k / 20k", "budżety ewaluacji", GREEN)
    add_bullets(slide, [
        "Zakres tej części pracy: autoenkodery i optymalizacja w przestrzeni ukrytej, nie główna implementacja CoSo.",
        "Wyniki są stochastyczne: istotna jest stabilność między seedami, a nie tylko pojedynczy rekord.",
        "Koszt ewaluacji Framsticks ogranicza liczbę powtórzeń, dlatego raportuję też przebiegi i budżety.",
        "Dekoder może wygenerować poprawny składniowo, ale słaby funkcjonalnie genotyp. Rekonstrukcja nie gwarantuje dobrej optymalizacji.",
    ], 0.85, 3.25, 11.4, 2.55, 15, TEXT)
    notes.append(SlideText("Protokół i ograniczenia", "Eksperymenty były ograniczone do części autoenkoderowej. Seedy startowe były celowo ograniczane do fitness około 1.2, żeby metoda musiała faktycznie poprawiać rozwiązanie. Wszystkie wartości fitness pochodzą z prawdziwego Framsticks vertpos. Istotne ograniczenie jest takie, że sama jakość rekonstrukcji nie wystarcza: model może świetnie odtwarzać przykłady, ale słabo generować nowe, dobre osobniki."))

    # 6
    slide = prs.slides.add_slide(blank)
    add_bg(slide, CYAN)
    add_title(slide, "Rekonstrukcja nie wystarcza", "wynik negatywny", CYAN)
    add_table(slide, [
        ["Obserwacja", "Co pokazały eksperymenty", "Znaczenie"],
        ["Najlepsze rekonstruktory", "modele LHS/conditional/structural osiągały najlepszą zgodność walidacyjną", "nie były najlepszymi generatorami fitness"],
        ["Najlepsze optymalizatory", "TreeVAE epoch 80 i TransformerVAE epoch 90", "słabsza rekonstrukcja, lepsza eksploracja"],
        ["SLP / projection", "w najnowszych wynikach pogarszało deep search", "projekcja za bardzo zawęża przestrzeń"],
    ], 0.75, 1.65, 11.8, 2.45, [2.35, 4.75, 4.7])
    add_card(slide, 0.9, 4.55, 5.65, 1.35, "Wniosek praktyczny", "Model do optymalizacji nie musi być najlepszym autoenkoderem rekonstrukcyjnym. Liczy się geometria okolic dekodera i zdolność do generowania użytecznych nowości.", AMBER)
    add_card(slide, 6.9, 4.55, 5.1, 1.35, "Decyzja", "Dalsze eksperymenty skupiły się na generatorach 09/10 i na operatorach bez SLP, bo dawały najwyższy zwrot z ewaluacji.", CYAN)
    notes.append(SlideText("Rekonstrukcja nie wystarcza", "Jednym z ważnych wniosków negatywnych jest to, że nie wystarczy wybrać modelu z najlepszą rekonstrukcją. Warianty conditional i structural poprawiały metryki odtwarzania, ale nie dawały najlepszej optymalizacji. Najlepsze wyniki fitness pochodziły z wcześniejszych modeli generatorowych: TreeVAE i TransformerVAE. To zmieniło kierunek eksperymentów."))

    # 7
    slide = prs.slides.add_slide(blank)
    add_bg(slide, AMBER)
    add_title(slide, "Porównanie: standardowe operatory vs przestrzeń ukryta", "wyniki", AMBER)
    add_picture(slide, charts["compare"], 0.65, 1.38, 8.4, 4.55)
    stats = summary["compare_stats"]
    add_metric(slide, 9.45, 1.55, 2.25, "5 : 5", "wygrane vs frams", CYAN)
    add_metric(slide, 9.45, 2.95, 2.25, fmt(stats["latent"]["mean"], 3), "średnia przestrzeni ukrytej", CYAN)
    add_metric(slide, 9.45, 4.35, 2.25, fmt(stats["frams"]["mean"], 3), "średnia frams", AMBER)
    notes.append(SlideText("Porównanie operatorów", "Ten slajd jest najważniejszym uczciwym benchmarkiem. Dla dziesięciu par: dwa modele razy pięć seedów, porównałem standardowe Framsticks mutate/crossOver z operatorami w przestrzeni ukrytej. Budżet to 5000 prawdziwych ewaluacji na parę. Wynik jest zbalansowany: przestrzeń ukryta wygrała pięć razy i Framsticks wygrał pięć razy. Średnia dla przestrzeni ukrytej jest minimalnie wyższa, ale najlepszy pojedynczy wynik w tym benchmarku ma Framsticks. To oznacza, że metoda jest konkurencyjna, ale nie dominuje bezwarunkowo."))

    # 8
    slide = prs.slides.add_slide(blank)
    add_bg(slide, CYAN)
    add_title(slide, "Głębsze uruchomienia w przestrzeni ukrytej", "wyniki", CYAN)
    add_picture(slide, charts["deep"], 0.72, 1.35, 8.6, 4.7)
    best_trans = max(summary["transformer"], key=lambda row: f(row["best_score"]))
    best_tree = max(summary["tree"], key=lambda row: f(row["best_score"]))
    add_metric(slide, 9.6, 1.65, 2.15, fmt(f(best_trans["best_score"]), 3), "TransformerVAE max", CYAN)
    add_metric(slide, 9.6, 3.05, 2.15, fmt(f(best_tree["best_score"]), 3), "TreeVAE max", VIOLET)
    add_metric(slide, 9.6, 4.45, 2.15, "20k", "ewaluacji na seed", AMBER)
    notes.append(SlideText("Głębsze uruchomienia w przestrzeni ukrytej", "Po krótkim benchmarku uruchomiłem głębsze warianty używające tylko operatorów w przestrzeni ukrytej. TransformerVAE osiągnął maksimum około 1.965, a TreeVAE około 1.961. Co ważne, te wyniki są novel, czyli nie są tylko odtworzeniem przykładów z datasetu. Widać jednak nadal zależność od seeda: nie każdy start daje równie dobry wynik."))

    # 9
    slide = prs.slides.add_slide(blank)
    add_bg(slide, VIOLET)
    add_title(slide, "Ewolucja w przestrzeni ukrytej bez SLP", "wyniki", VIOLET)
    evo_top = sorted(summary["evolution"], key=lambda row: f(row["best_score"]), reverse=True)[:5]
    table_rows = [["Model", "Seed", "Best", "Δ vs seed", "Ewaluacje"]]
    for row in evo_top:
        model = "TreeVAE" if row["label"].startswith("09") else "TransformerVAE"
        table_rows.append([
            model,
            row["seed_rank"],
            fmt(f(row["best_score"]), 3),
            fmt(f(row["best_minus_seed_source"]), 3),
            row["evaluated_unique"],
        ])
    add_table(slide, table_rows, 0.85, 1.55, 7.75, 3.0, [2.1, 0.9, 1.1, 1.2, 1.65])
    add_card(slide, 9.0, 1.7, 3.15, 1.2, "Najlepszy wynik", "1.914 dla TreeVAE epoch 80, seed 2", CYAN)
    add_card(slide, 9.0, 3.05, 3.15, 1.2, "Dlaczego bez SLP?", "Projekcja decode-encode poprawia porządek, ale w tych eksperymentach zawężała eksplorację i obniżała maksimum.", AMBER)
    add_card(slide, 9.0, 4.4, 3.15, 1.2, "Znaczenie", "No-SLP evolution pozostaje dobrym trybem do szukania rekordowych osobników.", VIOLET)
    notes.append(SlideText("Ewolucja w przestrzeni ukrytej bez SLP", "Oddzielnie testowałem ewolucję w przestrzeni ukrytej bez SLP. To jest mniej elegancki, ale praktycznie bardzo skuteczny tryb. Najlepszy wynik to około 1.914 dla TreeVAE. SLP, czyli projekcja decode-encode, okazała się w tej fazie niekorzystna, bo ograniczała eksplorację i pogarszała najlepsze wyniki."))

    # 10
    slide = prs.slides.add_slide(blank)
    add_bg(slide, GREEN)
    add_title(slide, "Jak rośnie wynik w czasie", "przebiegi", GREEN)
    add_picture(slide, charts["convergence"], 0.7, 1.35, 8.85, 4.65)
    add_card(slide, 9.85, 1.6, 2.65, 1.2, "Szybkie zyski", "W części seedów przestrzeń ukryta daje duży skok już na początku przebiegu.", CYAN)
    add_card(slide, 9.85, 3.0, 2.65, 1.2, "Plateau", "Po skoku pojawia się długi okres stabilizacji, więc potrzebna jest dywersyfikacja operatorów.", AMBER)
    add_card(slide, 9.85, 4.4, 2.65, 1.2, "Budżet", "Głębsze runy są opłacalne, ale wymagają kontroli liczby ewaluacji.", GREEN)
    notes.append(SlideText("Przebiegi", "Na przebiegach widać typowy charakter tych eksperymentów. Czasem metoda oparta na przestrzeni ukrytej szybko znajduje bardzo dobry skok, a potem przez długi czas stoi na plateau. To sugeruje, że problemem nie jest tylko lokalna jakość operatora, ale też dywersyfikacja populacji i unikanie przedwczesnego skupienia wokół jednej niszy."))

    # 11
    slide = prs.slides.add_slide(blank)
    add_bg(slide, AMBER)
    add_title(slide, "Co zrobiono dodatkowo", "rozszerzenia", AMBER)
    add_bullets(slide, [
        "EA z operatorami w przestrzeni ukrytej: osobny algorytm populacyjny, w którym mutacja i krzyżowanie są wykonywane na wektorach z.",
        "Porównanie matched-seed: te same seedy, ten sam limit fitness startowego i ten sam budżet prawdziwych ewaluacji.",
        "Modele fitness-aware i conditional oraz trening na podzbiorach high-fitness.",
        "Baselines pomocnicze: surrogate-guided mutation, grammar CEM oraz random/mutation checks.",
        "Automatyczny eksport wyników: CSV, history, details, Slurm logs, snapshot Git i sacct.",
    ], 0.9, 1.55, 11.8, 4.6, 17, TEXT)
    notes.append(SlideText("Co zrobiono dodatkowo", "W porównaniu z planem doszło kilka rozszerzeń. Najważniejsze to EA z operatorami w przestrzeni ukrytej, czyli wariant dokładnie odpowiadający pytaniu, czy da się zastąpić standardowe operatory operacjami na wektorach z. Dodałem też wiele elementów infrastrukturalnych, bo przy tej liczbie eksperymentów ręczne porównywanie wyników byłoby niewiarygodne."))

    # 12
    slide = prs.slides.add_slide(blank)
    add_bg(slide, RED)
    add_title(slide, "Ograniczenia i wyniki negatywne", "uczciwa ocena", RED)
    add_card(slide, 0.85, 1.55, 3.65, 3.8, "Stabilność", "Przestrzeń ukryta wygrała 5/10 par w benchmarku. Są mocne przypadki, ale metoda nie jest jeszcze niezawodna dla każdego seeda.", RED)
    add_card(slide, 4.75, 1.55, 3.65, 3.8, "Model", "Najlepszy rekonstruktor nie był najlepszym optymalizatorem. Trzeba wybierać model pod generowanie, nie tylko pod ValExact.", AMBER)
    add_card(slide, 8.65, 1.55, 3.65, 3.8, "Koszt", "Najlepsze wyniki wymagają wielu prawdziwych ewaluacji Framsticks, więc pełna statystyka jest kosztowna obliczeniowo.", CYAN)
    notes.append(SlideText("Ograniczenia", "Najważniejsze ograniczenie to stabilność. Mam wyniki pokazujące, że przestrzeń ukryta potrafi wygrać ze standardowymi operatorami, ale nie wygrywa zawsze. Druga rzecz to różnica między rekonstrukcją a optymalizacją. Trzecia to koszt ewaluacji. Dlatego obecne wnioski są pozytywne, ale warunkowe: metoda jest obiecująca, nie jest jeszcze rozwiązaniem dominującym."))

    # 13
    slide = prs.slides.add_slide(blank)
    add_bg(slide, CYAN)
    add_title(slide, "Podsumowanie i wnioski", "wnioski", CYAN)
    add_metric(slide, 0.85, 1.55, 2.35, "1.965", "najlepszy wynik przestrzeni ukrytej", CYAN)
    add_metric(slide, 3.55, 1.55, 2.35, "5 / 10", "wygrane w benchmarku", AMBER)
    add_metric(slide, 6.25, 1.55, 2.35, "+0.765", "największa poprawa vs seed", GREEN)
    add_metric(slide, 8.95, 1.55, 2.35, "novel", "najlepsze poza datasetem", VIOLET)
    add_bullets(slide, [
        "Zastąpienie operatorów Framsticks operacjami w przestrzeni ukrytej jest możliwe w sensie eksperymentalnym: istnieją seedy, gdzie ta metoda wyraźnie wygrywa.",
        "Wyniki są konkurencyjne z klasycznym EA, ale wymagają doboru modelu, operatorów i budżetu.",
        "Najlepszy obecny kierunek to TransformerVAE / TreeVAE z operatorami w przestrzeni ukrytej bez SLP oraz kontrolowaną dywersyfikacją.",
        "Negatywny wniosek jest równie ważny: lepsza rekonstrukcja nie oznacza automatycznie lepszej optymalizacji.",
    ], 0.95, 3.15, 11.6, 2.55, 15, TEXT)
    notes.append(SlideText("Podsumowanie", "Podsumowując: udało się zbudować system, który faktycznie zastępuje standardowe operatory operacjami w przestrzeni ukrytej i w części przypadków daje lepsze wyniki. Najlepszy wynik dla tej gałęzi to około 1.965, a najlepsze osobniki są nowe względem datasetu. Jednocześnie metoda nie jest jeszcze stabilna we wszystkich seedach, więc wniosek jest pozytywny, ale nie bezwarunkowy."))

    # 14
    slide = prs.slides.add_slide(blank)
    add_bg(slide, VIOLET)
    add_title(slide, "Plany rozwojowe", "następne kroki", VIOLET)
    add_bullets(slide, [
        "Pełniejsza statystyka: więcej seedów i powtórzeń dla najlepszego protokołu operatorów w przestrzeni ukrytej.",
        "Adaptacyjne operatory: automatyczny dobór skali mutacji, krzyżowania i imigrantów losowych zależnie od plateau.",
        "Retraining na elitach: ponowne uczenie lub fine-tuning autoenkodera na genotypach znalezionych przez search.",
        "Miary różnorodności i novelty, żeby unikać szybkiego zamykania populacji w jednej niszy.",
        "Porównanie końcowe z CoSo i standardowym Framsticks EA w jednolitym protokole raportowania.",
    ], 0.95, 1.55, 11.7, 4.6, 17, TEXT)
    notes.append(SlideText("Plany", "Najbliższy plan to przejść z pojedynczych mocnych eksperymentów do pełniejszej statystyki. Chcę też dodać adaptacyjne operatory, bo przebiegi pokazują plateau. Drugim kierunkiem jest retraining na elitach: skoro search znajduje dobre osobniki poza datasetem, warto użyć ich do poprawienia samej reprezentacji. Na końcu potrzebne będzie wspólne porównanie z CoSo i klasycznym EA."))

    # 15
    slide = prs.slides.add_slide(blank)
    add_bg(slide, AMBER)
    add_title(slide, "Referencje", "źródła", AMBER)
    refs = [
        "Kingma, D. P., Welling, M. Auto-Encoding Variational Bayes. ICLR 2014.",
        "Kusner, M. J., Paige, B., Hernández-Lobato, J. M. Grammar Variational Autoencoder. arXiv:1703.01925.",
        "Kaszuba, P., Komosiński, M., Mensfelt, A. Automated Development of Latent Representations for Optimization of Sequences Using Autoencoders. IEEE CEC 2021.",
        "Liskowski, P., Krawiec, K., Toklu, N. E., Swan, J. Program Synthesis as Latent Continuous Optimization. GECCO 2020.",
        "Hansen, N. The CMA Evolution Strategy: A Tutorial. arXiv:1604.00772.",
        "Framsticks documentation and FramsticksLib API, criterion: vertpos.",
    ]
    add_bullets(slide, refs, 0.95, 1.45, 11.6, 4.8, 14, TEXT)
    add_textbox(slide, "Dziękuję za uwagę", 0.95, 6.35, 4.0, 0.4, 22, CYAN, True)
    notes.append(SlideText("Referencje", "Na końcu podaję podstawowe źródła: VAE, grammar VAE, wcześniejsze prace o optymalizacji sekwencji w przestrzeniach ukrytych, CMA-ES oraz dokumentację Framsticks. Dziękuję za uwagę."))

    prs.save(PPTX_PATH)
    return notes


def write_notes(notes: list[SlideText]) -> None:
    lines = ["# Wystąpienie nr 2 - tekst do slajdów", ""]
    for idx, slide in enumerate(notes, start=1):
        lines.append(f"## Slajd {idx}. {slide.title}")
        lines.append("")
        lines.append(slide.speaker)
        lines.append("")
    NOTES_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    summary = compute_summary()
    charts = {
        "compare": plot_compare(summary),
        "deep": plot_deep(summary),
        "convergence": plot_convergence(),
    }
    notes = create_presentation(summary, charts)
    write_notes(notes)
    print(f"Wrote: {PPTX_PATH}")
    print(f"Wrote: {NOTES_PATH}")
    print(f"Wrote assets: {ASSET_DIR}")


if __name__ == "__main__":
    main()
