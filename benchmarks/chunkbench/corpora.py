"""Workloads: deterministic synthetic corpora, optional real corpora, and gold excerpt spans.

Synthetic corpora are generated from a seeded PRNG, so everyone benchmarks identical bytes.
Real corpora are downloaded once with `python -m chunkbench fetch` into `benchmarks/corpora/`
and pinned by SHA-256 in `corpora.lock.json`.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CORPORA_DIR = HERE / "corpora"
LOCK_FILE = HERE / "corpora.lock.json"

MIB = 1024 * 1024


@dataclass
class Workload:
    name: str
    fmt: str  # "plain" | "markdown"
    text: str
    kind: str  # "synthetic" | "real" | "stress"
    gold: list[tuple[int, int]] = field(default_factory=list)  # excerpt spans (char offsets)
    gold_source: str = "sampled"  # "sampled" (seeded sentence spans) or "dataset"

    @property
    def size_bytes(self) -> int:
        return len(self.text.encode("utf-8"))


# --------------------------------------------------------------------------- synthetic text

_COMMON = (
    "the of and to in is that it was for on are as with his they at be this from have or by one "
    "had not but what all were when we there can an your which their said if do will each about "
    "how up out them then she many some so these would other into has more her two like him see "
    "time could no make than first been its who now people my made over did down only way find "
    "use may water long little very after words called just where most know get through back "
    "much before go good new write our used me man too any day same right look think also around "
    "another came come work three word must because does part even place well such here take why "
    "things help put years different away again off went old number great tell men say small "
    "every found still between name should home big give air line set own under read last never "
    "us left end along while might next sound below saw something thought both few those always "
    "looked show large often together asked house don't world going want school important until "
    "form food keep children feet land side without boy once animals life enough took sometimes "
    "four head above kind began almost live page got earth need far hand high year mother light "
    "country father let night picture being study second soon story since white ever paper hard "
    "near sentence better best across during today however sure knew try told young sun thing "
    "whole hear example heard several change answer room sea against top turned learn point city "
    "play toward five using himself usually system model data process memory token retrieval "
    "document context search query index vector network library function value result error"
).split()

_SYLLABLES = "ka lo mi ra ten sa vor el un di pre tion ing ou ex ab con der min po ver tra".split()


def _vocab(rnd: random.Random) -> tuple[list[str], list[float]]:
    words = list(_COMMON)
    # A long tail of invented words keeps the text from being unrealistically repetitive.
    for _ in range(3000):
        words.append("".join(rnd.choice(_SYLLABLES) for _ in range(rnd.randrange(1, 5))))
    weights = [1.0 / (i + 3) for i in range(len(words))]
    return words, weights


class _Gen:
    def __init__(self, seed: int) -> None:
        self.r = random.Random(seed)
        self.words, self.weights = _vocab(self.r)

    def sentence(self) -> str:
        n = self.r.randrange(6, 24)
        ws = self.r.choices(self.words, self.weights, k=n)
        if n > 12 and self.r.random() < 0.6:
            ws[self.r.randrange(4, n - 3)] += ","
        ws[0] = ws[0].capitalize()
        return " ".join(ws) + self.r.choice([".", ".", ".", ".", "?", "!"])

    def paragraph(self, lo: int = 2, hi: int = 7) -> str:
        return " ".join(self.sentence() for _ in range(self.r.randrange(lo, hi)))

    def code(self) -> str:
        r = self.r
        lang = r.choice(["python", "javascript", "bash"])
        lines = []
        if lang == "python":
            lines.append(f"def {r.choice(self.words)}_{r.randrange(100)}(x, y={r.randrange(10)}):")
            for _ in range(r.randrange(3, 14)):
                lines.append(f"    {r.choice(self.words)} = x * {r.randrange(1000)} + y  # {r.choice(self.words)}")
            lines.append("    return x")
        elif lang == "javascript":
            lines.append(f"function {r.choice(self.words)}{r.randrange(100)}(a, b) {{")
            for _ in range(r.randrange(3, 14)):
                lines.append(f"  const {r.choice(self.words)} = a + {r.randrange(1000)} - b;")
            lines.append("  return a;\n}")
        else:
            for _ in range(r.randrange(2, 8)):
                lines.append(f"$ {r.choice(self.words)} --{r.choice(self.words)}={r.randrange(100)} ./{r.choice(self.words)}.txt")
        return f"```{lang}\n" + "\n".join(lines) + "\n```"

    def markdown_block(self) -> str:
        r = self.r
        k = r.randrange(20)
        if k < 3:
            return "#" * r.randrange(1, 4) + " " + self.sentence().rstrip(".?!")[:70]
        if k < 10:
            return self.paragraph()
        if k < 13:
            return "\n".join(f"- {self.sentence()}" for _ in range(r.randrange(2, 7)))
        if k < 15:
            return "\n".join(f"{i}. {self.sentence()}" for i in range(1, r.randrange(3, 8)))
        if k < 17:
            return self.code()
        if k < 18:
            head = "| name | value | note |\n| --- | --- | --- |"
            rows = [f"| {r.choice(self.words)} | {r.randrange(10000)} | {r.choice(self.words)} |" for _ in range(r.randrange(3, 10))]
            return head + "\n" + "\n".join(rows)
        if k < 19:
            return "> " + self.paragraph(1, 4).replace(". ", ".\n> ")
        return f"See [{r.choice(self.words)}](https://example.com/{r.choice(self.words)}) and `{r.choice(self.words)}()` for **{r.choice(self.words)}** details. " + self.sentence()


def _fill(gen_piece, target_bytes: int, sep: str = "\n\n") -> str:
    parts: list[str] = []
    size = 0
    while size < target_bytes:
        piece = gen_piece()
        parts.append(piece)
        size += len(piece.encode("utf-8")) + len(sep)
    return sep.join(parts) + "\n"


# Sentences in several scripts (written for this benchmark).
_MULTI = {
    "en": ["Reliable retrieval starts with chunks that respect the structure of a document.", "Every request is split into windows that fit the model's context.", "Nothing is lost when the text is cut at a sensible place."],
    "de": ["Die Straße war nach dem Regen noch nass, und die Größe der Pfützen überraschte alle.", "Zuverlässige Suche beginnt mit sinnvoll geschnittenen Textabschnitten.", "Über den Wolken muss die Freiheit wohl grenzenlos sein."],
    "fr": ["La cohérence d'un texte dépend de la façon dont on le découpe en morceaux.", "Les élèves ont préparé un déjeuner à l'école avant l'été.", "Où êtes-vous allés après la réunion ?"],
    "es": ["El niño pequeño miró la montaña mientras el viento soplaba con fuerza.", "¿Cómo se divide un documento largo sin perder el significado?", "La información se organiza en fragmentos más pequeños."],
    "pl": ["Zażółć gęślą jaźń to zdanie zawiera wszystkie polskie litery.", "Dokument dzielimy na fragmenty, które mieszczą się w limicie."],
    "ru": ["Надёжный поиск начинается с разумного разделения текста на фрагменты.", "Каждый фрагмент должен помещаться в контекстное окно модели.", "Съешь же ещё этих мягких французских булок, да выпей чаю."],
    "el": ["Η αξιόπιστη αναζήτηση ξεκινά από σωστά τεμαχισμένα κείμενα.", "Καλημέρα κόσμε, σήμερα είναι μια όμορφη μέρα."],
    "ar": ["يبدأ البحث الموثوق بتقسيم النص إلى أجزاء ذات معنى.", "كل جزء يجب أن يتناسب مع نافذة السياق الخاصة بالنموذج."],
    "hi": ["विश्वसनीय खोज की शुरुआत पाठ को सही जगह से बाँटने से होती है।", "हर खंड मॉडल की संदर्भ सीमा में आना चाहिए।"],
    "zh": ["可靠的检索始于合理地切分文本。", "每个片段都必须适合模型的上下文窗口。", "在合适的位置切分不会丢失任何信息。"],
    "ja": ["信頼できる検索は、文章を適切に分割することから始まります。", "それぞれの断片はモデルの文脈に収まる必要があります。"],
    "ko": ["신뢰할 수 있는 검색은 문서를 적절하게 나누는 것에서 시작됩니다.", "각 조각은 모델의 컨텍스트 창에 들어가야 합니다."],
    "emoji": ["Deploy went well 🚀 and the team 👩🏽‍💻👨‍💻 celebrated 🎉 with ☕.", "Flags 🇩🇰🇯🇵🇧🇷 and families 👨‍👩‍👧‍👦 are single grapheme clusters."],
}
_NO_SPACE = {"zh", "ja"}


def synthetic_multilingual(target_bytes: int, seed: int) -> str:
    r = random.Random(seed)
    langs = list(_MULTI)

    def piece() -> str:
        lang = r.choice(langs)
        sents = [r.choice(_MULTI[lang]) for _ in range(r.randrange(2, 7))]
        return ("" if lang in _NO_SPACE else " ").join(sents)

    return _fill(piece, target_bytes)


def synthetic_prose(target_bytes: int, seed: int) -> str:
    g = _Gen(seed)
    return _fill(g.paragraph, target_bytes)


def synthetic_markdown(target_bytes: int, seed: int) -> str:
    g = _Gen(seed)
    return _fill(g.markdown_block, target_bytes)


def synthetic_code_heavy(target_bytes: int, seed: int) -> str:
    g = _Gen(seed)

    def piece() -> str:
        return g.code() if g.r.random() < 0.75 else g.paragraph(1, 3)

    return _fill(piece, target_bytes)


def stress_single_line(target_bytes: int, seed: int) -> str:
    return synthetic_prose(target_bytes, seed).replace("\n\n", " ").replace("\n", " ")


def stress_giant_list(target_bytes: int, seed: int) -> str:
    g = _Gen(seed)
    lines, size, i = [], 0, 0
    while size < target_bytes:
        i += 1
        line = f"{i}. {g.sentence()}"
        lines.append(line)
        size += len(line) + 1
    return "\n".join(lines) + "\n"


def stress_cjk_unbroken(target_bytes: int, seed: int) -> str:
    r = random.Random(seed)
    pool = [s.rstrip("。") for s in _MULTI["zh"] + _MULTI["ja"]]
    out, size = [], 0
    while size < target_bytes:
        s = r.choice(pool)
        out.append(s)
        size += len(s.encode("utf-8"))
    return "".join(out)


# (name, fmt, kind, generator, seed)
SYNTHETIC = [
    ("prose_en", "plain", "synthetic", synthetic_prose, 1),
    ("markdown_doc", "markdown", "synthetic", synthetic_markdown, 2),
    ("multilingual", "plain", "synthetic", synthetic_multilingual, 3),
    ("code_heavy", "markdown", "synthetic", synthetic_code_heavy, 4),
    ("stress_single_line", "plain", "stress", stress_single_line, 5),
    ("stress_giant_list", "markdown", "stress", stress_giant_list, 6),
    ("stress_cjk_unbroken", "plain", "stress", stress_cjk_unbroken, 7),
]


# --------------------------------------------------------------------------- gold spans

_SENT_END = re.compile(r"(?<=[.!?。！？])\s+|\n{2,}")


def sample_spans(text: str, n: int, seed: int) -> list[tuple[int, int]]:
    """Seeded excerpt spans of 1-3 consecutive sentences (40-700 chars), for an offline,
    dataset-independent measurement of how well chunk boundaries align with answer spans."""
    bounds = [0]
    for m in _SENT_END.finditer(text):
        bounds.append(m.end())
    bounds.append(len(text))
    if len(bounds) < 8:
        return []
    r = random.Random(seed)
    spans: list[tuple[int, int]] = []
    attempts = 0
    while len(spans) < n and attempts < n * 20:
        attempts += 1
        i = r.randrange(0, len(bounds) - 4)
        j = i + r.randrange(1, 4)
        s, e = bounds[i], bounds[j]
        piece = text[s:e].rstrip()
        e = s + len(piece)
        if 40 <= e - s <= 700:
            spans.append((s, e))
    return spans


# --------------------------------------------------------------------------- real corpora

CHROMA_BASE = (
    "https://raw.githubusercontent.com/brandonstarxel/chunking_evaluation/{ref}/"
    "chunking_evaluation/evaluation_framework/general_evaluation_data/"
)
CHROMA_FILES = [
    "corpora/wikitexts.md",
    "corpora/state_of_the_union.md",
    "corpora/finance.md",
    "corpora/pubmed.md",
    "corpora/chatlogs.md",
    "questions_df.csv",
]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_corpora(update_lock: bool = False) -> int:
    """Download the public Chroma `chunking_evaluation` corpora (MIT licensed) and pin them."""
    CORPORA_DIR.mkdir(parents=True, exist_ok=True)
    lock = json.loads(LOCK_FILE.read_text()) if LOCK_FILE.exists() else {}
    new_lock: dict[str, dict] = {}
    for rel in CHROMA_FILES:
        dest = CORPORA_DIR / Path(rel).name
        last_err: Exception | None = None
        for ref in ("main", "master"):
            url = CHROMA_BASE.format(ref=ref) + rel
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    data = resp.read()
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        else:
            print(f"error: could not download {rel}: {last_err}", file=sys.stderr)
            return 1
        dest.write_bytes(data)
        digest = _sha256(dest)
        new_lock[dest.name] = {"sha256": digest, "bytes": len(data), "url": url}
        known = lock.get(dest.name, {}).get("sha256")
        status = "new" if not known else ("ok" if known == digest else "MISMATCH")
        print(f"{dest.name:<28} {len(data):>10} bytes  sha256={digest[:12]}…  {status}")
        if status == "MISMATCH" and not update_lock:
            print("error: content differs from corpora.lock.json (use --update-lock to accept)", file=sys.stderr)
            return 2
    if update_lock or not lock:
        LOCK_FILE.write_text(json.dumps(new_lock, indent=2, sort_keys=True) + "\n")
        print(f"wrote {LOCK_FILE.name}")
    return 0


def _load_questions() -> dict[str, list[tuple[int, int]]]:
    path = CORPORA_DIR / "questions_df.csv"
    out: dict[str, list[tuple[int, int]]] = {}
    if not path.exists():
        return out
    csv.field_size_limit(1 << 26)
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                refs = json.loads(row["references"])
            except (KeyError, ValueError):
                continue
            for ref in refs:
                try:
                    out.setdefault(row["corpus_id"], []).append((int(ref["start_index"]), int(ref["end_index"])))
                except (KeyError, ValueError, TypeError):
                    continue
    return out


def real_workloads(max_gold: int, seed: int) -> list[Workload]:
    if not CORPORA_DIR.exists():
        return []
    questions = _load_questions()
    out = []
    for path in sorted(CORPORA_DIR.iterdir()):
        if path.suffix not in {".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        gold = [(s, e) for s, e in questions.get(path.stem, []) if 0 <= s < e <= len(text)]
        source = "dataset"
        if gold:
            random.Random(seed).shuffle(gold)
            gold = gold[:max_gold]
        else:
            gold, source = sample_spans(text, max_gold, seed), "sampled"
        out.append(Workload(f"real_{path.stem}", "markdown" if path.suffix == ".md" else "plain", text, "real", gold, source))
    return out


def build_workloads(size_mib: float = 3.0, max_gold: int = 300, seed: int = 20260101, include_real: bool = True) -> dict[str, Workload]:
    target = int(size_mib * MIB)
    out: dict[str, Workload] = {}
    for name, fmt, kind, gen, wseed in SYNTHETIC:
        size = min(target, 2 * MIB) if name == "stress_cjk_unbroken" else target
        text = gen(size, seed + wseed)
        out[name] = Workload(name, fmt, text, kind, sample_spans(text, max_gold, seed + wseed))
    if include_real:
        for w in real_workloads(max_gold, seed):
            out[w.name] = w
    return out
