"""Attribute extraction from product *names* (real metadata, e.g. 'Women Blue Printed
Straight Kurta with 3/4 Sleeves'). Values come from the text the retailer wrote, so they
are recorded with source='name_keywords'. Missing -> 'unknown'."""
from __future__ import annotations

import re

PATTERN_KEYWORDS = [
    ("floral", r"\bfloral\b|\bflower"), ("striped", r"\bstripe[sd]?\b"),
    ("checked", r"\bcheck(ed|s)?\b|\bplaid\b"), ("polka dot", r"\bpolka\b"),
    ("paisley", r"\bpaisley\b"), ("geometric", r"\bgeometric\b"), ("abstract", r"\babstract\b"),
    ("bandhani", r"\bbandhani|bandhej\b"), ("ikat", r"\bikat\b"), ("leheriya", r"\bleheriya\b"),
    ("ethnic motifs", r"\bethnic motif"), ("colourblocked", r"colou?r[- ]?block"),
    ("embroidered", r"\bembroider"), ("embellished", r"\bembellish|\bsequin"),
    ("self-design", r"\bself[- ]design"), ("woven design", r"\bwoven\b"),
    ("printed", r"\bprint(ed|s)?\b"), ("solid", r"\bsolid\b|\bplain\b"),
]
SLEEVE_KEYWORDS = [
    ("sleeveless", r"\bsleeveless\b"), ("three-quarter", r"3/4|three[- ]quarter"),
    ("full sleeves", r"\b(full|long)[- ]sleeve"), ("short sleeves", r"\bshort[- ]sleeve|\bhalf[- ]sleeve"),
    ("cap sleeves", r"\bcap[- ]sleeve"), ("puff sleeves", r"\bpuff"), ("bell sleeves", r"\bbell[- ]sleeve"),
]
NECKLINE_KEYWORDS = [
    ("v-neck", r"\bv[- ]neck"), ("round neck", r"\bround[- ]neck"), ("mandarin collar", r"\bmandarin"),
    ("boat neck", r"\bboat[- ]neck"), ("square neck", r"\bsquare[- ]neck"), ("high neck", r"\bhigh[- ]neck|\bturtle"),
    ("sweetheart", r"\bsweetheart"), ("halter", r"\bhalter"), ("off-shoulder", r"\boff[- ]shoulder"),
    ("keyhole", r"\bkeyhole"), ("cowl neck", r"\bcowl"), ("shirt collar", r"\bshirt collar|\bspread collar|\bcollar\b"),
]


def _match(text: str, table) -> str:
    t = text.lower()
    for value, rx in table:
        if re.search(rx, t):
            return value
    return "unknown"


def attributes_from_name(name: str) -> dict[str, str]:
    name = name or ""
    pattern = _match(name, PATTERN_KEYWORDS)
    return {
        "pattern": pattern,
        "sleeve_type": _match(name, SLEEVE_KEYWORDS),
        "neckline": _match(name, NECKLINE_KEYWORDS),
        "embroidery": "yes" if re.search(r"\bembroider", name.lower()) else "unknown",
    }


# ---------------------------------------------------------------------------- shopper text parsing
# Only words in these vocabularies can become attributes. Anything else is ignored, so an unknown
# word can never turn into a wrong filter.
TEXT_CATEGORY = {
    "kurtas-kurtis": ["kurti", "kurtis", "kurta", "kurtas", "kurthi", "kurthis", "kurtha", "kurty", "tunic", "anarkali"],
    "sarees": ["saree", "sarees", "sari", "saris", "saaree", "sadi"],
    "lehengas": ["lehenga", "lehengas", "lehnga", "lengha", "lahenga", "ghagra", "chaniya choli", "lehenga choli"],
    "western-dresses": ["dress", "dresses", "gown", "maxi dress", "frock", "jumpsuit", "midi dress"],
    "tops-shirts": ["top", "tops", "blouse", "shirt", "shirts", "t-shirt", "tshirt", "tee", "crop top"],
    "other-ethnic": ["salwar", "churidar", "dupatta", "palazzo", "patiala", "sharara", "salwar suit"],
}
TEXT_PATTERN = {
    "floral": ["floral", "flower", "flowers", "flowery", "flowered"], "striped": ["striped", "stripes", "stripe", "pinstripe"],
    "checked": ["checked", "checks", "check", "checkered", "chequered", "plaid", "gingham", "tartan"],
    "polka dot": ["polka", "polka dot", "polka dots", "dots", "dotted", "spotted"],
    "paisley": ["paisley", "mango motif"], "geometric": ["geometric", "chevron", "zigzag"],
    "abstract": ["abstract"], "ethnic motifs": ["block print", "ethnic print", "ethnic motif", "ethnic motifs", "motifs"],
    "bandhani": ["bandhani", "bandhej", "tie dye", "tie-dye"], "ikat": ["ikat"], "leheriya": ["leheriya", "lehariya"],
    "embroidered": ["embroidered", "embroidery", "chikankari", "chikan", "zari", "thread work", "mirror work", "zardozi"],
    "printed": ["printed", "print", "prints"], "solid": ["solid", "plain"],
}
TEXT_SLEEVE = {
    "sleeveless": ["sleeveless", "no sleeves", "without sleeves", "strapless"],
    "full sleeves": ["full sleeves", "full sleeve", "long sleeves", "long sleeve", "full-sleeved", "long-sleeved", "full hand"],
    "three-quarter": ["3/4 sleeves", "3/4 sleeve", "3/4th sleeves", "three quarter", "three-quarter", "elbow sleeves"],
    "short sleeves": ["short sleeves", "short sleeve", "half sleeves", "half sleeve", "half-sleeved", "short-sleeved"],
    "cap sleeves": ["cap sleeves", "cap sleeve"], "puff sleeves": ["puff sleeves", "puff sleeve", "puffed sleeves"],
    "bell sleeves": ["bell sleeves", "bell sleeve", "flared sleeves"],
}
TEXT_NECK = {
    "v-neck": ["v-neck", "v neck", "vneck"], "round neck": ["round neck", "round-neck", "crew neck"],
    "mandarin collar": ["mandarin collar", "chinese collar", "band collar"], "boat neck": ["boat neck", "boat-neck", "bateau"],
    "square neck": ["square neck", "square-neck"], "high neck": ["high neck", "turtle neck", "turtleneck"],
    "sweetheart": ["sweetheart neck", "sweetheart"], "halter": ["halter", "halter neck"],
    "off-shoulder": ["off shoulder", "off-shoulder"], "keyhole": ["keyhole neck", "keyhole"],
    "shirt collar": ["shirt collar", "collared"], "cowl neck": ["cowl neck", "cowl"],
}
TEXT_LENGTH = {"ankle length": ["ankle length", "floor length", "long dress", "maxi", "full length"],
               "knee length": ["knee length", "knee-length", "midi"], "cropped": ["cropped", "crop"],
               "hip length": ["hip length", "short kurti", "short kurta"]}
TEXT_FABRIC = {"cotton": ["cotton", "cotton-looking", "khadi", "linen"], "silk": ["silk", "silky", "satin", "banarasi"],
               "georgette": ["georgette", "chiffon", "net", "organza", "sheer"], "denim": ["denim", "jeans"],
               "velvet": ["velvet"], "knit": ["knit", "knitted", "jersey", "hosiery"]}
# frequent English words that must never be "typo-corrected" into fashion terms
COMMON_WORDS = {"similar", "design", "designs", "dress", "style", "styles", "something", "should", "colours", "colors",
                "pretty", "beautiful", "outfit", "party", "wedding", "sister", "brother", "mother", "please", "would",
                "looking", "shades", "shade", "darker", "lighter", "other", "these", "those", "there", "where", "which",
                "sleeves", "sleeve", "pattern", "patterns", "fabric", "neckline", "length", "slightly", "little"}
SHADE_WORDS = {"dark": "dark", "darker": "dark", "deep": "deep", "light": "light", "lighter": "light",
               "pastel": "pastel", "bright": None}
NEGATIONS = ("no ", "not ", "without ", "non ", "non-")


def _norm(text: str) -> str:
    t = (text or "").lower().replace("colour", " ").replace("color", " ")
    t = re.sub(r"[^a-z0-9/\- ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _find(t: str, table: dict[str, list[str]]):
    """Longest phrase match first; returns (value, phrase, start) or None."""
    best = None
    for value, phrases in table.items():
        for ph in phrases:
            m = re.search(rf"(?<![a-z]){re.escape(ph)}(?![a-z])", t)
            if m and (best is None or len(ph) > len(best[1])):
                best = (value, ph, m.start())
    return best


def _fuzzy_fix(t: str, vocab: list[str]) -> tuple[str, list[str]]:
    """Correct single-word typos (>= 5 letters) against the vocabulary, e.g. 'flroal' -> 'floral'."""
    import difflib

    fixed, words = [], t.split(" ")
    single = sorted({w for w in vocab if " " not in w and len(w) >= 5})
    # every word that occurs inside any phrase ('sleeves' in 'full sleeves') is already correct
    known = {part for ph in vocab for part in ph.split(" ")} | COMMON_WORDS
    for i, w in enumerate(words):
        if len(w) >= 5 and w not in known:
            # same first letter + high similarity: fixes 'flroal' / 'kurthi' style typos without
            # turning unrelated words into attributes
            m = difflib.get_close_matches(w, [v for v in single if v[0] == w[0]], n=1, cutoff=0.8)
            if m:
                fixed.append(f"{w}->{m[0]}"); words[i] = m[0]
    return " ".join(words), fixed


def parse_text_query(text: str) -> dict:
    """Structured parse of a shopper's description.

    Returns {"attributes": {...}, "matched": [{"attribute","value","phrase"}], "corrections": [...],
    "negated": [...]} . Supported attributes: category, colour, shade, pattern, sleeve_type, neckline,
    length, fabric_appearance, embroidery.
    """
    from app.services.colour import COLOUR_ALIASES, COLOUR_REFERENCE

    tables = {"category": TEXT_CATEGORY, "pattern": TEXT_PATTERN, "sleeve_type": TEXT_SLEEVE,
              "neckline": TEXT_NECK, "length": TEXT_LENGTH, "fabric_appearance": TEXT_FABRIC}
    colour_table: dict[str, list[str]] = {c: [c] for c in COLOUR_REFERENCE}
    for alias, target in COLOUR_ALIASES.items():
        colour_table.setdefault(target, [target]).append(alias)
    vocab = [ph for tb in list(tables.values()) + [colour_table] for phs in tb.values() for ph in phs]
    t, corrections = _fuzzy_fix(_norm(text), vocab)
    attrs: dict[str, str] = {}
    matched, negated = [], []
    for attr, table in list(tables.items()) + [("colour", colour_table)]:
        hit = _find(t, table)
        if not hit:
            continue
        value, phrase, start = hit
        before = t[max(0, start - 9):start]
        if any(before.endswith(n) for n in NEGATIONS):
            negated.append(f"{attr}:{value}")
            if attr == "pattern" and value in ("embroidered",):
                attrs["embroidery"] = "no"
            elif attr == "pattern":
                attrs["pattern"] = "solid"   # "no print" / "without pattern" means plain
            continue
        if value == "multicolour":
            continue
        attrs[attr] = value
        matched.append({"attribute": attr, "value": value, "phrase": phrase})
        if attr == "colour":
            prev = t[:start].split(" ")[-2:] if start else []
            for wd in reversed(prev):
                if wd in SHADE_WORDS and SHADE_WORDS[wd]:
                    attrs["shade"] = SHADE_WORDS[wd]
                    break
    if attrs.get("pattern") == "embroidered":
        attrs["embroidery"] = "yes"
    return {"attributes": attrs, "matched": matched, "corrections": corrections, "negated": negated}


def attributes_from_text_query(text: str) -> dict[str, str]:
    """Backward-compatible: just the attribute dict (e.g. {'colour': 'blue', 'sleeve_type': 'full sleeves'})."""
    return parse_text_query(text)["attributes"]
