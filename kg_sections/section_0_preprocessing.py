import re
import unicodedata


DEFAULT_TEXT_COLUMNS = ["title"]
DEFAULT_NORMALIZED_SUFFIX = "_normalized"

SEPARATOR_RE = re.compile(r"[\u2010-\u2015_\-]+")
REPEATED_PUNCT_RE = re.compile(r"([.,;:/+%])\1+")
SPACE_RE = re.compile(r"\s+")

UNIT_RULES = [
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(inches|inch|in\.)\b"), r"\1 in"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(feet|foot|ft\.)\b"), r"\1 ft"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(pounds|pound|lbs?|lb\.)\b"), r"\1 lb"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(ounces|ounce|oz\.)\b"), r"\1 oz"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(watts|watt)\b"), r"\1 w"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(volts|volt)\b"), r"\1 v"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(kilograms|kilogram|kgs?|kg\.)\b"), r"\1 kg"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(grams|gram|g\.)\b"), r"\1 g"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(millimeters|millimeter|mms?|mm\.)\b"), r"\1 mm"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(centimeters|centimeter|cms?|cm\.)\b"), r"\1 cm"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(meters|meter|m\.)\b"), r"\1 m"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(gigabytes|gigabyte|gbs?|gb\.)\b"), r"\1 gb"),
    (re.compile(r"\b(\d+(?:\.\d+)?)\s*(megabytes|megabyte|mbs?|mb\.)\b"), r"\1 mb"),
]

TEXT_REPLACEMENTS = {
    "\u00a0": " ",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "&": " and ",
}


def strip_accents(text):
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def normalize_text(value):
    if value is None or value != value:
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    for source, target in TEXT_REPLACEMENTS.items():
        text = text.replace(source, target)

    text = strip_accents(text)
    text = text.lower()
    text = SEPARATOR_RE.sub(" ", text)
    text = re.sub(r"\s*/\s*", " / ", text)
    text = re.sub(r"\s*\+\s*", " + ", text)
    text = REPEATED_PUNCT_RE.sub(r"\1", text)

    for pattern, replacement in UNIT_RULES:
        text = pattern.sub(replacement, text)

    text = re.sub(r"[^\w\s./+%]", " ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def preprocess_products(df, preprocessing_config=None, llm_columns=None):
    config = preprocessing_config or {}
    if not config.get("enabled", True):
        return df

    text_columns = config.get("text_columns") or llm_columns or DEFAULT_TEXT_COLUMNS
    normalized_suffix = config.get("normalized_suffix", DEFAULT_NORMALIZED_SUFFIX)
    keep_raw_columns = config.get("keep_raw_columns", True)

    prepared = df.copy()
    for column in text_columns:
        if column not in prepared.columns:
            continue

        if keep_raw_columns:
            raw_column = f"{column}_raw"
            if raw_column not in prepared.columns:
                prepared[raw_column] = prepared[column]

        prepared[f"{column}{normalized_suffix}"] = prepared[column].map(normalize_text)

    return prepared


def normalized_column_name(column, preprocessing_config=None):
    config = preprocessing_config or {}
    suffix = config.get("normalized_suffix", DEFAULT_NORMALIZED_SUFFIX)
    return f"{column}{suffix}"
