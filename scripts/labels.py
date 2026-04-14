import json
import os
import unicodedata
import urllib.error
import urllib.request
from functools import lru_cache

import pandas as pd
import regex as re

try:
    import nltk
    from nltk.corpus import words as nltk_words
except Exception:
    nltk = None
    nltk_words = None

try:
    import pyshorteners
except Exception:
    pyshorteners = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


INPUT_CSV = "files/mar30.csv"
OUTPUT_CSV = "files/mar30_classified.csv"
OUTPUT_REASON_CSV = "files/mar30_reasons.csv"
OUTPUT_AGENT_TEMPLATE_CSV = "files/mar30_agent_templates.csv"
CONFIG_PATH = "scripts/config.json"
CSV_ENCODING = "utf-8-sig"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_TIMEOUT_SECONDS = int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120"))

BRAND_COL = "BRAND_MESSAGE"
CUSTOMER_COL = "CUSTOMER_MESSAGE"

NEW_OUTPUT_COLS = [
    "OPT_OUT",
    "REASON",
    "Template",
    "Detected Direct Opt-Out",
    "Detected Direct Opt-Out Trigger",
    "Detected Keyword",
    "Detected Keyword Trigger",
    "Detected Under 13",
    "Detected Under 13 Trigger",
    "Detected Offensive",
    "Detected Offensive Trigger",
]

DIRECT_REASON_NAMES = {
    "symbols",
    "phrases",
    "wrong_number",
    "disengagement",
    "block",
    "journeys",
    "opt_out_device_not_working",
}

NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")
WHITESPACE_RE = re.compile(r"\s+")
REPEATED_CHAR_RE = re.compile(r"(.)\1+")

URL_RE = re.compile(
    r"""(?ix)
    \b(
        (?:https?://|www\.)
        [^\s]+
      |
        [a-z0-9.-]+\.[a-z]{2,}(?:/[^\s]*)?
    )
    """
)

DATE_RE = re.compile(
    r"""(?ix)
    \b(
        (?:\d{1,4}[/-]\d{1,2}[/-]\d{1,4})
      |
        (?:
            jan|january|feb|february|mar|march|apr|april|may|jun|june|
            jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december
        )
        \s+
        \d{1,2}(?:st|nd|rd|th)?
        (?:,?\s+\d{2,4})?
      |
        \d{1,2}
        \s+
        (?:
            jan|january|feb|february|mar|march|apr|april|may|jun|june|
            jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december
        )
        (?:\s+\d{2,4})?
    )\b
    """
)

NUMBER_RE = re.compile(
    r"""(?ix)
    \b(
        \d{1,3}(?:,\d{3})+(?:\.\d+)?
      |
        \d+\.\d+
      |
        \d+(?:st|nd|rd|th)?
      |
        \d+
    )\b
    """
)


if pyshorteners is not None:
    try:
        DAGD_SHORTENER = pyshorteners.Shortener(timeout=5)
    except Exception:
        DAGD_SHORTENER = None
else:
    DAGD_SHORTENER = None


@lru_cache(maxsize=200000)
def strip_diacritics_and_compat(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


@lru_cache(maxsize=200000)
def normalize_text_cached(text: str) -> str:
    s = strip_diacritics_and_compat(text).lower()
    s = re.sub(r"[\'’`]", "", s)

    s = URL_RE.sub(" <url> ", s)
    s = DATE_RE.sub(" <date> ", s)
    s = NUMBER_RE.sub(" <number> ", s)

    s = re.sub(r"<url>", " URLTOKEN ", s)
    s = re.sub(r"<date>", " DATETOKEN ", s)
    s = re.sub(r"<number>", " NUMBERTOKEN ", s)

    s = NON_ALNUM_RE.sub(" ", s)
    s = WHITESPACE_RE.sub(" ", s).strip()

    s = s.replace("URLTOKEN", "<url>")
    s = s.replace("DATETOKEN", "<date>")
    s = s.replace("NUMBERTOKEN", "<number>")

    return s


@lru_cache(maxsize=200000)
def normalize_keyword_text_cached(text: str) -> str:
    s = strip_diacritics_and_compat(text).lower()
    s = re.sub(r"[\'’`]", "", s)
    s = NON_ALNUM_RE.sub(" ", s)
    s = WHITESPACE_RE.sub(" ", s).strip()
    return s


@lru_cache(maxsize=50000)
def collapse_repeated_chars(token: str) -> str:
    return REPEATED_CHAR_RE.sub(r"\1", token)


@lru_cache(maxsize=500000)
def levenshtein_distance_leq_one(a: str, b: str) -> bool:
    if a == b:
        return True

    la = len(a)
    lb = len(b)
    if abs(la - lb) > 1:
        return False

    if la == lb:
        mismatches = 0
        for ca, cb in zip(a, b):
            if ca != cb:
                mismatches += 1
                if mismatches > 1:
                    return False
        return mismatches == 1

    if la > lb:
        a, b = b, a
        la, lb = lb, la

    i = 0
    j = 0
    edits = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            edits += 1
            if edits > 1:
                return False
            j += 1

    if j < lb or i < la:
        edits += 1

    return edits == 1


NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
}


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_config_section(config: dict, section_name: str) -> dict:
    section = config.get(section_name, {})
    if isinstance(section, dict):
        return section
    return {}


def load_english_words() -> set:
    if nltk_words is None:
        return set()

    try:
        return {w.lower() for w in nltk_words.words()}
    except LookupError:
        if nltk is None:
            return set()
        try:
            nltk.download("words", quiet=True)
            return {w.lower() for w in nltk_words.words()}
        except Exception:
            return set()
    except Exception:
        return set()


def normalize_text(text) -> str:
    if text is None or pd.isna(text):
        return ""
    return normalize_text_cached(str(text))


def normalize_keyword_text(text) -> str:
    if text is None or pd.isna(text):
        return ""
    return normalize_keyword_text_cached(str(text))


def compile_normalized_set(values):
    compiled = set()
    for value in values or []:
        normalized = normalize_text(value)
        if normalized:
            compiled.add(normalized)
    return compiled


def compile_phrase_list(values, *, normalizer):
    phrases = []
    seen = set()
    for value in values or []:
        normalized = normalizer(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            phrases.append(normalized)
    phrases.sort(key=len, reverse=True)
    return phrases


def compile_pattern_list(patterns):
    compiled = []
    for pattern in patterns or []:
        if not pattern:
            continue
        rx = re.compile(pattern, re.IGNORECASE)
        if rx.search("") is not None:
            continue
        compiled.append(rx)
    return compiled


def build_stop_regex(stop_emojis):
    if not stop_emojis:
        return None
    return re.compile(
        "(" + "|".join(re.escape(e) for e in sorted(stop_emojis, key=len, reverse=True)) + ")"
    )


def compile_journeys(journeys_config: dict) -> dict:
    compiled = {}

    for journey_name, journey_data in (journeys_config or {}).items():
        brand_message_norm = normalize_text(journey_data.get("brand_message"))
        raw_responses = journey_data.get("opt_out_responses", [])
        opt_out_responses_norm = compile_normalized_set(raw_responses)
        opt_out_numeric_tokens = set()
        for value in raw_responses:
            if value is None:
                continue
            value_norm = normalize_keyword_text(value)
            if value_norm and value_norm.isdigit():
                opt_out_numeric_tokens.add(value_norm)

        compiled[journey_name] = {
            "brand_message": brand_message_norm,
            "opt_out_responses": opt_out_responses_norm,
            "opt_out_responses_phrases": sorted(opt_out_responses_norm, key=len, reverse=True),
            "opt_out_numeric_tokens": opt_out_numeric_tokens,
        }

    return compiled


def compile_under_13_rules(rules_config: dict, age_limit_phrases) -> dict:
    rules_config = rules_config or {}
    return {
        "age_limit_phrases": compile_phrase_list(age_limit_phrases or [], normalizer=normalize_keyword_text),
        "explicit_age_patterns": compile_pattern_list(rules_config.get("explicit_age_patterns", [])),
        "explicit_word_age_patterns": compile_pattern_list(rules_config.get("explicit_word_age_patterns", [])),
        "implicit_patterns": compile_pattern_list(rules_config.get("implicit_patterns", [])),
        "indirect_patterns": compile_pattern_list(rules_config.get("indirect_patterns", [])),
    }


def compile_direct_opt_out_rules(rules_config: dict) -> dict:
    rules_config = rules_config or {}

    phrase_groups = {}
    phrase_group_map = {
        "phrases": "phrases",
        "wrong_number": "wrong_number",
        "disengagement": "disengagement",
        "block": "block",
    }
    for source_key, reason_key in phrase_group_map.items():
        phrase_groups[reason_key] = compile_phrase_list(
            rules_config.get(source_key, []),
            normalizer=normalize_text,
        )

    return {
        "symbols": sorted(set(rules_config.get("symbols", [])), key=len, reverse=True),
        "phrase_groups": phrase_groups,
        "code_with_stop_patterns": compile_pattern_list(rules_config.get("code_with_stop_patterns", [])),
        "groups": {
            "phrases": compile_pattern_list(rules_config.get("phrases_patterns", [])),
            "wrong_number": compile_pattern_list(rules_config.get("wrong_number_patterns", [])),
            "disengagement": compile_pattern_list(rules_config.get("disengagement_patterns", [])),
            "block": compile_pattern_list(rules_config.get("block_patterns", [])),
        },
    }


def compile_keyword_rules(opt_out_keywords, typo_opt_out_keyword, english_words):
    exact_entries = set()
    exact_single_tokens = set()
    exact_multi_phrases = []

    typo_exact_entries = set()
    typo_exact_tokens = set()
    typo_multi_phrases = []

    for value in opt_out_keywords or []:
        norm = normalize_keyword_text(value)
        if not norm:
            continue
        exact_entries.add(norm)
        if " " in norm:
            exact_multi_phrases.append(norm)
        else:
            exact_single_tokens.add(norm)

    for value in typo_opt_out_keyword or []:
        norm = normalize_keyword_text(value)
        if not norm:
            continue
        typo_exact_entries.add(norm)
        if " " in norm:
            typo_multi_phrases.append(norm)
            for token in norm.split():
                if token:
                    typo_exact_tokens.add(token)
        else:
            typo_exact_tokens.add(norm)

    exact_multi_phrases = sorted(set(exact_multi_phrases), key=len, reverse=True)
    typo_multi_phrases = sorted(set(typo_multi_phrases), key=len, reverse=True)

    fuzzy_reference_tokens = sorted({token for token in exact_single_tokens if token and len(token) >= 3})

    fuzzy_reference_by_first = {}
    for token in fuzzy_reference_tokens:
        fuzzy_reference_by_first.setdefault(token[0], []).append(token)

    exact_phrase_sequences = [tuple(phrase.split()) for phrase in exact_multi_phrases if " " in phrase]
    typo_phrase_sequences = [tuple(phrase.split()) for phrase in typo_multi_phrases if " " in phrase]

    return {
        "exact_entries": exact_entries,
        "exact_single_tokens": exact_single_tokens,
        "exact_single_tokens_sorted": sorted(exact_single_tokens, key=len, reverse=True),
        "exact_multi_phrases": exact_multi_phrases,
        "typo_exact_entries": typo_exact_entries,
        "typo_exact_tokens": typo_exact_tokens,
        "typo_multi_phrases": typo_multi_phrases,
        "exact_phrase_sequences": sorted(set(exact_phrase_sequences), key=len, reverse=True),
        "typo_phrase_sequences": sorted(set(typo_phrase_sequences), key=len, reverse=True),
        "fuzzy_reference_by_first": fuzzy_reference_by_first,
        "english_words": english_words,
    }


def is_under_13(text, under_13_rules):
    if text is None or pd.isna(text):
        return False, ""

    raw_text = str(text).strip()
    if not raw_text:
        return False, ""

    text_norm = normalize_keyword_text(raw_text)
    if not text_norm:
        return False, ""

    words = text_norm.split()
    if not words:
        return False, ""

    age_limit_phrases = under_13_rules.get("age_limit_phrases", [])
    for phrase in age_limit_phrases:
        phrase_tokens = phrase.split()
        if not phrase_tokens:
            continue

        phrase_len = len(phrase_tokens)
        for i in range(0, len(words) - phrase_len):
            if words[i : i + phrase_len] != phrase_tokens:
                continue

            age_token = words[i + phrase_len]
            if age_token.isdigit():
                if int(age_token) < 13:
                    return True, f"{phrase} {age_token}".strip()
            else:
                word_age = NUMBER_WORDS.get(age_token)
                if word_age is not None and word_age < 13:
                    return True, f"{phrase} {age_token}".strip()

    for pattern in under_13_rules.get("explicit_age_patterns", []):
        match = pattern.search(text_norm)
        if match:
            age_token = next((group for group in reversed(match.groups()) if group), "")
            if age_token.isdigit() and int(age_token) < 13:
                return True, match.group(0).strip()

    for pattern in under_13_rules.get("explicit_word_age_patterns", []):
        match = pattern.search(text_norm)
        if match:
            age_token = next((group for group in reversed(match.groups()) if group), "")
            word_age = NUMBER_WORDS.get(age_token)
            if word_age is not None and word_age < 13:
                return True, match.group(0).strip()

    for pattern_group in ("implicit_patterns", "indirect_patterns"):
        for pattern in under_13_rules.get(pattern_group, []):
            match = pattern.search(text_norm)
            if match:
                return True, match.group(0).strip()

    return False, ""

def is_general_keyword_typo(word: str, keyword_rules: dict) -> bool:
    if len(word) < 3:
        return False
    if word in keyword_rules["exact_single_tokens"]:
        return False
    if word in keyword_rules["english_words"]:
        return False

    candidates = keyword_rules["fuzzy_reference_by_first"].get(word[0], [])
    for keyword in candidates:
        if word != keyword and levenshtein_distance_leq_one(word, keyword):
            return True

    return False


def is_stop_family_typo(word: str, keyword_rules: dict) -> bool:
    if not (3 <= len(word) <= 8):
        return False
    if not word.startswith("s"):
        return False
    if word in keyword_rules["english_words"]:
        return False
    return levenshtein_distance_leq_one(word, "stop")


def can_be_composed_of_keywords(words, single_tokens, phrase_sequences):
    if not words:
        return False, []

    matched_triggers = []
    i = 0
    while i < len(words):
        token = words[i]
        if token in single_tokens:
            matched_triggers.append(token)
            i += 1
            continue

        matched_phrase = None
        for phrase_tokens in phrase_sequences:
            phrase_len = len(phrase_tokens)
            if phrase_len <= 1:
                continue
            if tuple(words[i : i + phrase_len]) == phrase_tokens:
                matched_phrase = phrase_tokens
                break

        if matched_phrase is None:
            return False, []

        matched_triggers.append(" ".join(matched_phrase))
        i += len(matched_phrase)

    return True, list(dict.fromkeys(matched_triggers))


def decompose_string_into_tokens(value: str, sorted_tokens):
    if not value:
        return None
    parts = []
    i = 0
    while i < len(value):
        matched = None
        for token in sorted_tokens:
            if value.startswith(token, i):
                matched = token
                break
        if matched is None:
            return None
        parts.append(matched)
        i += len(matched)
    return parts


def analyze_keyword_like_fragment(fragment: str, keyword_rules: dict) -> dict:
    empty = {"reason": None, "triggers": []}
    fragment = normalize_keyword_text(fragment)
    if not fragment:
        return empty

    words = fragment.split()
    if not words:
        return empty

    exact_entries = keyword_rules["exact_entries"]
    exact_single_tokens = keyword_rules["exact_single_tokens"]
    exact_phrase_sequences = keyword_rules["exact_phrase_sequences"]
    typo_exact_entries = keyword_rules["typo_exact_entries"]
    typo_exact_tokens = keyword_rules["typo_exact_tokens"]
    typo_phrase_sequences = keyword_rules["typo_phrase_sequences"]
    exact_single_tokens_sorted = keyword_rules["exact_single_tokens_sorted"]

    if fragment in exact_entries:
        return {"reason": "opt_out_keywords", "triggers": [fragment]}

    if fragment in typo_exact_entries:
        return {"reason": "typo_opt_out_keyword", "triggers": [fragment]}

    exact_only, exact_triggers = can_be_composed_of_keywords(
        words,
        exact_single_tokens,
        exact_phrase_sequences,
    )
    if exact_only:
        return {"reason": "opt_out_keywords", "triggers": exact_triggers}

    typo_only, typo_triggers = can_be_composed_of_keywords(
        words,
        typo_exact_tokens,
        typo_phrase_sequences,
    )
    if typo_only:
        return {"reason": "typo_opt_out_keyword", "triggers": typo_triggers}

    if len(words) == 1:
        word = words[0]
        if word in exact_single_tokens:
            return {"reason": "opt_out_keywords", "triggers": [word]}
        if word in typo_exact_tokens:
            return {"reason": "typo_opt_out_keyword", "triggers": [word]}
        if is_general_keyword_typo(word, keyword_rules) or is_stop_family_typo(word, keyword_rules):
            return {"reason": "typo_opt_out_keyword", "triggers": [word]}

        collapsed = collapse_repeated_chars(word) if len(word) >= 4 else word
        if collapsed != word:
            if collapsed in exact_single_tokens:
                return {"reason": "opt_out_keywords", "triggers": [word]}
            if collapsed in typo_exact_tokens:
                return {"reason": "typo_opt_out_keyword", "triggers": [word]}
            if is_general_keyword_typo(collapsed, keyword_rules) or is_stop_family_typo(collapsed, keyword_rules):
                return {"reason": "typo_opt_out_keyword", "triggers": [word]}

        parts = decompose_string_into_tokens(word, exact_single_tokens_sorted)
        if parts:
            return {"reason": "opt_out_keywords", "triggers": parts}

    return empty


def analyze_brand_keyword_combo(raw_customer, brand_message, keyword_rules, reference_messages=None):
    if raw_customer is None or pd.isna(raw_customer):
        return []
    if brand_message is None or pd.isna(brand_message):
        brand_message = ""

    customer_norm = normalize_keyword_text(raw_customer)
    brand_norm = normalize_keyword_text(brand_message)
    if not customer_norm:
        return []

    customer_words = customer_norm.split()
    if not customer_words:
        return []

    exact_single_tokens = keyword_rules["exact_single_tokens"]
    typo_exact_tokens = keyword_rules["typo_exact_tokens"]

    def is_distinctive_brand_token(token: str) -> bool:
        if len(token) < 4:
            return False
        if token in exact_single_tokens:
            return False
        return any(ch.isdigit() for ch in token) or len(token) >= 6

    triggers_by_reason = {
        "opt_out_keywords": [],
        "typo_opt_out_keyword": [],
    }
    brand_tokens = {t for t in brand_norm.split() if t}
    distinctive_brand_tokens = {t for t in brand_tokens if is_distinctive_brand_token(t)}

    if distinctive_brand_tokens:
        for i, word in enumerate(customer_words):
            if word not in distinctive_brand_tokens:
                continue
            prev_word = customer_words[i - 1] if i > 0 else ""
            next_word = customer_words[i + 1] if i + 1 < len(customer_words) else ""
            if prev_word in exact_single_tokens:
                triggers_by_reason["opt_out_keywords"].append(prev_word)
            elif prev_word in typo_exact_tokens:
                triggers_by_reason["typo_opt_out_keyword"].append(prev_word)
            if next_word in exact_single_tokens:
                triggers_by_reason["opt_out_keywords"].append(next_word)
            elif next_word in typo_exact_tokens:
                triggers_by_reason["typo_opt_out_keyword"].append(next_word)

        for word in customer_words:
            for bt in distinctive_brand_tokens:
                if not bt or bt == word or bt not in word:
                    continue

                if word.startswith(bt):
                    tail = word[len(bt) :]
                    result = analyze_keyword_like_fragment(tail, keyword_rules)
                    if result["reason"]:
                        triggers_by_reason[result["reason"]].extend(result["triggers"])

                if word.endswith(bt):
                    head = word[: -len(bt)]
                    result = analyze_keyword_like_fragment(head, keyword_rules)
                    if result["reason"]:
                        triggers_by_reason[result["reason"]].extend(result["triggers"])

    normalized_references = []
    if brand_norm:
        normalized_references.append(brand_norm)
    for reference in reference_messages or []:
        reference_norm = normalize_keyword_text(reference)
        if reference_norm and reference_norm not in normalized_references:
            normalized_references.append(reference_norm)

    for reference_norm in normalized_references:
        if customer_norm == reference_norm:
            continue

        if customer_norm.startswith(reference_norm):
            suffix = customer_norm[len(reference_norm) :].strip()
            suffix_result = analyze_keyword_like_fragment(suffix, keyword_rules)
            if suffix_result["reason"]:
                triggers_by_reason[suffix_result["reason"]].extend(suffix_result["triggers"])
            elif suffix:
                suffix_words = suffix.split()
                edge_candidates = []
                if suffix_words:
                    edge_candidates.append(suffix_words[0])
                    edge_candidates.append(suffix_words[-1])
                for candidate in edge_candidates:
                    candidate_result = analyze_keyword_like_fragment(candidate, keyword_rules)
                    if candidate_result["reason"]:
                        triggers_by_reason[candidate_result["reason"]].extend(candidate_result["triggers"])

        customer_compact = customer_norm.replace(" ", "")
        reference_compact = reference_norm.replace(" ", "")
        if customer_compact.startswith(reference_compact) and customer_compact != reference_compact:
            suffix = customer_compact[len(reference_compact) :]
            suffix_result = analyze_keyword_like_fragment(suffix, keyword_rules)
            if suffix_result["reason"]:
                triggers_by_reason[suffix_result["reason"]].extend(suffix_result["triggers"])

    return {
        reason: list(dict.fromkeys(triggers))
        for reason, triggers in triggers_by_reason.items()
        if triggers
    }


def analyze_keyword_message(raw_text, keyword_rules):
    empty = {
        "reason": None,
        "triggers": [],
        "exact_in_phrase_triggers": [],
        "keyword_only_message": False,
    }

    if raw_text is None or pd.isna(raw_text):
        return empty

    normalized = normalize_keyword_text(raw_text)
    if not normalized:
        return empty

    words = normalized.split()
    if not words:
        return empty

    exact_entries = keyword_rules["exact_entries"]
    exact_single_tokens = keyword_rules["exact_single_tokens"]
    exact_phrase_sequences = keyword_rules["exact_phrase_sequences"]

    typo_exact_entries = keyword_rules["typo_exact_entries"]
    typo_exact_tokens = keyword_rules["typo_exact_tokens"]
    typo_phrase_sequences = keyword_rules["typo_phrase_sequences"]

    if normalized in exact_entries:
        return {
            "reason": "opt_out_keywords",
            "triggers": [normalized],
            "exact_in_phrase_triggers": [],
            "keyword_only_message": True,
        }

    if normalized in typo_exact_entries:
        return {
            "reason": "typo_opt_out_keyword",
            "triggers": [normalized],
            "exact_in_phrase_triggers": [],
            "keyword_only_message": True,
        }

    exact_only, exact_triggers = can_be_composed_of_keywords(
        words,
        exact_single_tokens,
        exact_phrase_sequences,
    )
    if exact_only:
        return {
            "reason": "opt_out_keywords",
            "triggers": exact_triggers,
            "exact_in_phrase_triggers": [],
            "keyword_only_message": True,
        }

    typo_only, typo_triggers = can_be_composed_of_keywords(
        words,
        typo_exact_tokens,
        typo_phrase_sequences,
    )
    if typo_only:
        return {
            "reason": "typo_opt_out_keyword",
            "triggers": typo_triggers,
            "exact_in_phrase_triggers": [],
            "keyword_only_message": True,
        }

    levenshtein_typo_triggers = []
    for word in words:
        candidates = [word]
        if len(word) >= 4:
            collapsed = collapse_repeated_chars(word)
            if collapsed != word:
                candidates.append(collapsed)

        matched = False
        for candidate in candidates:
            if candidate in typo_exact_tokens:
                matched = True
                break
            if is_general_keyword_typo(candidate, keyword_rules) or is_stop_family_typo(candidate, keyword_rules):
                matched = True
                break

        if matched:
            levenshtein_typo_triggers.append(word)
        else:
            levenshtein_typo_triggers = []
            break

    if levenshtein_typo_triggers:
        return {
            "reason": "typo_opt_out_keyword",
            "triggers": list(dict.fromkeys(levenshtein_typo_triggers)),
            "exact_in_phrase_triggers": [],
            "keyword_only_message": True,
        }

    mixed_triggers = []
    has_typo_like = False
    for word in words:
        candidates = [word]
        if len(word) >= 4:
            collapsed = collapse_repeated_chars(word)
            if collapsed != word:
                candidates.append(collapsed)

        token_is_exact = any(candidate in exact_single_tokens for candidate in candidates)
        token_is_typo = any(candidate in typo_exact_tokens for candidate in candidates)
        if not token_is_typo:
            token_is_typo = any(
                is_general_keyword_typo(candidate, keyword_rules) or is_stop_family_typo(candidate, keyword_rules)
                for candidate in candidates
            )

        if token_is_typo:
            has_typo_like = True
            mixed_triggers.append(word)
            continue

        if token_is_exact:
            mixed_triggers.append(word)
            continue

        mixed_triggers = []
        has_typo_like = False
        break

    if mixed_triggers and has_typo_like:
        return {
            "reason": "typo_opt_out_keyword",
            "triggers": list(dict.fromkeys(mixed_triggers)),
            "exact_in_phrase_triggers": [],
            "keyword_only_message": True,
        }

    # Allow otherwise keyword-only messages with extra numeric noise
    # like "out 9" or "stop 2" to fall into the typo bucket.
    non_numeric_words = [word for word in words if not word.isdigit()]
    numeric_words = [word for word in words if word.isdigit()]
    if non_numeric_words and numeric_words:
        exact_only_with_numbers, exact_number_triggers = can_be_composed_of_keywords(
            non_numeric_words,
            exact_single_tokens,
            exact_phrase_sequences,
        )
        if exact_only_with_numbers:
            return {
                "reason": "typo_opt_out_keyword",
                "triggers": [normalized],
                "exact_in_phrase_triggers": [],
                "keyword_only_message": True,
            }

        typo_only_with_numbers, typo_number_triggers = can_be_composed_of_keywords(
            non_numeric_words,
            typo_exact_tokens,
            typo_phrase_sequences,
        )
        if typo_only_with_numbers:
            return {
                "reason": "typo_opt_out_keyword",
                "triggers": [normalized],
                "exact_in_phrase_triggers": [],
                "keyword_only_message": True,
            }

    return empty


def find_full_phrase_hits(normalized_text_value: str, normalized_phrases) -> list:
    if not normalized_text_value:
        return []
    padded = f" {normalized_text_value} "
    hits = []
    for phrase in normalized_phrases:
        if f" {phrase} " in padded:
            hits.append(phrase)
    return hits


def get_direct_opt_out_matches(text, direct_opt_out_rules):
    if text is None or pd.isna(text):
        return []

    raw_text = str(text).strip()
    if not raw_text:
        return []

    normalized_text = normalize_text(raw_text)

    matches = []

    for symbol in direct_opt_out_rules.get("symbols", []):
        if symbol and symbol in raw_text:
            matches.append({"reason": "symbols", "trigger": symbol})

    for pattern in direct_opt_out_rules["code_with_stop_patterns"]:
        match = pattern.search(normalized_text)
        if match:
            matches.append({"reason": "phrases", "trigger": match.group(0).strip()})

    for reason_name, patterns in direct_opt_out_rules["groups"].items():
        for pattern in patterns:
            match = pattern.search(normalized_text)
            if match:
                matches.append({"reason": reason_name, "trigger": match.group(0).strip()})
                break

    for reason_name, phrases in direct_opt_out_rules.get("phrase_groups", {}).items():
        for trigger in find_full_phrase_hits(normalized_text, phrases):
            matches.append({"reason": reason_name, "trigger": trigger})

    deduped = []
    seen = set()
    for item in matches:
        key = (item["reason"], item["trigger"])
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    return deduped


def build_output_flags(reasons, triggers_by_reason):
    direct_reasons = [r for r in reasons if r in DIRECT_REASON_NAMES]
    under_13_reasons = [r for r in reasons if r == "under_13"]
    offensive_reasons = [r for r in reasons if r in {"offensive_symbols", "frustration"}]

    direct_triggers = []
    for reason in direct_reasons:
        direct_triggers.extend(triggers_by_reason.get(reason, []))

    if direct_reasons:
        detected_direct_opt_out = "Direct Opt-Out"
        detected_direct_opt_out_trigger = ", ".join(dict.fromkeys(direct_triggers))
    else:
        detected_direct_opt_out = "No Opt-Out"
        detected_direct_opt_out_trigger = ""

    if direct_reasons:
        detected_keyword = "No Keyword"
        detected_keyword_trigger = ""
    elif "opt_out_keywords" in reasons:
        detected_keyword = "Keyword"
        detected_keyword_trigger = ", ".join(dict.fromkeys(triggers_by_reason.get("opt_out_keywords", [])))
    elif "typo_opt_out_keyword" in reasons:
        detected_keyword = "Typo Keyword"
        detected_keyword_trigger = ", ".join(dict.fromkeys(triggers_by_reason.get("typo_opt_out_keyword", [])))
    else:
        detected_keyword = "No Keyword"
        detected_keyword_trigger = ""

    if under_13_reasons:
        detected_under_13 = "Under 13"
        detected_under_13_trigger = ", ".join(dict.fromkeys(triggers_by_reason.get("under_13", [])))
    else:
        detected_under_13 = "Not Under 13"
        detected_under_13_trigger = ""

    offensive_triggers = []
    for reason in offensive_reasons:
        offensive_triggers.extend(triggers_by_reason.get(reason, []))

    if offensive_reasons:
        detected_offensive = "Offensive"
        detected_offensive_trigger = ", ".join(dict.fromkeys(offensive_triggers))
    else:
        detected_offensive = "Not Offensive"
        detected_offensive_trigger = ""

    return {
        "Detected Direct Opt-Out": detected_direct_opt_out,
        "Detected Direct Opt-Out Trigger": detected_direct_opt_out_trigger,
        "Detected Keyword": detected_keyword,
        "Detected Keyword Trigger": detected_keyword_trigger,
        "Detected Under 13": detected_under_13,
        "Detected Under 13 Trigger": detected_under_13_trigger,
        "Detected Offensive": detected_offensive,
        "Detected Offensive Trigger": detected_offensive_trigger,
    }


def unique_triggers_for_reasons(triggers_by_reason, reason_names):
    merged = []
    for reason in reason_names:
        merged.extend(triggers_by_reason.get(reason, []))
    return list(dict.fromkeys(merged))


def build_template_text(reasons, triggers_by_reason):
    if not reasons:
        return ""

    templates = []

    if "opt_out_keywords" in reasons:
        kw = ", ".join(unique_triggers_for_reasons(triggers_by_reason, ["opt_out_keywords"])) or "the detected keyword"
        templates.append(
            f'The word "{kw}" is a keyword for unsubscribing.'
        )

    if "typo_opt_out_keyword" in reasons:
        typo_kw = ", ".join(unique_triggers_for_reasons(triggers_by_reason, ["typo_opt_out_keyword"])) or "the detected variation"
        templates.append(
            f'The phrase "{typo_kw}" should be treated as an unsubscribe request.'
        )

    if "under_13" in reasons:
        templates.append(
            "The customer implied they are under 13 years old. Due to legal requirements, customers under 13 must not be messaged."
        )

    if "symbols" in reasons:
        sym = ", ".join(unique_triggers_for_reasons(triggers_by_reason, ["symbols"])) or "the detected symbol"
        templates.append(
            f'The symbol "{sym}" expresses an unsubscribe intent.'
        )

    if "offensive_symbols" in reasons:
        sym = ", ".join(unique_triggers_for_reasons(triggers_by_reason, ["offensive_symbols"])) or "the detected symbol"
        templates.append(
            f'The offensive symbol "{sym}" indicates the customer does not want further messages.'
        )

    if "wrong_number" in reasons:
        templates.append(
            "Any phrase that indicates a number is not working or the customer didn't subscribe means they need to be unsubscribed."
        )

    if "block" in reasons:
        blk = ", ".join(unique_triggers_for_reasons(triggers_by_reason, ["block"])) or "the detected phrase"
        templates.append(
            f'Saying "{blk}" is a clear unsubscribe intent by the customer which we must respect.'
        )

    if "disengagement" in reasons:
        diseng = ", ".join(unique_triggers_for_reasons(triggers_by_reason, ["disengagement"])) or "the detected phrase"
        templates.append(
            f'Saying "{diseng}" needs to be treated as an unsubscribe request since it indicates the customer''s disengagement.'
        )

    if "frustration" in reasons:
        frustr = ", ".join(unique_triggers_for_reasons(triggers_by_reason, ["frustration"])) or "the detected phrase"
        templates.append(
            f'Offensive phrases such as "{frustr}" are an indication for us to stop messaging.'
        )

    if "phrases" in reasons:
        phr = ", ".join(unique_triggers_for_reasons(triggers_by_reason, ["phrases"])) or "the detected phrase"
        templates.append(
            f'Saying "{phr}" is a clear unsubscribe intent by the customer which we must respect.'
        )

    if "opt_out_device_not_working" in reasons:
        templates.append(
            "The message indicates the device cannot receive text messages and should be treated as an unsubscribe request."
        )

    if "journeys" in reasons:
        templates.append(
            "The journey asked for the customer's preference, and they told us they would not like to receive texts."
        )

    if not templates:
        return ""

    connectors = ["Additionally", "Also", "Furthermore"]
    merged = [templates[0]]
    for idx, text in enumerate(templates[1:], start=1):
        connector = connectors[(idx - 1) % len(connectors)]
        merged.append(f"{connector}, {text[0].lower()}{text[1:]}")
    return " ".join(merged)


def iter_with_progress(iterable, total: int, description: str):
    if tqdm is not None:
        return tqdm(iterable, total=total, desc=description)
    return iterable


def normalize_ollama_guard_text(text: str) -> str:
    normalized = strip_diacritics_and_compat(text).lower()
    normalized = re.sub(r'[\'’`"“”]', '', normalized)
    normalized = WHITESPACE_RE.sub(' ', normalized).strip()
    return normalized


def protect_quoted_phrases(text: str, quoted_phrases: list[str]) -> tuple[str, dict]:
    protected_text = text
    placeholder_map = {}
    for idx, phrase in enumerate(quoted_phrases):
        placeholder = f"__QUOTED_PHRASE_{idx}__"
        exact_quoted_phrase = f'"{phrase}"'
        protected_text = protected_text.replace(exact_quoted_phrase, placeholder)
        placeholder_map[placeholder] = exact_quoted_phrase
    return protected_text, placeholder_map


def restore_quoted_phrases(text: str, placeholder_map: dict) -> str:
    restored = text
    for placeholder, exact_quoted_phrase in placeholder_map.items():
        restored = restored.replace(placeholder, exact_quoted_phrase)
    return restored


def strip_placeholder_wrapper_quotes(text: str, placeholder_map: dict) -> str:
    cleaned = text
    for placeholder in placeholder_map:
        pattern = re.compile(rf'(?:"|“|”)+{re.escape(placeholder)}(?:"|“|”)+')
        cleaned = pattern.sub(placeholder, cleaned)
    return cleaned


def strip_ollama_wrapper_quotes(text: str) -> str:
    cleaned = text
    wrapper_quote_patterns = [
        re.compile(r'(:\s+)[“"]([^"\n]{20,}?[,.!?])["”](?=(?:\s|$))'),
        re.compile(r'(\b(?:contained|contains|regarding|specifically)\s+)[“"]([^"\n]{20,}?[,.!?])["”](?=(?:\s|$))', re.IGNORECASE),
    ]
    for pattern in wrapper_quote_patterns:
        cleaned = pattern.sub(r'\1\2', cleaned)
    return cleaned


def rephrase_template_with_ollama(template_text: str) -> str:
    if not template_text:
        return ""
    preview = template_text.replace("\n", " ")[:80]
    print(f"[ollama] Rephrasing template: {preview}")
    return rephrase_template_with_ollama_cached(template_text)


@lru_cache(maxsize=50000)
def rephrase_template_with_ollama_cached(template_text: str) -> str:
    quoted_phrases = re.findall(r'"([^"]+)"', template_text)
    protected_template, placeholder_map = protect_quoted_phrases(template_text, quoted_phrases)
    prompt = (
        "Lightly rephrase the following customer-support compliance note so it does not sound identical to other notes. Use first person. "
        "Preserve the exact meaning. Preserve all protected placeholders exactly as written. "
        "Do not add any new quotation marks around sentences, clauses, or summaries. "
        "Only the original quoted trigger phrases should appear in quotation marks, and they must stay in double quotes exactly. "
        "Do not swap channels, do not add or remove facts, do not change the reason, and keep it concise. "
        "If the template is already concise, make only minimal changes. Return plain text only.\n\n"
        f"Template:\n{protected_template}"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            raw_response = response.read().decode("utf-8")
        parsed = json.loads(raw_response)
        rewritten = str(parsed.get("response", "")).strip()
        if not rewritten:
            print("[ollama] Empty response, using original template")
            return template_text
        missing_placeholders = [placeholder for placeholder in placeholder_map if placeholder not in rewritten]
        if missing_placeholders:
            print(f"[ollama] Rephrase dropped protected quoted placeholders {missing_placeholders}, using original template")
            return template_text
        rewritten = strip_ollama_wrapper_quotes(rewritten)
        rewritten = strip_placeholder_wrapper_quotes(rewritten, placeholder_map)
        rewritten = restore_quoted_phrases(rewritten, placeholder_map)
        normalized_rewritten = normalize_ollama_guard_text(rewritten)
        missing_phrases = [
            phrase for phrase in quoted_phrases
            if normalize_ollama_guard_text(phrase) not in normalized_rewritten
        ]
        if missing_phrases:
            print(f"[ollama] Rephrase dropped quoted wording {missing_phrases}, using original template")
            return template_text
        print("[ollama] Rephrase succeeded")
        return rewritten
    except (urllib.error.URLError, TimeoutError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"[ollama] Rephrase failed, using original template: {exc}")
        return template_text


@lru_cache(maxsize=50000)
def shorten_link_to_dagd(url: str) -> str:
    if not url:
        return ""
    if DAGD_SHORTENER is None:
        return url
    try:
        short = DAGD_SHORTENER.dagd.short(url)
        return short or url
    except Exception:
        return url


def first_present_column(df: pd.DataFrame, candidates) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return ""


def parse_week_to_mmm_dd(value) -> str:
    if value is None or pd.isna(value):
        return ""
    dt = pd.to_datetime(value, errors="coerce")
    if not pd.isna(dt):
        return dt.strftime("%b %d")
    text = str(value).strip()
    dt2 = pd.to_datetime(text, errors="coerce")
    if not pd.isna(dt2):
        return dt2.strftime("%b %d")
    return text


def build_agent_message(first_name: str, week_value, conversation_lines: list) -> str:
    week_display = parse_week_to_mmm_dd(week_value) or "N/A"
    first_name_display = (first_name or "").strip() or "there"
    conversation_count = len(conversation_lines)
    flag_phrase = "a flag" if conversation_count == 1 else "some flags"
    conversation_noun = "conversation" if conversation_count == 1 else "conversations"

    intro = (
        f"Hi {first_name_display}, hope you're doing well! We were reviewing opt outs for the week of "
        f"{week_display} and came across {flag_phrase} for you."
    )
    bullets = "\n".join(f"- {line}" for line in conversation_lines)
    close = (
        f"Please take some time to review the {conversation_noun}, and let me know if you have any questions. "
        "Each subscriber we do not opt out can sue our client, bringing a lot of legal trouble to them and "
        "mistrust to Attentive, so this is a serious issue that should not be repeated."
    )
    return f"{intro}\n{bullets}\n{close}"


def split_trigger_values(value: str) -> list:
    if not value:
        return []
    text = str(value).strip()
    if not text:
        return []
    parts = [part.strip() for part in text.split(",")]
    return [part for part in parts if part]


@lru_cache(maxsize=1)
def get_agent_template_category_patterns() -> dict:
    config = load_config(CONFIG_PATH)
    direct_opt_out_config = get_config_section(config, "direct_opt_out")
    offensive_config = get_config_section(config, "offensive")

    return {
        "phrases": compile_pattern_list(direct_opt_out_config.get("phrases_patterns", [])),
        "wrong_number": compile_pattern_list(direct_opt_out_config.get("wrong_number_patterns", [])),
        "disengagement": compile_pattern_list(direct_opt_out_config.get("disengagement_patterns", [])),
        "block": compile_pattern_list(direct_opt_out_config.get("block_patterns", [])),
        "frustration": compile_pattern_list(offensive_config.get("frustration_patterns", [])),
        "offensive_symbols": sorted(set(offensive_config.get("symbols", [])), key=len, reverse=True),
    }


def collapse_similar_triggers(triggers: list) -> list:
    normalized_pairs = []
    for trigger in triggers:
        text = str(trigger).strip()
        if not text:
            continue
        normalized_pairs.append((text, normalize_text(text)))

    normalized_pairs.sort(key=lambda item: (-len(item[1]), item[0]))
    kept = []
    kept_norms = []
    for original, normalized in normalized_pairs:
        if not normalized:
            continue
        if any(
            normalized == existing
            or normalized in existing
            or existing in normalized
            for existing in kept_norms
        ):
            continue
        kept.append(original)
        kept_norms.append(normalized)

    return kept


def extract_raw_phrase_from_message(message_text: str, normalized_trigger: str) -> str:
    if not message_text or not normalized_trigger:
        return ""

    trigger_tokens = normalized_trigger.split()
    if not trigger_tokens:
        return ""

    raw_tokens = []
    for match in re.finditer(r"[A-Za-z0-9]+(?:['’`][A-Za-z0-9]+)?", message_text):
        raw_token = match.group(0)
        normalized_token = normalize_text(raw_token)
        if not normalized_token:
            continue
        raw_tokens.append(
            {
                "raw": raw_token,
                "normalized": normalized_token,
                "start": match.start(),
                "end": match.end(),
            }
        )

    if not raw_tokens:
        return ""

    trigger_len = len(trigger_tokens)
    for start_idx in range(len(raw_tokens) - trigger_len + 1):
        window = raw_tokens[start_idx : start_idx + trigger_len]
        if [token["normalized"] for token in window] == trigger_tokens:
            start = window[0]["start"]
            end = window[-1]["end"]
            return message_text[start:end].strip()

    return ""


def select_category_trigger(category: str, triggers: list) -> list:
    cleaned = collapse_similar_triggers(triggers)
    if not cleaned:
        return []

    if category == "direct":
        return cleaned

    if category in {"keyword", "typo_keyword", "symbols", "age_limit", "journeys", "device_not_working"}:
        return cleaned[:1]

    if category == "offensive_symbols":
        return cleaned[:1]

    return cleaned[:1]


def select_category_trigger_from_message(category: str, message_text: str, fallback_triggers: list) -> list:
    normalized_message = normalize_text(message_text)
    category_patterns = get_agent_template_category_patterns().get(category, [])
    if category_patterns and normalized_message:
        if category == "offensive_symbols":
            symbol_hits = [symbol for symbol in category_patterns if symbol and symbol in message_text]
            symbol_hits = collapse_similar_triggers(symbol_hits)
            if symbol_hits:
                return symbol_hits[:1]

        pattern_hits = []
        for pattern in category_patterns:
            match = pattern.search(normalized_message)
            if match:
                normalized_hit = match.group(0).strip()
                raw_hit = extract_raw_phrase_from_message(message_text, normalized_hit)
                pattern_hits.append(raw_hit or normalized_hit)
        pattern_hits = collapse_similar_triggers(pattern_hits)
        if pattern_hits:
            return pattern_hits[:1]

    return select_category_trigger(category, fallback_triggers)


def render_collapsed_reason_bullet(category: str, links: list, triggers: list) -> str:
    links = list(dict.fromkeys([l for l in links if l]))
    triggers = select_category_trigger(category, list(dict.fromkeys([t for t in triggers if t])))
    is_multi = len(links) > 1
    conversation_prefix = "Conversations" if is_multi else "Conversation"
    link_text = " & ".join(links) if links else "N/A"
    convo_label = f"{conversation_prefix} {link_text}:"

    if category == "keyword":
        kw_text = '" and "'.join(triggers) if triggers else "the detected keyword"
        noun = "word" if len(triggers) == 1 else "words"
        verb = "is" if len(triggers) == 1 else "are"
        predicate = "a keyword" if len(triggers) == 1 else "keywords"
        return f'{convo_label} The {noun} "{kw_text}" {verb} {predicate} for unsubscribing.'

    if category == "typo_keyword":
        kw_text = '" and "'.join(triggers) if triggers else "the detected variation"
        noun = "phrase" if len(triggers) == 1 else "phrases"
        verb = "is" if len(triggers) == 1 else "are"
        return f'{convo_label} The {noun} "{kw_text}" should be treated as an unsubscribe request.'

    if category == "age_limit":
        return f"{convo_label} The customer implied they are under 13 years old. Due to legal requirements, customers under 13 must not be messaged."

    if category == "symbols":
        sym = '" and "'.join(triggers) if triggers else "the detected symbol"
        noun = "symbol" if len(triggers) == 1 else "symbols"
        verb = "expresses" if len(triggers) == 1 else "express"
        return f'{convo_label} The {noun} "{sym}" {verb} an unsubscribe intent.'

    if category == "offensive_symbols":
        sym = '" and "'.join(triggers) if triggers else "the detected symbol"
        noun = "symbol" if len(triggers) == 1 else "symbols"
        return f'{convo_label} The offensive {noun} "{sym}" indicates the customer does not want further messages.'

    if category == "direct":
        phr = '" and "'.join(triggers) if triggers else "the detected phrase"
        verb = "is" if len(triggers) == 1 else "are"
        noun = "intent" if len(triggers) == 1 else "intents"
        return f'{convo_label} Saying "{phr}" {verb} clear unsubscribe {noun} by the customer which we must respect.'

    if category == "wrong_number":
        return f"{convo_label} Any phrase that indicates a number is not working or the customer didn't subscribe means they need to be unsubscribed."

    if category == "block":
        phr = '" and "'.join(triggers) if triggers else "the detected phrase"
        return f'{convo_label} Saying "{phr}" is a clear unsubscribe intent by the customer which we must respect.'

    if category == "disengagement":
        phr = '" and "'.join(triggers) if triggers else "the detected phrase"
        return f'{convo_label} Saying "{phr}" needs to be treated as an unsubscribe request since it indicates the customer\'s disengagement.'

    if category == "frustration":
        phr = '" and "'.join(triggers) if triggers else "the detected phrase"
        return f'{convo_label} Offensive phrases such as "{phr}" are an indication for us to stop messaging.'

    if category == "phrases":
        phr = '" and "'.join(triggers) if triggers else "the detected phrase"
        return f'{convo_label} Saying "{phr}" is a clear unsubscribe intent by the customer which we must respect.'

    if category == "journeys":
        return f"{convo_label} The journey asked for the customer's preference, and they told us they would not like to receive texts."

    if category == "device_not_working":
        return f"{convo_label} The message indicates the device cannot receive text messages and should be treated as an unsubscribe request."

    return f"{convo_label} {', '.join(triggers) if triggers else 'Flagged opt-out reason.'}"


def build_agent_templates_output(output_df: pd.DataFrame) -> pd.DataFrame:
    agent_email_col = first_present_column(output_df, ["AGENT_EMAIL", "Agent Email", "agent_email"])
    agent_full_name_col = first_present_column(
        output_df,
        ["AGENT_NAME", "Agent Name", "agent_name", "AGENT_FULL_NAME", "Agent Full Name", "agent_full_name"],
    )
    first_name_col = first_present_column(output_df, ["FIRST_NAME", "First Name", "first_name"])
    week_col = first_present_column(output_df, ["WEEK", "Week", "week", "EVENT_WEEK", "Event Week"])
    customer_message_col = first_present_column(
        output_df,
        [CUSTOMER_COL, "CUSTOMER_MESSAGE", "Customer Message", "customer_message"],
    )
    link_col = first_present_column(
        output_df,
        ["LINK", "Link", "link", "CONVERSATION_URL", "Conversation URL", "conversation_url"],
    )

    result_columns = [
        "Agent Email",
        "Agent Full Name",
        "First Name",
        "Week",
        "Flag Count",
        "Conversation Messages",
        "Template",
        "OLLAMA_TEMPLATE",
    ]
    if not agent_email_col:
        return pd.DataFrame(columns=result_columns)

    flagged = output_df[output_df["OPT_OUT"] == 1].copy()
    if flagged.empty:
        return pd.DataFrame(columns=result_columns)

    rows = []
    for agent_email, group in flagged.groupby(agent_email_col, dropna=False):
        group = group.copy()
        if group.empty:
            continue

        agent_full_name = ""
        if agent_full_name_col and agent_full_name_col in group.columns:
            agent_full_name = str(group[agent_full_name_col].iloc[0]) if pd.notna(group[agent_full_name_col].iloc[0]) else ""

        first_name = ""
        if first_name_col and first_name_col in group.columns:
            first_name = str(group[first_name_col].iloc[0]) if pd.notna(group[first_name_col].iloc[0]) else ""
        elif agent_full_name:
            first_name = agent_full_name.split()[0]

        week_value = ""
        if week_col and week_col in group.columns:
            week_value = group[week_col].iloc[0]

        reason_buckets = {}
        conversation_message_lines = []
        for _, row in group.iterrows():
            link_value = row.get(link_col, "") if link_col else ""
            link_text = str(link_value).strip() if pd.notna(link_value) else ""
            link_display = shorten_link_to_dagd(link_text) if link_text else ""
            raw_customer_message = row.get(customer_message_col, "") if customer_message_col else ""
            customer_message_text = str(raw_customer_message).replace("\n", " ").strip() if pd.notna(raw_customer_message) else ""
            convo_idx = len(conversation_message_lines) + 1
            conversation_message_lines.append(f"Convo {convo_idx}: {customer_message_text}")
            reason_values = [r.strip() for r in str(row.get("REASON", "")).split(",") if r.strip()]
            keyword_triggers = split_trigger_values(str(row.get("Detected Keyword Trigger", "")))
            under13_triggers = split_trigger_values(str(row.get("Detected Under 13 Trigger", "")))
            direct_triggers = split_trigger_values(str(row.get("Detected Direct Opt-Out Trigger", "")))
            offensive_triggers = split_trigger_values(str(row.get("Detected Offensive Trigger", "")))

            for reason in reason_values:
                if reason == "opt_out_keywords":
                    key = "keyword"
                    trigs = select_category_trigger_from_message(key, customer_message_text, keyword_triggers)
                elif reason == "typo_opt_out_keyword":
                    key = "typo_keyword"
                    trigs = select_category_trigger_from_message(key, customer_message_text, keyword_triggers)
                elif reason == "under_13":
                    key = "age_limit"
                    trigs = select_category_trigger_from_message(key, customer_message_text, under13_triggers)
                elif reason == "symbols":
                    key = "symbols"
                    trigs = select_category_trigger_from_message(key, customer_message_text, direct_triggers + offensive_triggers)
                elif reason == "offensive_symbols":
                    key = "offensive_symbols"
                    trigs = select_category_trigger_from_message(key, customer_message_text, offensive_triggers)
                elif reason == "wrong_number":
                    key = "wrong_number"
                    trigs = select_category_trigger_from_message(key, customer_message_text, direct_triggers)
                elif reason == "block":
                    key = "direct"
                    trigs = select_category_trigger_from_message("block", customer_message_text, direct_triggers)
                elif reason == "disengagement":
                    key = "direct"
                    trigs = select_category_trigger_from_message("disengagement", customer_message_text, direct_triggers)
                elif reason == "frustration":
                    key = "frustration"
                    trigs = select_category_trigger_from_message(key, customer_message_text, offensive_triggers or direct_triggers)
                elif reason == "phrases":
                    key = "direct"
                    trigs = select_category_trigger_from_message("phrases", customer_message_text, direct_triggers)
                elif reason == "opt_out_device_not_working":
                    key = "device_not_working"
                    trigs = select_category_trigger_from_message(key, customer_message_text, direct_triggers)
                elif reason == "journeys":
                    key = "journeys"
                    trigs = select_category_trigger_from_message(key, customer_message_text, direct_triggers)
                else:
                    continue

                bucket = reason_buckets.setdefault(key, {"links": [], "triggers": []})
                if link_display:
                    bucket["links"].append(link_display)
                bucket["triggers"].extend(trigs)

        conversation_lines = [
            render_collapsed_reason_bullet(cat, data["links"], data["triggers"])
            for cat, data in reason_buckets.items()
        ]
        if not conversation_lines:
            conversation_lines = ["Conversation: No reason provided."]

        template_text = build_agent_message(first_name, week_value, conversation_lines)
        agent_email_display = str(agent_email).strip() if pd.notna(agent_email) else ""
        rows.append(
            {
                "Agent Email": agent_email_display,
                "Agent Full Name": agent_full_name,
                "First Name": first_name,
                "Week": parse_week_to_mmm_dd(week_value),
                "Flag Count": len(group),
                "Conversation Messages": "; ".join(conversation_message_lines),
                "Template": template_text,
                "OLLAMA_TEMPLATE": rephrase_template_with_ollama(template_text),
            }
        )

    return pd.DataFrame(rows, columns=result_columns)


def build_no_match_output():
    output = {
        "OPT_OUT": 0,
        "REASON": "no_reason",
        "Template": "",
    }
    output.update(build_output_flags([], {}))
    return output


def classify_message(
    customer_message,
    brand_message,
    stop_re,
    journeys,
    opt_in_texts,
    auto_reply,
    reaction_reply,
    common_phrases,
    opt_in_reference_messages,
    opt_out_device_not_working,
    opt_out_offensive_phrases,
    opt_out_phrases_phrases,
    keyword_rules,
    under_13_rules,
    direct_opt_out_rules,
    offensive_symbols,
    offensive_pattern_rules,
):
    if customer_message is None or pd.isna(customer_message):
        return build_no_match_output()

    raw_customer = str(customer_message).strip()
    if not raw_customer:
        return build_no_match_output()

    reasons = []
    zero_reasons = []
    triggers_by_reason = {}

    def add_reason(reason, trigger=""):
        if reason not in reasons:
            reasons.append(reason)
            triggers_by_reason[reason] = []
        if trigger:
            triggers_by_reason[reason].append(trigger)

    customer_norm = normalize_text(raw_customer)
    customer_keyword_norm = normalize_keyword_text(raw_customer)
    brand_norm = normalize_text(brand_message)
    device_not_working_exact = customer_norm in opt_out_device_not_working

    if customer_norm in common_phrases:
        zero_reasons.append("common_phrases")

    keyword_analysis = analyze_keyword_message(raw_customer, keyword_rules)
    brand_keyword_triggers = analyze_brand_keyword_combo(
        raw_customer,
        brand_message,
        keyword_rules,
        reference_messages=opt_in_reference_messages,
    )

    if stop_re:
        stop_matches = [m.group(0) for m in stop_re.finditer(raw_customer)]
        if stop_matches:
            for trigger in dict.fromkeys(stop_matches):
                add_reason("symbols", trigger)

    for symbol in offensive_symbols:
        if symbol and symbol in raw_customer:
            add_reason("offensive_symbols", symbol)

    if keyword_analysis["reason"] in {"opt_out_keywords", "typo_opt_out_keyword"}:
        for trigger in keyword_analysis["triggers"]:
            add_reason(keyword_analysis["reason"], trigger)
    elif brand_keyword_triggers:
        for reason_name in ("opt_out_keywords", "typo_opt_out_keyword"):
            for trigger in brand_keyword_triggers.get(reason_name, []):
                add_reason(reason_name, trigger)

    if customer_norm or customer_keyword_norm:
        if device_not_working_exact:
            add_reason("opt_out_device_not_working", raw_customer)

        if customer_norm:
            offensive_hits = find_full_phrase_hits(customer_norm, opt_out_offensive_phrases)
            for trigger in offensive_hits:
                add_reason("frustration", trigger)

            for pattern in offensive_pattern_rules:
                match = pattern.search(customer_norm)
                if match:
                    add_reason("frustration", match.group(0).strip())
                    break

            if not device_not_working_exact:
                opt_out_phrase_hits = find_full_phrase_hits(customer_norm, opt_out_phrases_phrases)
                for trigger in opt_out_phrase_hits:
                    add_reason("phrases", trigger)

        for _, journey_rule in journeys.items():
            customer_keyword_tokens = set(customer_keyword_norm.split()) if customer_keyword_norm else set()
            if (
                brand_norm == journey_rule["brand_message"]
                and (
                    customer_norm in journey_rule["opt_out_responses"]
                    or find_full_phrase_hits(customer_norm, journey_rule["opt_out_responses_phrases"])
                    or (
                        journey_rule.get("opt_out_numeric_tokens")
                        and journey_rule["opt_out_numeric_tokens"].intersection(customer_keyword_tokens)
                    )
                )
            ):
                add_reason("journeys", raw_customer)
                break

        if customer_norm in opt_in_texts:
            zero_reasons.append("opt_in_texts")

        if customer_norm in auto_reply:
            zero_reasons.append("auto_reply")

        if customer_norm in reaction_reply:
            zero_reasons.append("reaction_reply")

    under_13_match, under_13_trigger = is_under_13(raw_customer, under_13_rules)
    if under_13_match:
        add_reason("under_13", under_13_trigger)

    keyword_like_detected = any(reason in {"opt_out_keywords", "typo_opt_out_keyword"} for reason in reasons)
    if not keyword_like_detected and not device_not_working_exact:
        direct_matches = get_direct_opt_out_matches(raw_customer, direct_opt_out_rules)
        for match in direct_matches:
            add_reason(match["reason"], match["trigger"])

    if "journeys" in reasons:
        for reason in ["opt_out_keywords", "typo_opt_out_keyword"]:
            if reason in reasons:
                reasons.remove(reason)
                triggers_by_reason.pop(reason, None)

    zero_reasons = list(dict.fromkeys(zero_reasons))

    if reasons:
        template_text = build_template_text(reasons, triggers_by_reason)
        output = {
            "OPT_OUT": 1,
            "REASON": ", ".join(reasons),
            "Template": template_text,
        }
        output.update(build_output_flags(reasons, triggers_by_reason))
        return output

    if zero_reasons:
        output = {
            "OPT_OUT": 0,
            "REASON": ", ".join(zero_reasons),
            "Template": "",
        }
        output.update(build_output_flags([], {}))
        return output

    return build_no_match_output()


def main():
    config = load_config(CONFIG_PATH)
    english_words = load_english_words()

    keyword_config = get_config_section(config, "keyword")
    direct_opt_out_config = get_config_section(config, "direct_opt_out")
    offensive_config = get_config_section(config, "offensive")
    under_13_config = get_config_section(config, "under_13")
    non_opt_out_config = get_config_section(config, "non_opt_out")

    effective_direct_opt_out_config = dict(direct_opt_out_config)

    stop_emojis = effective_direct_opt_out_config.get("symbols", [])
    stop_re = build_stop_regex(stop_emojis)

    journeys = compile_journeys(config.get("journeys", {}))

    opt_in_texts = compile_normalized_set(non_opt_out_config.get("opt_in_texts", []))
    opt_in_reference_messages = list(non_opt_out_config.get("opt_in_texts", []))
    auto_reply = compile_normalized_set(non_opt_out_config.get("auto_reply", []))
    reaction_reply = compile_normalized_set(non_opt_out_config.get("reaction_reply", []))
    common_phrases = compile_normalized_set(non_opt_out_config.get("common_phrases", []))

    opt_out_device_not_working = compile_normalized_set(effective_direct_opt_out_config.get("device_not_working", []))
    opt_out_offensive_raw = offensive_config.get("frustration", [])
    offensive_symbols = sorted(set(offensive_config.get("symbols", [])), key=len, reverse=True)
    offensive_pattern_rules = compile_pattern_list(offensive_config.get("frustration_patterns", []))
    opt_out_phrases_raw = effective_direct_opt_out_config.get("phrases", [])
    opt_out_offensive_phrases = compile_phrase_list(opt_out_offensive_raw, normalizer=normalize_text)
    opt_out_phrases_phrases = compile_phrase_list(opt_out_phrases_raw, normalizer=normalize_text)

    keyword_rules = compile_keyword_rules(
        keyword_config.get("exact", []),
        keyword_config.get("typo", []),
        english_words,
    )

    under_13_rules = compile_under_13_rules(
        under_13_config.get("rules", {}),
        under_13_config.get("age_limit_phrases", []),
    )
    direct_opt_out_rules = compile_direct_opt_out_rules(effective_direct_opt_out_config)

    df = pd.read_csv(INPUT_CSV)

    missing = [col for col in [BRAND_COL, CUSTOMER_COL] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    results = []
    message_pairs = list(zip(df[BRAND_COL], df[CUSTOMER_COL]))
    for brand_message, customer_message in iter_with_progress(
        message_pairs,
        total=len(message_pairs),
        description="Classifying rows",
    ):
        results.append(
            classify_message(
                customer_message=customer_message,
                brand_message=brand_message,
                stop_re=stop_re,
                journeys=journeys,
                opt_in_texts=opt_in_texts,
                auto_reply=auto_reply,
                reaction_reply=reaction_reply,
                common_phrases=common_phrases,
                opt_in_reference_messages=opt_in_reference_messages,
                opt_out_device_not_working=opt_out_device_not_working,
                opt_out_offensive_phrases=opt_out_offensive_phrases,
                opt_out_phrases_phrases=opt_out_phrases_phrases,
                keyword_rules=keyword_rules,
                under_13_rules=under_13_rules,
                direct_opt_out_rules=direct_opt_out_rules,
                offensive_symbols=offensive_symbols,
                offensive_pattern_rules=offensive_pattern_rules,
            )
        )

    results_df = pd.DataFrame(results, columns=NEW_OUTPUT_COLS)
    output_df = pd.concat([df.copy(), results_df], axis=1)
    output_df = output_df[list(df.columns) + NEW_OUTPUT_COLS]

    output_df.to_csv(OUTPUT_CSV, index=False, encoding=CSV_ENCODING)
    customer_reason_col = first_present_column(
        output_df,
        [CUSTOMER_COL, "CUSTOMER_MESSAGE", "Customer Message", "customer_message"],
    )
    if not customer_reason_col:
        customer_reason_col = CUSTOMER_COL
    reasons_only_columns = [customer_reason_col, "REASON", "Template"]
    link_for_reason_col = ""
    for candidate in ["CONVERSATION_URL", "Conversation URL", "conversation_url", "LINK", "link", "Link"]:
        if candidate in output_df.columns:
            reasons_only_columns = [candidate] + reasons_only_columns
            link_for_reason_col = candidate
            break
    reasons_only_df = output_df[reasons_only_columns].copy()
    if link_for_reason_col:
        reasons_only_df["REASON"] = reasons_only_df.apply(
            lambda row: f'Conversation [{str(row[link_for_reason_col]).strip()}] {str(row["REASON"]).strip()}'
            if pd.notna(row[link_for_reason_col]) and str(row[link_for_reason_col]).strip()
            else str(row["REASON"]).strip(),
            axis=1,
        )
    reasons_only_df.to_csv(OUTPUT_REASON_CSV, index=False, encoding=CSV_ENCODING)
    agent_templates_df = build_agent_templates_output(output_df)
    agent_templates_df.to_csv(OUTPUT_AGENT_TEMPLATE_CSV, index=False, encoding=CSV_ENCODING)
    print(f"Wrote {len(output_df)} rows to {OUTPUT_CSV}")
    print(f"Wrote {len(reasons_only_df)} rows to {OUTPUT_REASON_CSV}")
    print(f"Wrote {len(agent_templates_df)} rows to {OUTPUT_AGENT_TEMPLATE_CSV}")


if __name__ == "__main__":
    main()
