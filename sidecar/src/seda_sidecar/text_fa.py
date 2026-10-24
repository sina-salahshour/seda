"""Persian text normalisation for TTS (shared by the OmniVoice and Pocket engines).

normalize_fa_text(text) folds Arabic letter variants onto Persian ones, spells
every number out in Persian words (integers, decimals, negatives, percentages,
times, ordinals, ranges, digit strings), tidies ZWNJ, punctuation, quotes and
whitespace, and leaves Latin words alone (engines decide what to do with them).

The number reading follows the conventions of normalize_fa.py from
github.com/mallahyari/pocket-tts (training/farsi, MIT licence): "هزار" without
"یک", "یک میلیون", " و " between groups, long or zero-led digit strings read
one digit at a time.

Self-test: `python -m seda_sidecar.text_fa`.
"""
from __future__ import annotations

import re
import unicodedata

ZWNJ = "‌"
PERSIAN_LETTERS = "آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیئءأؤ"

CHAR_MAP = str.maketrans(
    {
        "ي": "ی",  # Arabic yeh
        "ى": "ی",  # alef maksura
        "ۍ": "ی",
        "ې": "ی",
        "ك": "ک",  # Arabic kaf
        "ڪ": "ک",
        "ة": "ه",  # teh marbuta
        "ۀ": "ه",  # heh with yeh above (ezafe is predicted by the engines)
        "ۂ": "ه",
        "ۃ": "ه",
        "ٱ": "ا",
        "ٲ": "ا",
        "ٳ": "ا",
        "ـ": "",  # tatweel
    }
)
DIACRITICS = re.compile("[ً-ٰٟۖ-ۭ]")
INVISIBLES = re.compile("[​‍‎‏‪-‮⁦-⁩﻿­]")
DIGITS = str.maketrans(
    {**{chr(0x06F0 + i): str(i) for i in range(10)}, **{chr(0x0660 + i): str(i) for i in range(10)}}
)

# ------------------------------------------------------------------ numbers
_ONES = ["", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه"]
_TEENS = ["ده", "یازده", "دوازده", "سیزده", "چهارده", "پانزده", "شانزده", "هفده", "هجده", "نوزده"]
_TENS = ["", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود"]
_HUNDREDS = ["", "صد", "دویست", "سیصد", "چهارصد", "پانصد", "ششصد", "هفتصد", "هشتصد", "نهصد"]
_SCALES = [(10**12, "تریلیون"), (10**9, "میلیارد"), (10**6, "میلیون"), (10**3, "هزار")]
MAX_NUMBER_DIGITS = 15


def _under_thousand(n: int) -> str:
    parts = []
    if n >= 100:
        parts.append(_HUNDREDS[n // 100])
        n %= 100
    if 10 <= n < 20:
        parts.append(_TEENS[n - 10])
        n = 0
    elif n >= 20:
        parts.append(_TENS[n // 10])
        n %= 10
    if n > 0:
        parts.append(_ONES[n])
    return " و ".join(parts)


def number_to_words(n: int) -> str:
    """Non-negative integer -> Persian words (1402 -> "هزار و چهارصد و دو")."""
    if n == 0:
        return "صفر"
    parts = []
    for value, name in _SCALES:
        if n >= value:
            count, n = divmod(n, value)
            head = "" if (count == 1 and value == 1000) else _under_thousand(count) + " "
            parts.append(f"{head}{name}")
    if n:
        parts.append(_under_thousand(n))
    return " و ".join(parts)


def digits_one_by_one(digits: str) -> str:
    return " ".join("صفر" if d == "0" else _ONES[int(d)] for d in digits if d.isdigit())


def ordinal(words: str) -> str:
    """"بیست و سه" -> "بیست و سوم", "سی" -> "سی‌ام", "یک" -> "یکم"."""
    if words.endswith("سه"):
        return words[:-2] + "سوم"
    if words.endswith("ی"):
        return words + ZWNJ + "ام"
    return words + "م"


def _int_words(s: str) -> str:
    if len(s) > MAX_NUMBER_DIGITS or (len(s) > 1 and s.startswith("0")):
        return digits_one_by_one(s)
    return number_to_words(int(s))


def _decimal_words(whole: str, frac: str | None) -> str:
    words = _int_words(whole)
    if frac:
        f = frac.rstrip("0")
        if f:
            if f.startswith("0") or len(f) > 3:
                words += " ممیز " + digits_one_by_one(f)
            else:
                words += " ممیز " + number_to_words(int(f))
    return words


def _sp(s: str) -> str:
    return f" {s} "


_NUM = r"\d+(?:\.\d+)?"


def numbers_to_words(text: str) -> str:
    """Every ASCII digit run in `text` (digits already folded to ASCII) -> words."""
    # thousands separators: 1,234,567 / 1٬234
    text = re.sub(r"(?<=\d)[,٬](?=\d{3}(?!\d))", "", text)
    text = text.replace("٫", ".")
    # times hh:mm(:ss)
    def time(m: re.Match) -> str:
        h, mi, s = int(m.group(1)), int(m.group(2)), m.group(3)
        out = number_to_words(h)
        if mi:
            out += " و " + number_to_words(mi) + " دقیقه"
        if s and int(s):
            out += " و " + number_to_words(int(s)) + " ثانیه"
        return _sp(out)

    text = re.sub(r"(?<![\d.])(\d{1,2}):(\d{2})(?::(\d{2}))?(?![\d])", time, text)
    # percent: 50% / 50٪ / ٪50
    text = re.sub(rf"({_NUM})\s*[%٪]", lambda m: _sp(_num_token(m.group(1)) + " درصد"), text)
    text = re.sub(rf"[%٪]\s*({_NUM})", lambda m: _sp(_num_token(m.group(1)) + " درصد"), text)
    # ranges 10-20 -> ده تا بیست
    text = re.sub(
        rf"(?<![\d.])({_NUM})\s*[-–—]\s*({_NUM})(?![\d.])",
        lambda m: _sp(_num_token(m.group(1)) + " تا " + _num_token(m.group(2))),
        text,
    )
    # negatives: a minus sign at a word start
    text = re.sub(
        rf"(^|[\s(\[«:=])[-−–]({_NUM})",
        lambda m: m.group(1) + _sp("منفی " + _num_token(m.group(2))),
        text,
    )
    # ordinals: ۳م، ۲۰ام، 21st
    text = re.sub(
        r"(\d+)\s*(?:ام|م|st|nd|rd|th)(?![\w؀-ۿ])",
        lambda m: _sp(ordinal(_int_words(m.group(1)))),
        text,
    )
    # dates 1402/05/12 and similar: read the parts in order
    text = re.sub(
        r"(?<![\d.])(\d{2,4})[/\-.](\d{1,2})[/\-.](\d{1,4})(?![\d.])",
        lambda m: _sp(" ".join(number_to_words(int(g)) for g in m.groups())),
        text,
    )
    text = re.sub(r"(?<=\d)[/](?=\d)", " ", text)
    return re.sub(_NUM, lambda m: _sp(_num_token(m.group(0))), text)


def _num_token(s: str) -> str:
    whole, _, frac = s.partition(".")
    return _decimal_words(whole, frac or None)


# ------------------------------------------------------------------ punctuation
PUNCT_MAP = str.maketrans(
    {
        "?": "؟",
        ";": "؛",
        "…": ".",
        "«": " ",
        "»": " ",
        "“": " ",
        "”": " ",
        "„": " ",
        '"': " ",
        "‘": " ",
        "’": "'",
        "`": " ",
        "(": "، ",
        ")": "، ",
        "[": "، ",
        "]": "، ",
        "{": " ",
        "}": " ",
        "–": "، ",
        "—": "، ",
        "*": " ",
        "-": " ",
        "−": " ",
        "‐": " ",
        "/": " ",
        "\\": " ",
        "_": " ",
        "|": " ",
        "#": " ",
        "~": " ",
        "^": " ",
        "<": " ",
        ">": " ",
        "\t": " ",
        "\r": " ",
        "\n": " ",
    }
)
_PUNCT = ".،؛؟!:"

_PLURAL = "ها|های|هایی|هایم|هایت|هایش|هایمان|هایتان|هایشان|هاست|هام|هات|هاش"
_TARIN = "تر|ترین|تری|ترها"


def zwnj_cleanup(text: str) -> str:
    text = re.sub(f"{ZWNJ}+", ZWNJ, text)
    text = re.sub(rf" *{ZWNJ} *", lambda m: ZWNJ if m.group(0) == ZWNJ else " ", text)
    p = PERSIAN_LETTERS
    text = re.sub(rf"(?<![{p}]){ZWNJ}|{ZWNJ}(?![{p}])", "", text)
    # "می رود" -> "می‌رود", "نمی دانم" -> "نمی‌دانم" (verb prefix, next word >= 2 letters)
    text = re.sub(rf"(?<![{p}\w])(ن?می) (?=[{p}]{{2,}})", rf"\1{ZWNJ}", text)
    # "کتاب ها" -> "کتاب‌ها"  (plural suffix written as a separate word)
    text = re.sub(rf"(?<=[{p}]) ({_PLURAL})(?![{p}{ZWNJ}])", rf"{ZWNJ}\1", text)
    # "بزرگ ترین" -> "بزرگ‌ترین"
    text = re.sub(rf"(?<=[{p}]{{2}}) ({_TARIN})(?![{p}{ZWNJ}])", rf"{ZWNJ}\1", text)
    return text


def normalize_fa_text(text: str, strip_diacritics: bool = True) -> str:
    """General Persian TTS normalisation. Latin words are kept as they are."""
    text = unicodedata.normalize("NFC", text or "")
    text = INVISIBLES.sub("", text)
    if strip_diacritics:
        text = DIACRITICS.sub("", text)
    text = text.translate(CHAR_MAP).translate(DIGITS)
    # Latin "," inside numbers is handled by numbers_to_words; elsewhere -> "،"
    text = numbers_to_words(text)
    text = re.sub(r"(?<!\d),|,(?!\d)", "،", text)
    text = text.replace("&", " و ").replace("+", " به علاوه ").replace("=", " مساوی ")
    text = text.translate(PUNCT_MAP)
    text = re.sub(r"\.{2,}", ".", text)
    # drop emoji and other symbols the voice cannot say
    text = "".join(
        c for c in text if not unicodedata.category(c).startswith(("S", "C")) or c == ZWNJ
    )
    text = zwnj_cleanup(text)
    # punctuation spacing
    text = re.sub(rf"\s+([{_PUNCT}])", r"\1", text)
    text = re.sub(r"[،؛:]\s*([.!؟])", r"\1", text)
    text = re.sub(r"([،؛])(?:\s*[،؛])+", r"\1", text)
    text = re.sub(r"([.!؟])\1+", r"\1", text)
    text = re.sub(rf"([{_PUNCT}])(?=[^\s{_PUNCT}\d])", r"\1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # RTL text often arrives with leading punctuation / a dangling comma
    text = text.lstrip(_PUNCT + "، ").rstrip("،؛ ")
    return text


_EXAMPLES = [
    "سلام! حالت چطوره؟؟ ساعت ۱۰:۳۰ جلسه داریم.",
    "قیمت ۳.۵ میلیون تومان است و ۲۵٪ تخفیف دارد.",
    "دما به -۵ درجه رسید؛ در سال ۱۴۰۲ اینطور نبود.",
    "شماره من ۰۹۱۲۳۴۵۶۷۸۹ است و 1,250,000 ریال بدهکارم.",
    "كتاب هاي جديد را مي خواهم ، لطفا  ...",
    "من از Python و GPT-4 استفاده می‌کنم (برای کار).",
    "فصل ۳م و صفحه‌ی ۲۰ام را بخوان؛ ۱۰-۱۵ دقیقه وقت داری.",
    "«این بزرگ ترین   پروژه‌ ‌است» گفت.",
    "تاریخ ۱۴۰۳/۰۵/۱۲ ثبت شد. ۱۲۳۴۵۶۷",
]

if __name__ == "__main__":
    for ex in _EXAMPLES:
        print(ex)
        print("  ->", normalize_fa_text(ex))
