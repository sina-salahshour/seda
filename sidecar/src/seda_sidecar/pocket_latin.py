"""Latin words -> rough Persian script, for the pocket_v2 engine.

pocket-tts-farsi-v2 reads phonemes produced by a Persian G2P, and the model's
normaliser deletes every Latin character, so "من از Python استفاده می‌کنم" would
silently lose the word. We respell Latin words in Persian letters first:
a small lexicon of common names/tech words, letter names for acronyms, and a
crude grapheme mapping for everything else. It is approximate by design; the
G2P model then picks a pronunciation for the Persian spelling.
"""
from __future__ import annotations

import re

LETTER_NAMES = {
    "a": "ای", "b": "بی", "c": "سی", "d": "دی", "e": "ای", "f": "اف", "g": "جی",
    "h": "اچ", "i": "آی", "j": "جی", "k": "کی", "l": "ال", "m": "ام", "n": "ان",
    "o": "او", "p": "پی", "q": "کیو", "r": "آر", "s": "اس", "t": "تی", "u": "یو",
    "v": "وی", "w": "دبلیو", "x": "اکس", "y": "وای", "z": "زد",
}

LEXICON = {
    "ok": "اوکی", "okay": "اوکی", "hello": "هلو", "hi": "های", "bye": "بای",
    "thanks": "تنکس", "please": "پلیز", "yes": "یس", "no": "نو",
    "google": "گوگل", "apple": "اپل", "microsoft": "مایکروسافت", "windows": "ویندوز",
    "linux": "لینوکس", "arch": "آرچ", "ubuntu": "اوبونتو", "android": "اندروید",
    "iphone": "آیفون", "python": "پایتون", "java": "جاوا", "javascript": "جاوااسکریپت",
    "rust": "راست", "github": "گیت‌هاب", "git": "گیت", "docker": "داکر",
    "claude": "کلاد", "anthropic": "آنتروپیک", "openai": "اوپن ای آی", "chatgpt": "چت جی پی تی",
    "ollama": "اولاما", "llama": "لاما", "qwen": "کوئن", "gemini": "جمینای",
    "code": "کد", "email": "ایمیل", "online": "آنلاین", "offline": "آفلاین",
    "internet": "اینترنت", "web": "وب", "app": "اپ", "server": "سرور", "file": "فایل",
    "data": "دیتا", "model": "مدل", "token": "توکن", "prompt": "پرامپت", "chat": "چت",
    "bug": "باگ", "commit": "کامیت", "push": "پوش", "update": "آپدیت", "download": "دانلود",
    "upload": "آپلود", "browser": "بروزر", "chrome": "کروم", "firefox": "فایرفاکس",
    "youtube": "یوتیوب", "instagram": "اینستاگرام", "telegram": "تلگرام", "whatsapp": "واتس‌اپ",
    "twitter": "توییتر", "facebook": "فیسبوک", "amazon": "آمازون", "nvidia": "انویدیا",
    "intel": "اینتل", "amd": "ای ام دی", "cpu": "سی پی یو", "gpu": "جی پی یو",
    "wifi": "وای فای", "bluetooth": "بلوتوث", "usb": "یو اس بی", "pdf": "پی دی اف",
    "the": "دِ", "and": "اند", "of": "آو", "to": "تو", "for": "فور", "with": "ویت",
    "is": "ایز", "new": "نیو", "pro": "پرو", "max": "مکس", "mini": "مینی", "plus": "پلاس",
    "ai": "ای آی", "api": "ای پی آی", "tts": "تی تی اس", "ui": "یو آی",
    "seda": "صدا", "hyprland": "هایپرلند", "tehran": "تهران", "iran": "ایران",
}

_DIGRAPHS = [
    ("tion", "شن"), ("sh", "ش"), ("ch", "چ"), ("th", "ت"), ("ph", "ف"), ("kh", "خ"),
    ("gh", "ق"), ("zh", "ژ"), ("ck", "ک"), ("qu", "کو"), ("oo", "و"), ("ee", "ی"),
    ("ea", "ی"), ("ou", "و"), ("ai", "ی"), ("ay", "ی"), ("ey", "ی"), ("ie", "ی"),
    ("ow", "او"), ("aw", "ا"), ("ng", "نگ"),
]
_INITIAL = {"oo": "او", "ou": "او", "ow": "او", "ee": "ای", "ea": "ای", "ie": "ای",
            "ai": "ای", "ay": "ای", "ey": "ای", "aw": "آ"}
_CONS = {
    "b": "ب", "d": "د", "f": "ف", "g": "گ", "h": "ه", "j": "ج", "k": "ک", "l": "ل",
    "m": "م", "n": "ن", "p": "پ", "q": "ک", "r": "ر", "s": "س", "t": "ت", "v": "و",
    "w": "و", "x": "کس", "z": "ز",
}


def _graphemes(word: str) -> str:
    w = word.lower()
    out: list[str] = []
    i = 0
    n = len(w)
    while i < n:
        for dg, fa in _DIGRAPHS:
            if w.startswith(dg, i):
                if i == 0 and dg in _INITIAL:
                    out.append(_INITIAL[dg])
                else:
                    out.append(fa)
                i += len(dg)
                break
        else:
            c = w[i]
            first, last = i == 0, i == n - 1
            nxt = w[i + 1] if i + 1 < n else ""
            if c == "c":
                out.append("س" if nxt in ("e", "i", "y") else "ک")
            elif c == "y":
                out.append("ی")
            elif c in _CONS:
                out.append(_CONS[c])
            elif c == "a":
                out.append("آ" if first else "ا")
            elif c == "e":
                if first:
                    out.append("ا")
                elif last:
                    pass  # silent final e
                # medial e: short vowel, unwritten
            elif c == "i":
                out.append("ای" if first else "ی")
            elif c in ("o", "u"):
                out.append("او" if first else "و")
            i += 1
    s = "".join(out)
    return s or word


def latin_word_to_persian(word: str) -> str:
    low = word.lower()
    if low in LEXICON:
        return LEXICON[low]
    letters = re.sub(r"[^A-Za-z]", "", word)
    if not letters:
        return word
    # acronyms: all caps (GPT, NASA-like short ones are spelled out) or no vowels
    if (letters.isupper() and len(letters) <= 5) or not re.search(r"[aeiouy]", low):
        return " ".join(LETTER_NAMES[c] for c in low if c in LETTER_NAMES)
    if re.search(r"[a-z][A-Z]", word):  # camelCase -> pieces
        return " ".join(latin_word_to_persian(p) for p in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", word))
    fa = _graphemes(letters)
    if re.match(r"s[tpkcmnlw]", low):  # "stream" -> "استریم", as Persian speakers say it
        fa = "ا" + fa
    return fa


def latin_to_persian(text: str) -> str:
    """Respell every Latin word in `text` with Persian letters."""
    return re.sub(r"[A-Za-z]+(?:'[A-Za-z]+)?", lambda m: latin_word_to_persian(m.group(0).replace("'", "")), text)


if __name__ == "__main__":
    for w in ["Python", "GPT", "Claude", "hello world", "Hyprland", "machine learning",
              "OpenAI", "Ubuntu", "computer", "Sina", "NVIDIA", "streaming", "GitHub"]:
        print(w, "->", latin_to_persian(w))
