"""Text cleaning rules, used from iteration 2 (notebook 5) on.

Each rule takes a Spark column of text and returns the cleaned column.
Most are Spark functions (regexp_replace, lower). Two need Python, so they
are pandas UDFs: Spark sends the text to Pandas in batches.
"""

import re

import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

# Unicode plane 14 is invisible on screen. Unicode tags (U+E0041 = hidden "A")
# live there, and some attackers shift whole texts into it.
PLANE_14 = re.compile("[\U000E0000-\U000EFFFF]+")

# Zero-width characters, soft hyphen, and the marks that change text direction.
INVISIBLE = r"[\x{00AD}\x{180E}\x{200B}-\x{200F}\x{202A}-\x{202E}\x{2060}-\x{2064}\x{2066}-\x{2069}\x{FEFF}]"
# Control characters, except tab, new line and carriage return.
CONTROL = r"[\x{00}-\x{08}\x{0B}\x{0C}\x{0E}-\x{1F}\x{7F}-\x{9F}]"

# The challenge's target address, also written to dodge filters:
# "contact at contact.com", "contact[at]contact[dot]com", "contact@contact,com".
TARGET_ADDRESS = r"(?i)contact\s*(?:@|\[at\]|\(at\)|\{at\}|\bat\b)\s*contact\s*(?:\.|,|\[dot\]|\(dot\)|\{dot\}|\bdot\b)\s*com"
# (?<!...) starts a match only at the beginning of a word and ++ never goes back:
# without them the regex is very slow on long texts with no spaces (Chinese, glued words).
EMAIL_ADDRESS = r"(?<![\p{L}\p{N}._%+-])[\p{L}\p{N}._%+-]++@[\p{L}\p{N}-]++(?:\.[\p{L}\p{N}-]++)+"
# "RE:", "FW:", "Fwd:" added by the mail software at the start of the subject.
REPLY_PREFIX = r"(?i)^(Subject of the email: )(?:(?:re|fwd?)\s*:\s*)+"


def _shift_back(match):
    chars = (chr(ord(c) - 0xE0000) for c in match.group())
    return " " + "".join(c for c in chars if c.isprintable()) + " "


@F.pandas_udf(StringType())
def reveal_hidden(text: pd.Series) -> pd.Series:
    """Plane 14 text shifted back to normal characters, so the model reads it."""
    return text.str.replace(PLANE_14, _shift_back, regex=True)


@F.pandas_udf(StringType())
def nfkc(text: pd.Series) -> pd.Series:
    """Unicode NFKC: ideographic space -> space, full-width letters -> letters, ﬁ -> fi."""
    return text.str.normalize("NFKC")


def remove_invisible(text):
    return F.regexp_replace(F.regexp_replace(text, INVISIBLE, ""), CONTROL, " ")


def remove_reply_prefix(text):
    return F.regexp_replace(text, REPLY_PREFIX, "$1")


def mask_addresses(text):
    # We keep the fact that there is an address, not which one.
    text = F.regexp_replace(text, TARGET_ADDRESS, " emailaddress ")
    return F.regexp_replace(text, EMAIL_ADDRESS, " emailaddress ")


def remove_word(pattern):
    return lambda text: F.regexp_replace(text, "(?i)" + pattern, " ")


def split_punctuation(text):
    # Punctuation becomes separate words: "confirmation." -> "confirmation ."
    # We keep it, because attacks use symbols like <| , ### or {}.
    return F.regexp_replace(text, r"([^\p{L}\p{M}\p{N}\s])", " $1 ")


def collapse_spaces(text):
    return F.trim(F.regexp_replace(text, r"[\s\p{Z}]+", " "))


# A: generic cleaning, the same for any text.
HIDDEN_TEXT = [
    ("reveal hidden text", reveal_hidden),
    ("Unicode NFKC", nfkc),
    ("remove invisible characters", remove_invisible),
]
# B only: shortcuts of our data, found in notebooks 2 to 4.
SHORTCUTS = [
    ("remove RE: / FW:", remove_reply_prefix),
    ("mask email addresses", mask_addresses),
    ('remove "enron"', remove_word("enron")),
    ('remove "confirmation"', remove_word("confirmations?")),
]
FINAL = [
    ("lowercase", F.lower),
    ("split punctuation", split_punctuation),
    ("collapse spaces", collapse_spaces),
]


def cleaning_steps(remove_shortcuts=False):
    """The rules in order, as (name, function)."""
    return HIDDEN_TEXT + (SHORTCUTS if remove_shortcuts else []) + FINAL


def clean(text, remove_shortcuts=False):
    for _, step in cleaning_steps(remove_shortcuts):
        text = step(text)
    return text
