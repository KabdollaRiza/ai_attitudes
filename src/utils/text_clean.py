import re

URL_RE = re.compile(r"http[s]?://\S+")
SPACE_RE = re.compile(r"\s+")

def basic_clean(text: str) -> str:
    """Remove URLs, newlines, and extra spaces."""
    t = text or ""
    t = URL_RE.sub(" ", t)
    t = t.replace("\n", " ")
    t = SPACE_RE.sub(" ", t)
    return t.strip()
