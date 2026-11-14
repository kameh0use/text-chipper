import tiktoken
from text_chipper import TextChipper

FENCE = "`" * 3

DOC = f"""# Installing the client

Install the package from PyPI and create a client. The client reads its
configuration from environment variables, so no config file is needed.

## Quick example

{FENCE}python
import os
from acme import Client

client = Client(api_key=os.environ["ACME_KEY"], timeout=10)
result = client.search("chunking", limit=5)
for item in result:
    print(item.title)
{FENCE}

## Notes

Retries are enabled by default and can be turned off with retries=0.
"""

enc = tiktoken.get_encoding("cl100k_base")


def broken_fences(text: str) -> bool:
    # an odd number of fence lines means a code block was cut
    return sum(1 for l in text.splitlines() if l.lstrip().startswith(FENCE)) % 2 == 1


def plain_window(text: str, size: int) -> list[str]:
    toks = enc.encode(text)
    return [enc.decode(toks[i:i + size]) for i in range(0, len(toks), size)]


def chipper(text: str, size: int):
    c = TextChipper(max_length=size, overlap=0, encoding="cl100k_base")
    return list(c.chunk(text))


def show(title: str, items: list[tuple[str, str]]):
    # items: (shown_text, text_to_check)
    print(f"\n=== {title} ({len(items)} chunks) ===")
    for i, (shown, body) in enumerate(items, 1):
        flag = "  <-- code fence cut!" if broken_fences(body) else ""
        print(f"--- chunk {i} ({len(enc.encode(shown))} tokens){flag}")
        print(shown.rstrip())


# find the first budget where the plain window cuts a fence but text-chipper doesn't
for size in range(40, 130, 2):
    plain = plain_window(DOC, size)
    ours = chipper(DOC, size)
    plain_bad = sum(broken_fences(t) for t in plain)
    ours_bad = sum(broken_fences(c.text) for c in ours)
    if plain_bad and not ours_bad and all(c.token_count <= size for c in ours):
        print(f"max_length = {size}")
        show("tiktoken window", [(t, t) for t in plain])
        show("text-chipper", [(c.contextualized_text, c.text) for c in ours])
        break
else:
    print("No budget in range showed the difference; widen the range or edit DOC.")