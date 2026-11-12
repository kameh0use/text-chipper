"""Stub contestants for the self-tests: no third-party packages needed."""
import re

from chunkbench.adapters import Adapter, Runner, View, register_test_adapter

_TOK = re.compile(r"\w+|[^\w\s]")


def _window(size):
    def run(text):
        toks = list(_TOK.finditer(text))
        out = []
        for i in range(0, len(toks), size):
            grp = toks[i : i + size]
            out.append(text[grp[0].start() : grp[-1].end()])
        return out

    return run


def _views(raw):
    return [View(s, s) for s in raw]


register_test_adapter(Adapter("stub_window", "stub window", None, "", "exact token windows", False,
                              lambda ctx: Runner(_window(ctx.chunk_size), _views)))


def _chars(ctx):
    # Cuts every 12*chunk_size characters: ignores word boundaries and over-runs the budget.
    n = ctx.chunk_size * 12
    return Runner(lambda t: [t[i : i + n] for i in range(0, len(t), n)], _views)


register_test_adapter(Adapter("stub_chars", "stub chars", None, "", "char cuts", False, _chars))


def _lossy(ctx):
    inner = _window(ctx.chunk_size)
    return Runner(lambda t: [c for i, c in enumerate(inner(t)) if i % 5 != 4], _views)


register_test_adapter(Adapter("stub_lossy", "stub lossy", None, "", "drops every fifth chunk", False, _lossy))
