"""Turn results.jsonl (from streaming_bench.py) into a Markdown table.

Time, throughput, memory and CPU come from runs WITHOUT --verify (verification builds extra
per-chunk objects and would inflate memory). Chunk counts, largest chunk and over-limit counts
come from the --verify run of the same mode.
"""
import json
import statistics as st
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
keys = list(dict.fromkeys((r.get("lane", "?"), r["mode"]) for r in rows))

print("| Lane | Mode | Runs | Time (s) | MiB/s | Peak memory growth (MB) | CPU / wall | Chunks | Largest chunk (tokens) | Over limit | Result |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for lane, mode in keys:
    mine = [r for r in rows if r.get("lane", "?") == lane and r["mode"] == mode]
    bad = [r for r in mine if r.get("status") != "ok"]
    timing = [r for r in mine if r.get("status") == "ok" and not r["verified"]]
    ver = next((r for r in mine if r.get("status") == "ok" and r["verified"]), None)
    if not timing:
        code = bad[0].get("exit_code") if bad else None
        why = {124: "timed out", 137: "killed (out of memory?)", None: "no timed run"}.get(code, f"failed (exit {code})")
        print(f"| {lane} | {mode} | 0 | | | | | | | | {why} |")
        continue
    med = lambda k: st.median(r[k] for r in timing)
    cpu_wall = st.median(r["cpu_s"] / r["time_s"] for r in timing if r.get("cpu_s"))
    note = "ok" if not bad else "ok (a later run failed)"
    print(
        f"| {lane} | {mode} | {len(timing)} | {med('time_s'):.1f} | {med('mib_per_s'):.1f} | "
        f"{med('growth_mb'):.0f} | {cpu_wall:.1f} | {timing[0]['chunks']:,} | "
        f"{ver['largest_chunk_tokens'] if ver else 'n/a'} | "
        f"{ver['chunks_over_limit'] if ver else 'n/a'} | {note} |"
    )
