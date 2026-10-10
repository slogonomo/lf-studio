#!/usr/bin/env python3
"""Summarize py-spy raw (collapsed) output: category totals + top inclusive frames. usage: pyspy_summary.py FILE [N]

Record a worker first, for example (py-spy needs root or ptrace permission to attach to another process):
    py-spy record --rate 20 --format raw --subprocesses -o stage1.txt --pid <stage1 worker pid>
Times assume the 20 Hz sample rate above.
"""
import re, sys, collections
path = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
CATS = [("dynamo trace", r"torch/_dynamo/(symbolic_convert|output_graph|convert_frame)"),
        ("aot/inductor compile", r"torch/_(functorch/_aot_autograd|inductor/(compile_fx|graph|codecache|scheduler|lowering))"),
        ("inductor autotune", r"(select_algorithm|autotune)"),
        ("triton compile/load", r"triton/(compiler|runtime/(jit|build))"),
        ("cute/cutlass dsl", r"(cutlass|cute)"),
        ("safetensors load", r"(safetensors|weight_utils|load_model_from_full)"),
        ("lora", r"lora"),
        ("fp8 convert", r"fp8"),
        ("lookup", r"lookup\.py"),
        ("forward compute", r"(minimax_h3\.py|transformer|ltx_core)")]
total = 0; cats = collections.Counter(); incl = collections.Counter()
for line in open(path):
    line = line.rstrip("\n")
    m = re.match(r"(.*) (\d+)$", line)
    if not m: continue
    stack, n = m.group(1), int(m.group(2)); total += n
    for name, pat in CATS:
        if re.search(pat, stack):
            cats[name] += n; break
    else:
        cats["other"] += n
    seen = set()
    for fr in stack.split(";"):
        fr = re.sub(r":\d+\)", ")", fr)
        if fr not in seen:
            seen.add(fr); incl[fr] += n
print(f"{path}: {total} samples (20 Hz -> ~{total/20:.0f} thread-s)")
for name, n in cats.most_common(): print(f"  {name:22s} {n/20:7.1f} s  {100*n/max(total,1):5.1f}%")
print("top inclusive frames:")
for fr, n in incl.most_common(N): print(f"  {n/20:7.1f} s  {fr[:150]}")
