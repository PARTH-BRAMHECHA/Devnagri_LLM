"""
Shared error-diagnostics helper.

Stage 6 and human_eval are both launched via `subprocess.call([sys.executable,
"-m", ...])` from the notebook. Two things can make a subprocess failure show
up as nothing but "exited with code 1" in the notebook, with the actual
traceback missing or truncated above it:

1. Python's stdout is line-buffered by default when it's not attached to a
   real terminal (which is exactly the situation inside a subprocess whose
   output is being relayed through Jupyter) -- if the process dies abruptly,
   whatever was still sitting in that buffer can be lost.
2. A raw `torch.cuda.OutOfMemoryError` traceback doesn't say "reduce
   batch size" or "free X GB" in a way that's obvious at a glance if you're
   skimming a long stack trace.

`run_with_diagnostics` wraps a stage's entry point so that on any exception
it: force-flushes stdout/stderr, prints the full traceback explicitly (not
just relying on the interpreter's default unhandled-exception printer), and
-- if it looks CUDA/memory related -- prints `torch.cuda.memory_summary()`
plus a one-line hint, before re-raising so the process still exits non-zero.
"""

import sys
import traceback


def run_with_diagnostics(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        sys.stdout.flush()
        sys.stderr.flush()
        print("\n" + "=" * 70, file=sys.stderr)
        print("  UNHANDLED EXCEPTION -- full traceback below", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        traceback.print_exc()

        msg = str(e).lower()
        if "cuda" in msg or "memory" in type(e).__name__.lower() or "out of memory" in msg:
            print("\n  Looks CUDA/memory related. GPU memory summary:", file=sys.stderr)
            try:
                import torch
                if torch.cuda.is_available():
                    print(torch.cuda.memory_summary(), file=sys.stderr)
                    print(
                        "\n  Hint: this stage loads a FRESH copy of the fine-tuned "
                        "7B model in its own process (no sharing with Stage 3/4). "
                        "If this is a T4 16GB and it's OOMing, try: smaller "
                        "--n-train/--n-eval (stage6_downstream), fewer "
                        "--n-samples (human_eval), or confirm no other process "
                        "still holds GPU memory (`nvidia-smi` in a terminal cell).",
                        file=sys.stderr,
                    )
                else:
                    print("  torch.cuda.is_available() is False -- not a GPU memory "
                          "issue, something else raised this.", file=sys.stderr)
            except Exception as diag_err:
                print(f"  (couldn't gather CUDA diagnostics: {diag_err})", file=sys.stderr)

        print("=" * 70 + "\n", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
        raise
