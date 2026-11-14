from __future__ import annotations


def harmonic_mean(a: float, b: float) -> float:
    return 2 * a * b / (a + b) if a + b else 0.0


def format_metrics(m: dict) -> str:
    base = f"top1 {m['top1']:6.2f} | top5 {m['top5']:6.2f} | n={m['n']}"
    if "field_top1" in m:
        base += f" | field top1 {m['field_top1']:6.2f} top5 {m['field_top5']:6.2f}"
    return base
