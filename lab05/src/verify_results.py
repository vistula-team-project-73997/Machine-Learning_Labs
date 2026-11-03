"""Verify persisted evidence and reload the traced model in a fresh process."""
import csv
import json
import statistics
from pathlib import Path
import numpy as np
import torch
from experiments import torch_model, arrays_from_flat

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def js(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def rows(name):
    with (R / name).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def verify():
    g = rows("gradient_check.csv")
    assert len(g) == 24
    for row in g:
        assert abs(float(row["engine"])-float(row["pytorch"])) < 1e-6
        assert abs(float(row["engine"])-float(row["numerical"])) < 1e-6
    runs, summary = rows("training_runs.csv"), js("training_summary.json")
    assert len(runs) == 10
    for name in ("scalar", "pytorch"):
        times = [float(r["seconds"]) for r in runs if r["engine"] == name]
        assert len(times) == 5 and all(t > 0 for t in times)
        assert statistics.median(times) == summary[f"{name}_median_seconds"]
    weights = np.load(R / "trained_weights.npz")
    np.testing.assert_allclose(weights["scalar"], weights["pytorch"], atol=1e-9, rtol=0)
    model = torch_model(arrays_from_flat(weights["pytorch"])).eval()
    traced = torch.jit.load(str(R / "tiny_model_traced.pt")).eval()
    data = np.load(R / "moons_split.npz")
    with torch.inference_mode():
        for partition in ("train", "test"):
            x = torch.from_numpy(data[f"x_{partition}"])
            torch.testing.assert_close(model(x), traced(x), rtol=1e-10, atol=1e-12)
        p = model(torch.from_numpy(data["x_train"]))
        loss = torch.nn.functional.binary_cross_entropy(p, torch.from_numpy(data["y_train"][:, None])).item()
        assert abs(loss-summary["final_bce"]) < 1e-12
    execution, raw = js("execution_summary.json"), rows("execution_timings.csv")
    for batch in execution["results"]:
        for name, median in batch["median_seconds"].items():
            times = [float(r["seconds_per_call"]) for r in raw
                     if r["group"] == f"batch_{batch['batch']}" and r["variant"] == name]
            assert len(times) == 7 and statistics.median(times) == median
    layout, raw = js("contiguity_summary.json"), rows("contiguity_timings.csv")
    for name, median in layout["median_seconds"].items():
        times = [float(r["seconds_per_call"]) for r in raw if r["variant"] == name]
        assert len(times) == 7 and statistics.median(times) == median
    original = ROOT / "references" / "Lab Manual_ Labs 05.pdf"
    assert original.is_file()
    print("PASS: saved gradients, five-run medians, matching trained weights, BCE, fresh-process trace reload, execution timings and tensor-layout timings.")


if __name__ == "__main__":
    verify()
