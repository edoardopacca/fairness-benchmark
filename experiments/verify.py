"""Verify persisted evidence, pairing, checkpoint replay and validation-only selection."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import make_data
from .metrics import basic, group_statistics
from .run import jobs, run_id, source_hash
from .upstream import provenance


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    out = args.out
    manifest = json.loads((out / "manifest.json").read_text())
    identity = manifest["identity"]
    config = identity["config"]
    assert identity["source_sha256"] == source_hash(), "Source changed since experiment"
    assert identity["upstream"] == provenance(), "Upstream changed since experiment"
    torch.set_num_threads(1)
    frame = pd.read_csv(out / "results.csv")
    assert len(frame) == len(list(jobs(config)))
    assert set(frame.status) == {"ok"}
    assert frame.run_id.nunique() == len(frame)
    reference, replayed = {}, set()
    for job in jobs(config):
        path = out / "runs" / run_id(job)
        result = json.loads((path / "result.json").read_text())
        key = (job["scenario"], job["seed"])
        if key not in reference:
            reference[key] = result["data"]["sha256"]
        assert result["data"]["sha256"] == reference[key]
        with np.load(path / "predictions.npz") as arrays:
            ids = np.concatenate([arrays[f"{split}_ids"] for split in config["sizes"]])
            assert len(np.unique(ids)) == sum(config["sizes"].values())
            assert all(np.isfinite(arrays[f"{split}_p"]).all() for split in config["sizes"])
            # Replay one checkpoint for every scenario/method/batch, including every data seed elsewhere.
            replay_key = (job["scenario"], job["method"], job["batch_size"])
            if replay_key not in replayed:
                checkpoint = torch.load(path / "checkpoint.pt", weights_only=False, map_location="cpu")
                parts, info = make_data(job["seed"], job["scenario"], config["sizes"])
                assert info == result["data"]
                layers, dim = [], 4
                for width in config["hidden"]:
                    layers.extend([nn.Linear(dim, width), nn.ReLU()])
                    dim = width
                layers.append(nn.Linear(dim, 1))
                model = nn.Sequential(*layers)
                model.load_state_dict(checkpoint["model"])
                with torch.no_grad():
                    p = model(torch.from_numpy(parts["test"]["x"])).sigmoid().numpy().reshape(-1)
                np.testing.assert_array_equal(p, arrays["test_p"])
                values = basic(arrays["test_s"], arrays["test_y"], p, max_n=config["eval_max_n"],
                               permutations=config["permutations"])
                for name, value in values.items():
                    np.testing.assert_allclose(value, result[f"test_{name}"], atol=1e-12)
                partitions = json.loads((path / "partitions.json").read_text())
                for name, boundaries in partitions.items():
                    stats, _ = group_statistics(arrays["test_s"], p, boundaries)
                    for metric, value in stats.items():
                        np.testing.assert_allclose(value, result[f"test_{name}_{metric}"], atol=1e-12)
                replayed.add(replay_key)
    # Reconstruct all selected lambdas using ONLY validation columns.
    keys = ["scenario", "method", "batch_size", "lambda"]
    candidates = frame.groupby(keys).agg(accuracy=("validation_accuracy", "mean"),
                                        dcor=("validation_dcor", "mean")).reset_index()
    selection = pd.read_csv(out / "validation_selection.csv")
    for key, group in candidates.groupby(keys[:-1]):
        eligible = group[group.dcor <= config["selection_dcor_limit"]]
        best = (eligible.sort_values(["accuracy", "dcor", "lambda"], ascending=[False, True, True])
                if len(eligible) else group.sort_values(["dcor", "accuracy", "lambda"], ascending=[True, False, True])).iloc[0]
        match = selection[(selection.scenario == key[0]) & (selection.method == key[1]) & (selection.batch_size == key[2])]
        assert len(match) == 1 and match.iloc[0]["lambda"] == best["lambda"]
    summary = {"runs_verified": len(frame), "paired_datasets": len(reference),
               "checkpoints_replayed": len(replayed), "selections_verified": len(selection),
               "upstream_unchanged": True, "source_matches_manifest": True}
    (out / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
