"""Dev selection and tamper-evident fresh-model provenance checks."""

import importlib
import json

import pytest

train = importlib.import_module("11_train_v2")
contract = importlib.import_module("_contract")


@pytest.mark.unit
def test_dev_selection_order():
    """Accuracy dominates loss; loss dominates checkpoint age."""
    key = train.selection_key
    assert key({"accuracy": 0.9, "loss": 0.8}, 3) < key({"accuracy": 0.8, "loss": 0.1}, 1)
    assert key({"accuracy": 0.9, "loss": 0.2}, 3) < key({"accuracy": 0.9, "loss": 0.3}, 1)
    assert key({"accuracy": 0.9, "loss": 0.2}, 1) < key({"accuracy": 0.9, "loss": 0.2}, 2)


@pytest.mark.unit
def test_v2_identity_rejects_changes_and_smoke(tmp_path):
    """Changing weights or provenance, or supplying smoke output, cannot pass."""
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    inventory = {"model.safetensors": contract.file_hash(tmp_path / "model.safetensors")}
    sha = contract.run_identity(inventory)
    meta = {
        "schema": "p2-fresh-training-v2",
        "smoke": False,
        "artifacts": inventory,
        "artifact_fingerprint": sha,
        "weights_sha256": inventory["model.safetensors"],
        "tokenizer_fingerprint": "fixture",
        "model_revision": "pinned",
        "tokenizer_revision": "pinned",
    }
    (tmp_path / "train_meta.json").write_text(json.dumps(meta))
    (tmp_path / "model_sha256.txt").write_text(sha)
    files = {p.name: contract.file_hash(p) for p in tmp_path.iterdir()}
    (tmp_path / "run_manifest.json").write_text(json.dumps({"files": files}))
    assert contract.checkpoint_identity(tmp_path)["training_sha256"] == sha
    (tmp_path / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        contract.checkpoint_identity(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"fixture")
    meta["model_revision"] = "unfrozen"
    (tmp_path / "train_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="provenance changed"):
        contract.checkpoint_identity(tmp_path)
    meta["smoke"] = True
    (tmp_path / "train_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="smoke checkpoint"):
        contract.checkpoint_identity(tmp_path)
