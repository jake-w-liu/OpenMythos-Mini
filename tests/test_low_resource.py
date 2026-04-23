from argparse import Namespace
import json

import torch

from open_mythos import ByteTokenizer, OpenMythos, mythos_nano, mythos_small, mythos_tiny
from training.train_mythos_mini import (
    apply_preset,
    load_corpus,
    load_checkpoint,
    resolve_resume_path,
    save_checkpoint,
    write_metrics,
)
from training.compare_mythos_mini_runs import format_table, summarize_run


def test_byte_tokenizer_roundtrip():
    tokenizer = ByteTokenizer()
    text = "OpenMythos byte test."
    ids = tokenizer.encode(text)
    assert tokenizer.vocab_size == 256
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert tokenizer.decode(ids) == text


def test_low_resource_variants_disable_heavy_features():
    for builder in (mythos_nano, mythos_tiny, mythos_small):
        cfg = builder()
        assert cfg.attn_type == "gqa"
        assert cfg.recurrent_use_moe is False
        assert cfg.use_act is False
        assert cfg.vocab_size == 256


def test_nano_model_forward_shape():
    cfg = mythos_nano()
    model = OpenMythos(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    logits = model(ids)
    assert model.recurrent.act is None
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_mini_checkpoint_roundtrip(tmp_path):
    cfg = mythos_nano()
    model = OpenMythos(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    args = Namespace(variant="nano", steps=1)

    path = save_checkpoint(
        tmp_path,
        3,
        model,
        optimizer,
        scaler=None,
        cfg=cfg,
        args=args,
    )
    assert resolve_resume_path("latest", tmp_path) == path

    restored = OpenMythos(cfg)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    checkpoint = load_checkpoint(path, restored, optimizer=restored_optimizer)

    assert int(checkpoint["step"]) == 3
    for left, right in zip(model.parameters(), restored.parameters()):
        assert torch.allclose(left, right)


def test_named_preset_applies_defaults():
    args = Namespace(
        train_data=None,
        variant="nano",
        preset="baseline",
        device="auto",
        resume=None,
        steps=10,
        batch_size=4,
        grad_accum=1,
        seq_len=None,
        n_loops=None,
        lr=3e-4,
        min_lr=3e-5,
        warmup_steps=20,
        weight_decay=0.1,
        dropout=None,
        dense_ffn_mult=None,
        attn_type=None,
        use_moe=False,
        use_act=False,
        eval_every=10,
        eval_iters=1,
        sample_every=0,
        sample_prompt="OpenMythos",
        sample_tokens=16,
        temperature=0.8,
        top_k=32,
        checkpoint_every=0,
        out_dir="checkpoints-mini",
        metrics_file=None,
        seed=42,
    )
    merged = apply_preset(args)
    assert merged.variant == "tiny"
    assert merged.attn_type == "gqa"
    assert merged.use_moe is False
    assert merged.use_act is False


def test_write_metrics_jsonl(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    payload = {"step": 1, "train_loss": 1.23}
    write_metrics(str(metrics_path), payload)
    lines = metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == payload


def test_compare_run_summary(tmp_path):
    metrics_path = tmp_path / "baseline.jsonl"
    write_metrics(
        str(metrics_path),
        {
            "step": 1,
            "train_loss": 3.0,
            "val_loss": 4.0,
            "tokens_per_sec": 100.0,
            "variant": "tiny",
            "preset": "baseline",
            "attn_type": "gqa",
            "recurrent_use_moe": False,
            "use_act": False,
            "n_loops": 4,
            "corpus_files": 2,
        },
    )
    write_metrics(
        str(metrics_path),
        {
            "step": 2,
            "train_loss": 2.5,
            "val_loss": 3.5,
            "tokens_per_sec": 120.0,
            "variant": "tiny",
            "preset": "baseline",
            "attn_type": "gqa",
            "recurrent_use_moe": False,
            "use_act": False,
            "n_loops": 4,
            "corpus_files": 2,
        },
    )

    summary = summarize_run(metrics_path)
    assert summary["label"] == "baseline"
    assert summary["best_val_loss"] == 3.5
    assert summary["last_step"] == 2
    assert summary["last_tokens_per_sec"] == 120.0


def test_compare_table_format(tmp_path):
    rows = [
        {
            "label": "baseline",
            "preset": "baseline",
            "variant": "tiny",
            "attn_type": "gqa",
            "n_loops": 4,
            "recurrent_use_moe": False,
            "use_act": False,
            "best_val_loss": 3.5,
            "last_val_loss": 3.7,
            "best_train_loss": 2.5,
            "last_step": 10,
        }
    ]
    table = format_table(rows)
    assert "baseline" in table
    assert "best_val_loss" in table


def test_load_corpus_directory_can_include_code_and_skip_generated_dirs(tmp_path):
    (tmp_path / "notes.md").write_text("markdown corpus", encoding="utf-8")
    (tmp_path / "module.py").write_text("print('code corpus')", encoding="utf-8")
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "ignored.md").write_text("should be ignored", encoding="utf-8")

    default_text, default_meta = load_corpus(str(tmp_path))
    assert "markdown corpus" in default_text
    assert "code corpus" not in default_text
    assert default_meta["files_loaded"] == 1

    code_text, code_meta = load_corpus(str(tmp_path), include_code=True)
    assert "markdown corpus" in code_text
    assert "code corpus" in code_text
    assert "should be ignored" not in code_text
    assert code_meta["files_loaded"] == 2


def test_load_corpus_respects_max_chars(tmp_path):
    path = tmp_path / "notes.md"
    path.write_text("abcdefghij", encoding="utf-8")

    text, meta = load_corpus(str(path), max_chars=4)
    assert text == "abcd"
    assert meta["truncated"] is True


def test_generate_respects_context_limit():
    cfg = mythos_nano()
    cfg.max_seq_len = 8
    model = OpenMythos(cfg)
    ids = torch.randint(0, cfg.vocab_size, (1, 7))
    out = model.generate(ids, max_new_tokens=4, n_loops=2)
    assert out.shape[1] == 8
