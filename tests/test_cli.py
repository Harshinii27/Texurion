"""CLI: exit codes, output and JSON snapshot."""

import json
import os

import pytest

import cli


def test_cli_runs_on_sample_image(capsys, tmp_path):
    code = cli.main(["images/brick.jpeg"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Image      : brick" in out
    assert "Features:" in out


def test_cli_missing_file_returns_error(capsys):
    code = cli.main(["images/not_there.png"])
    assert code == 2


def test_cli_writes_json_snapshot(capsys, tmp_path):
    out = tmp_path / "snapshot.json"
    code = cli.main(["images/brick.jpeg", "--json", str(out)])
    assert code == 0
    with open(out, encoding="utf-8") as handle:
        snapshot = json.load(handle)
    assert snapshot["parameters"]["method"] == "uniform"
    assert "features" in snapshot
    assert "classification" in snapshot


def test_cli_accepts_additional_parameters(capsys):
    code = cli.main(["images/brick.jpeg", "--radius", "2", "--method", "nri_uniform"])
    assert code == 0