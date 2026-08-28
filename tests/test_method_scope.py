import ast
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OURS = {"BR-LG-Risk-VC4-LCB95", "BR-LG-VarStop80-LCB95"}
BASELINES = {
    "PPL", "PE", "SentSAR", "LS", "SE", "SNNE", "KLE", "Degree",
    "EigV", "SD", "CoCoA-MaxProb", "CoCoA-PPL", "MC-OE",
    "BSE-Ciosek-Fixed", "BSE-Ciosek-Adaptive", "SAUP", "UProp",
}


def test_authoritative_method_registry_is_exactly_the_submission_scope():
    config = json.loads((ROOT / "config/paper.json").read_text())
    assert {config["fixed_method"], config["adaptive_method"]} == OURS
    assert set(config["baselines"]) == BASELINES
    assert len(config["baselines"]) == 17


def test_no_exploratory_method_scripts_are_shipped():
    forbidden = ("brqg", "sboe", "turf", "soft_semantic", "stopping_registry", "outcome_ptrue", "outcome_selfverify", "pmi_oe", "outcome_pmi")
    names = [path.name.lower() for path in (ROOT / "raw_pipeline/scripts").glob("*.py")]
    assert not [name for name in names if any(token in name for token in forbidden)]
    assert not [
        name for name in names
        if name.startswith(("aggregate_brlg_webshop", "verify_brlg_webshop"))
    ]


def test_unreported_standalone_modules_are_absent():
    absent = (
        "raw_pipeline/src/ltuq/uq/uam.py",
        "raw_pipeline/src/ltuq/estimators/local_branching.py",
        "raw_pipeline/src/ltuq/estimators/sequential_bayesian_oe.py",
        "raw_pipeline/src/ltuq/estimators/multistep_baseline.py",
        "raw_pipeline/src/ltuq/estimators/bayesian_risk_stopping.py",
        "raw_pipeline/src/ltuq/probes/hidden_states.py",
        "raw_pipeline/src/ltuq/probes/materialize.py",
        "raw_pipeline/src/ltuq/probes/model.py",
    )
    assert not [path for path in absent if (ROOT / path).exists()]


def test_retained_runtime_has_only_submitted_dataset_adapters():
    forbidden = ("knowledgegraph", "agentbench_os", "agentbenchos", "alfworld", "gaia", "telecom", "tau2")
    violations = []
    for path in (ROOT / "raw_pipeline/src").rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        if any(token in text for token in forbidden):
            violations.append(str(path.relative_to(ROOT)))
    assert not violations


def test_no_unreported_top_level_method_implementations():
    forbidden_symbols = (
        "uam",
        "sboe",
        "outcome_ptrue",
        "outcome_selfverify",
        "adaptive_local_branch",
        "local_branching_estimator",
        "sequential_bayesian",
        "main_trajectory_length",
        "saup_pd",
        "turf",
        "softvn",
        "brqg",
        "pmi_oe",
        "outcome-pmi",
        "outcome_pmi",
        "single-trajectory",
    )
    violations = []
    for path in (ROOT / "raw_pipeline").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                normalized = node.name.lower()
                if normalized == "compute_saup_d" or any(token in normalized for token in forbidden_symbols):
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.name}")
    assert not violations


def test_raw_generation_cli_is_paper_scoped():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "raw_pipeline/main.py"), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    help_text = completed.stdout.lower()
    assert "--method {uprop}" in help_text
    for token in ("uam", "sboe", "outcome-ptrue", "outcome-selfverify", "saup-pd", "sgd"):
        assert token not in help_text
    for command in (
        "run-strategyqa",
        "run-hotpotqa",
        "run-agentbench-dbbench",
        "run-agentbench-webshop",
    ):
        assert command in help_text


def test_all_paper_cli_commands_build_valid_configs():
    sys.path.insert(0, str(ROOT / "raw_pipeline/src"))
    from ltuq.cli import (
        _build_agentbench_dbbench_config,
        _build_agentbench_webshop_config,
        _build_hotpotqa_config,
        _build_strategyqa_config,
        build_parser,
    )

    parser = build_parser()
    commands = (
        ("run-strategyqa", _build_strategyqa_config),
        ("run-hotpotqa", _build_hotpotqa_config),
        ("run-agentbench-dbbench", _build_agentbench_dbbench_config),
        ("run-agentbench-webshop", _build_agentbench_webshop_config),
    )
    for command, builder in commands:
        config = builder(parser.parse_args([command]))
        assert config.method == "uprop"


def test_no_legacy_method_names_or_private_paths_are_shipped():
    forbidden_terms = (
        "sboe",
        "outcome_ptrue",
        "outcome_selfverify",
        "saup_pd",
        "main_trajectory_length",
        "turf",
        "softvn",
        "brqg",
        "pmi_oe",
        "outcome-pmi",
        "outcome_pmi",
        "final_step_pmi_u_mean",
        "final_step_pmi_u_max",
        "semantic-cluster",
        "semantic_ids",
        "hybrid-equivalence",
        "request-attribute",
        "kle_weighted",
        "pe_reasoning",
        "pe_decision",
        "hierarchical-normalized",
        "product-no-options",
    )
    forbidden_paths = ("/home/fxc190007", "/datapool/data/fxc190007")
    violations = []
    roots = (ROOT / "raw_pipeline", ROOT / "scripts", ROOT / "src", ROOT / "docs")
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".md", ".json"}:
                continue
            text = path.read_text(encoding="utf-8").lower()
            if any(term in text for term in forbidden_terms) or any(term in text for term in forbidden_paths):
                violations.append(str(path.relative_to(ROOT)))
    assert not violations
