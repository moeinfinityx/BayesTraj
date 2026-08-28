from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


TaskFamily = Literal[
    "strategyqa",
    "hotpotqa",
    "agentbench_webshop",
    "agentbench_webshop_emulated_tool_calls",
    "agentbench_dbbench",
    "agentbench_dbbench_emulated_tool_calls",
]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model: str
    task_family: TaskFamily
    output_dir: str = "outputs"
    notes: str = ""
