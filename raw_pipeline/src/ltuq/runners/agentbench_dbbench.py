from __future__ import annotations

from dataclasses import dataclass

from .agentbench import (
    AgentBenchRunnerConfig,
    run_agentbench_dbbench,
    run_agentbench_dbbench_samples,
    summarize_agentbench_dbbench_results,
    write_agentbench_dbbench_results,
)


@dataclass(frozen=True)
class AgentBenchDBBenchRunnerConfig(AgentBenchRunnerConfig):
    task_name: str = "dbbench-std"


__all__ = [
    "AgentBenchDBBenchRunnerConfig",
    "run_agentbench_dbbench",
    "run_agentbench_dbbench_samples",
    "summarize_agentbench_dbbench_results",
    "write_agentbench_dbbench_results",
]
