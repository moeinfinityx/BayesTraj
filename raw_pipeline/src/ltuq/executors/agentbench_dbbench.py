"""DBBench bindings for the paper's shared trajectory generator."""

from __future__ import annotations

from dataclasses import dataclass

from .agentbench import (
    AgentBenchControllerClient,
    AgentBenchFunctionCallingSession,
    AgentBenchSample,
    AgentBenchSingleTrajectoryExecutor,
    AgentBenchTaskExecution,
)


@dataclass(frozen=True)
class AgentBenchDBBenchSample(AgentBenchSample):
    pass


class AgentBenchDBBenchExecutor(AgentBenchSingleTrajectoryExecutor):
    """Use the generic AgentBench executor with a DBBench-specific name."""

    name = "agentbench_dbbench"


__all__ = [
    "AgentBenchControllerClient",
    "AgentBenchDBBenchExecutor",
    "AgentBenchDBBenchSample",
    "AgentBenchFunctionCallingSession",
    "AgentBenchTaskExecution",
]
