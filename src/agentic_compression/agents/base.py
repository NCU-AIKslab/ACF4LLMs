"""
Base agent class for compression strategies.

Currently a placeholder - agents are handled by LangGraph workflow and LangChain tools.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseCompressionAgent(ABC):
    """
    Base class for compression agents.

    Note: In the current implementation, agent functionality is provided by:
    - LangChain tools (src/agentic_compression/tools/)
    - LangGraph workflow (src/agentic_compression/graph/workflow.py)

    This class is a placeholder for future agent-based implementations.
    """

    def __init__(self, agent_id: str, agent_type: str):
        """
        Initialize base compression agent.

        Args:
            agent_id: Unique identifier for this agent
            agent_type: Type of compression (quantization, pruning, etc.)
        """
        self.agent_id = agent_id
        self.agent_type = agent_type

    @abstractmethod
    async def compress(self, model_path: str, config: dict[str, Any]) -> dict[str, Any]:
        """
        Execute compression strategy.

        Args:
            model_path: Path to model to compress
            config: Compression configuration

        Returns:
            Dictionary with compression results
        """
        pass

    def get_info(self) -> dict[str, str]:
        """Get agent information"""
        return {"agent_id": self.agent_id, "agent_type": self.agent_type}
