"""
Local Compression Deep Agent using Ollama

No API key required - runs 100% locally!
"""

import os
from typing import Optional

from langchain_community.llms import Ollama

from .compression_deep_agent import CompressionDeepAgent


class LocalCompressionAgent(CompressionDeepAgent):
    """
    Compression Deep Agent using local LLM via Ollama.

    No API key required!

    Setup:
    1. Install Ollama: https://ollama.ai/
    2. Pull a model: ollama pull llama2
    3. Use this agent instead of the default one
    """

    def __init__(
        self,
        model_name: str = "llama2",  # or "mistral", "codellama", etc.
        workspace_dir: str = "./workspace",
        mlflow_tracking_uri: str = "./mlruns",
        temperature: float = 0.7,
        ollama_base_url: str = "http://localhost:11434",
    ):
        """
        Initialize Local Compression Deep Agent.

        Args:
            model_name: Ollama model name (llama2, mistral, codellama, etc.)
            workspace_dir: Directory for workspace memory
            mlflow_tracking_uri: MLflow tracking URI
            temperature: LLM temperature
            ollama_base_url: Ollama server URL
        """
        # Don't call super().__init__() - we'll set up manually
        from .compression_deep_agent import WorkspaceManager

        self.workspace = WorkspaceManager(workspace_dir)

        # Initialize LOCAL LLM via Ollama
        print(f"🏠 Initializing LOCAL agent with model: {model_name}")
        self.llm = Ollama(
            model=model_name,
            temperature=temperature,
            base_url=ollama_base_url,
        )

        # Create tools
        self.tools = self._create_tools(mlflow_tracking_uri)
        self.tool_dict = {tool.name: tool for tool in self.tools}

        print(f"✓ Local agent ready with {len(self.tools)} tools")


def create_local_compression_agent(
    model_name: str = "llama2",
    workspace_dir: str = "./workspace",
    **kwargs,
) -> LocalCompressionAgent:
    """
    Factory function to create a Local Compression Deep Agent.

    No API key required!

    Args:
        model_name: Ollama model name
        workspace_dir: Workspace directory for agent memory
        **kwargs: Additional arguments

    Returns:
        Initialized LocalCompressionAgent

    Example:
        >>> agent = create_local_compression_agent(model_name="llama2")
        >>> plan = agent.plan_compression(
        ...     model_name="meta-llama/Llama-2-7b-hf",
        ...     objective="edge deployment",
        ... )
    """
    return LocalCompressionAgent(
        model_name=model_name,
        workspace_dir=workspace_dir,
        **kwargs,
    )
