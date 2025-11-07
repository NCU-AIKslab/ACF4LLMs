"""
Compression Deep Agent

Main autonomous agent for model compression optimization using LangChain.
Based on LangChain Deep Agents architecture with 4 pillars:
1. Detailed System Prompt
2. Planning Tool (TodoList)
3. Sub-Agents (LoRA, Distillation, Evaluation)
4. File System (Workspace Memory)
"""

import json
import os
from typing import Any, Dict, List, Optional

from langchain_core.tools import Tool
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Import prompts
from .prompts import (
    COMPRESSION_DEEP_AGENT_PROMPT,
    DISTILLATION_SUB_AGENT_PROMPT,
    EVALUATION_SUB_AGENT_PROMPT,
    LORA_SUB_AGENT_PROMPT,
)

# Import tools
from .sub_agents.distillation_sub_agent import create_distillation_tools
from .sub_agents.lora_sub_agent import create_lora_tools
from .tracking_tool import create_tracking_tools
from ..tools.compression_tools import (
    DistillationTool,
    KVCacheTool,
    PruningTool,
    QuantizationTool,
)


class WorkspaceManager:
    """Manages workspace directory for Deep Agent memory."""

    def __init__(self, workspace_dir: str = "./workspace"):
        self.workspace_dir = workspace_dir
        self.experiments_dir = os.path.join(workspace_dir, "experiments")
        self.knowledge_dir = os.path.join(workspace_dir, "knowledge")
        self.checkpoints_dir = os.path.join(workspace_dir, "checkpoints")

        # Create directories
        for directory in [self.experiments_dir, self.knowledge_dir, self.checkpoints_dir]:
            os.makedirs(directory, exist_ok=True)

    def save_experiment(self, experiment_id: str, config: Dict, metrics: Dict) -> str:
        """Save experiment configuration and results."""
        filepath = os.path.join(self.experiments_dir, f"{experiment_id}.json")
        data = {
            "id": experiment_id,
            "config": config,
            "metrics": metrics,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        return filepath

    def load_experiments(self) -> List[Dict]:
        """Load all previous experiments."""
        experiments = []
        if not os.path.exists(self.experiments_dir):
            return experiments

        for filename in os.listdir(self.experiments_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.experiments_dir, filename)
                with open(filepath, "r") as f:
                    experiments.append(json.load(f))
        return experiments

    def save_knowledge(self, topic: str, content: str) -> str:
        """Save learned knowledge/best practices."""
        filepath = os.path.join(self.knowledge_dir, f"{topic}.md")
        with open(filepath, "w") as f:
            f.write(content)
        return filepath

    def load_knowledge(self, topic: str) -> Optional[str]:
        """Load knowledge on a specific topic."""
        filepath = os.path.join(self.knowledge_dir, f"{topic}.md")
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return f.read()
        return None


class CompressionDeepAgent:
    """
    Deep Agent for autonomous model compression optimization.

    Combines:
    - Detailed system prompts with domain expertise
    - Multiple compression tools (quantization, pruning, LoRA, distillation)
    - Sub-agents for specialized tasks
    - Workspace for long-term memory
    - MLflow for experiment tracking
    """

    def __init__(
        self,
        model_name: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        workspace_dir: str = "./workspace",
        mlflow_tracking_uri: str = "./mlruns",
        temperature: float = 0.7,
    ):
        """
        Initialize the Compression Deep Agent.

        Args:
            model_name: Anthropic model to use for the agent
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            workspace_dir: Directory for workspace memory
            mlflow_tracking_uri: MLflow tracking URI
            temperature: LLM temperature for agent reasoning
        """
        self.workspace = WorkspaceManager(workspace_dir)

        # Initialize LLM
        self.llm = ChatAnthropic(
            model=model_name,
            temperature=temperature,
            api_key=api_key,
            timeout=120,
        )

        # Create all tools
        self.tools = self._create_tools(mlflow_tracking_uri)
        self.tool_dict = {tool.name: tool for tool in self.tools}

    def _create_tools(self, mlflow_tracking_uri: str) -> List[Tool]:
        """Create all tools for the agent."""
        tools = []

        # Core compression tools
        tools.extend([
            QuantizationTool(),
            PruningTool(),
            KVCacheTool(),
            DistillationTool(),
        ])

        # LoRA tools
        tools.extend(create_lora_tools())

        # Distillation tools
        tools.extend(create_distillation_tools())

        # MLflow tracking tools
        tools.extend(create_tracking_tools(tracking_uri=mlflow_tracking_uri))

        # Workspace tools
        tools.extend([
            Tool(
                name="save_experiment_to_workspace",
                description="Save experiment configuration and results to workspace for future reference. "
                "Input: JSON with 'experiment_id', 'config', and 'metrics' fields.",
                func=lambda x: self._save_experiment_tool(json.loads(x)),
            ),
            Tool(
                name="load_previous_experiments",
                description="Load all previous experiments from workspace to learn from history. "
                "Input: empty string or 'all'",
                func=lambda x: json.dumps(self.workspace.load_experiments(), indent=2),
            ),
            Tool(
                name="save_knowledge",
                description="Save learned knowledge or best practices to workspace. "
                "Input: JSON with 'topic' and 'content' fields.",
                func=lambda x: self._save_knowledge_tool(json.loads(x)),
            ),
            Tool(
                name="load_knowledge",
                description="Load previously saved knowledge on a topic. "
                "Input: topic name (e.g., 'quantization_best_practices')",
                func=self.workspace.load_knowledge,
            ),
        ])

        return tools

    def _save_experiment_tool(self, data: Dict) -> str:
        """Tool wrapper for saving experiments."""
        try:
            filepath = self.workspace.save_experiment(
                experiment_id=data["experiment_id"],
                config=data["config"],
                metrics=data["metrics"],
            )
            return f"✓ Experiment saved to {filepath}"
        except Exception as e:
            return f"✗ Failed to save experiment: {str(e)}"

    def _save_knowledge_tool(self, data: Dict) -> str:
        """Tool wrapper for saving knowledge."""
        try:
            filepath = self.workspace.save_knowledge(
                topic=data["topic"],
                content=data["content"],
            )
            return f"✓ Knowledge saved to {filepath}"
        except Exception as e:
            return f"✗ Failed to save knowledge: {str(e)}"

    def run(self, task: str) -> Dict[str, Any]:
        """
        Run the agent on a compression optimization task.

        Simplified version: Uses LLM with tool descriptions for now.
        Full ReAct agent implementation can be added when needed.

        Args:
            task: Task description (e.g., "Find optimal 8-bit quantization config for Llama-2-7b")

        Returns:
            Agent result dictionary with 'output' key
        """
        # Build tool descriptions
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description[:100]}..."
            for tool in self.tools
        ])

        # Create prompt
        prompt = f"""
{COMPRESSION_DEEP_AGENT_PROMPT}

Available Tools:
{tool_descriptions}

Task: {task}

Please provide your analysis and recommendations for this compression task.
Consider which tools would be most useful and what configurations to try.
"""

        # Call LLM
        messages = [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)

        return {
            "output": response.content if hasattr(response, 'content') else str(response),
            "task": task,
        }

    def plan_compression(
        self,
        model_name: str,
        objective: str,
        carbon_budget: Optional[float] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Plan compression strategy for a model.

        Args:
            model_name: Model to compress (e.g., "meta-llama/Llama-2-7b-hf")
            objective: Optimization objective (e.g., "maximize speedup with <2% accuracy loss")
            carbon_budget: Optional carbon budget in kg CO2
            constraints: Optional additional constraints

        Returns:
            Compression plan with recommended configurations
        """
        task = f"""
Plan compression optimization for the following model:

Model: {model_name}
Objective: {objective}
Carbon Budget: {carbon_budget if carbon_budget else "No limit"} kg CO2
Constraints: {constraints if constraints else "None"}

Your task:
1. Analyze the objective and determine which compression techniques to try
2. Check workspace for any previous experiments on this model
3. Recommend specific configurations to test (at least 3-5 configurations)
4. Estimate expected results for each configuration
5. Prioritize configurations based on likelihood of meeting objectives

Provide a structured compression plan with:
- Recommended techniques and configurations
- Expected metrics for each
- Rationale for each recommendation
- Suggested evaluation order
"""
        return self.run(task)

    def execute_experiment(
        self,
        config: Dict[str, Any],
        log_to_mlflow: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a compression experiment based on configuration.

        Args:
            config: Compression configuration dict
            log_to_mlflow: Whether to log to MLflow automatically

        Returns:
            Experiment results
        """
        task = f"""
Execute the following compression experiment:

Configuration: {json.dumps(config, indent=2)}
Auto-log to MLflow: {log_to_mlflow}

Your task:
1. Apply the compression technique specified in the config
2. If using LoRA/PEFT or distillation, delegate to the appropriate sub-agent
3. Measure theoretical metrics (compression ratio, expected speedup, memory reduction)
4. {'Log results to MLflow using the log_experiment tool' if log_to_mlflow else 'Do not log to MLflow'}
5. Save experiment to workspace using save_experiment_to_workspace
6. Return comprehensive results

Provide structured results with all metrics and any issues encountered.
"""
        return self.run(task)

    def reflect_and_improve(
        self,
        current_solutions: List[Dict[str, Any]],
        objective: str,
    ) -> Dict[str, Any]:
        """
        Reflect on current solutions and suggest improvements.

        Args:
            current_solutions: List of evaluated compression configurations
            objective: Original optimization objective

        Returns:
            Improvement recommendations
        """
        task = f"""
Reflect on the current compression solutions and suggest improvements:

Objective: {objective}
Current Solutions: {json.dumps(current_solutions, indent=2)}

Your task:
1. Analyze the Pareto frontier of current solutions
2. Identify gaps or weaknesses (e.g., poor accuracy, insufficient speedup)
3. Query MLflow for similar past experiments using query_experiments
4. Load relevant knowledge from workspace
5. Suggest new configurations to try that might improve the frontier
6. Document your findings and save insights to workspace

Provide:
- Analysis of current solutions
- Identified gaps
- 3-5 new configurations to try
- Rationale for each recommendation
- Updated best practices (save to workspace)
"""
        return self.run(task)


def create_compression_deep_agent(
    model_name: str = "claude-3-5-sonnet-20241022",
    workspace_dir: str = "./workspace",
    **kwargs,
) -> CompressionDeepAgent:
    """
    Factory function to create a Compression Deep Agent.

    Args:
        model_name: Anthropic model name
        workspace_dir: Workspace directory for agent memory
        **kwargs: Additional arguments for CompressionDeepAgent

    Returns:
        Initialized CompressionDeepAgent
    """
    return CompressionDeepAgent(
        model_name=model_name,
        workspace_dir=workspace_dir,
        **kwargs,
    )
