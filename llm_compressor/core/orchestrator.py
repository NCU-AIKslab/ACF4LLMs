"""LLM-driven orchestrator using LangGraph for multi-agent workflows."""

import asyncio
import logging
from typing import Dict, List, Any, Optional, TypedDict, Annotated
import time
from pathlib import Path
import yaml

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from .registry import Registry
from .pareto import ParetoFrontier
from .metrics import MetricsCollector
from ..agents.base import AgentResult
from ..agents.llm_base import LLMBaseAgent
from ..agents.llm_quantization import LLMQuantizationAgent
from ..agents.llm_pruning import LLMPruningAgent
from ..agents.llm_distillation import LLMDistillationAgent
from ..agents.llm_kv_optimization import LLMKVOptimizationAgent
from ..agents.llm_performance import LLMPerformanceAgent
from ..agents.llm_evaluation import LLMEvaluationAgent
from ..agents.llm_recipe_planner import LLMRecipePlannerAgent


class WorkflowState(TypedDict):
    """State for the LangGraph workflow."""
    messages: Annotated[List[BaseMessage], add_messages]
    recipe: Dict[str, Any]
    context: Dict[str, Any]
    agent_results: Dict[str, AgentResult]
    current_step: str
    optimization_progress: Dict[str, Any]
    error_count: int
    success: bool


class Orchestrator:
    """LangGraph-based orchestrator for LLM-driven agent workflows."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)

        # Setup logging
        self.logger = logging.getLogger("llm_orchestrator")
        logging.basicConfig(
            level=getattr(logging, self.config.get("log_level", "INFO")),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        self.registry = Registry(self.config.get("registry", {}))
        self.pareto = ParetoFrontier(self.config.get("pareto", {}))
        self.metrics = MetricsCollector(self.config.get("metrics", {}))

        # Initialize LLM agents
        self.agents = self._initialize_llm_agents()

        # Build LangGraph workflow
        self.workflow = self._build_workflow()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _initialize_llm_agents(self) -> Dict[str, LLMBaseAgent]:
        """Initialize LLM-driven agents."""
        agents = {}

        # Map agent names to classes
        agent_classes = {
            "quantization": LLMQuantizationAgent,
            "pruning_sparsity": LLMPruningAgent,
            "distillation": LLMDistillationAgent,
            "kv_longcontext": LLMKVOptimizationAgent,
            "perf_carbon": LLMPerformanceAgent,
            "eval_safety": LLMEvaluationAgent,
            "recipe_planner": LLMRecipePlannerAgent
        }

        # Initialize agents based on configuration
        for agent_name, agent_class in agent_classes.items():
            if agent_name in self.config.get("agents", {}):
                agent_config = self.config["agents"][agent_name]

                # Add LLM configuration to agent config
                agent_config["llm"] = self.config.get("llm", {
                    "provider": "openai",
                    "model": "gpt-4",
                    "temperature": 0.1
                })

                agents[agent_name] = agent_class(agent_name, agent_config)
                self.logger.info(f"Initialized LLM agent: {agent_name}")

        return agents

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(WorkflowState)

        # Add nodes
        workflow.add_node("planner", self._planning_node)
        workflow.add_node("quantization", self._quantization_node)
        workflow.add_node("pruning", self._pruning_node)
        workflow.add_node("distillation", self._distillation_node)
        workflow.add_node("kv_optimization", self._kv_optimization_node)
        workflow.add_node("performance", self._performance_node)
        workflow.add_node("evaluation", self._evaluation_node)
        workflow.add_node("coordinator", self._coordination_node)
        workflow.add_node("error_handler", self._error_handler_node)

        # Add edges - conditional routing based on recipe and results
        workflow.add_edge(START, "planner")
        workflow.add_conditional_edges(
            "planner",
            self._route_from_planner,
            {
                "quantization": "quantization",
                "pruning": "pruning",
                "evaluation": "evaluation",
                "error": "error_handler"
            }
        )

        workflow.add_conditional_edges(
            "quantization",
            self._route_from_agent,
            {
                "pruning": "pruning",
                "evaluation": "evaluation",
                "coordinator": "coordinator",
                "error": "error_handler"
            }
        )

        workflow.add_conditional_edges(
            "pruning",
            self._route_from_agent,
            {
                "quantization": "quantization",
                "evaluation": "evaluation",
                "coordinator": "coordinator",
                "error": "error_handler"
            }
        )

        workflow.add_conditional_edges(
            "evaluation",
            self._route_from_evaluation,
            {
                "coordinator": "coordinator",
                "error": "error_handler",
                "end": END
            }
        )

        workflow.add_conditional_edges(
            "coordinator",
            self._route_from_coordinator,
            {
                "quantization": "quantization",
                "pruning": "pruning",
                "evaluation": "evaluation",
                "end": END,
                "error": "error_handler"
            }
        )

        workflow.add_edge("error_handler", END)

        return workflow.compile()

    def _planning_node(self, state: WorkflowState) -> WorkflowState:
        """Plan the optimization strategy."""
        self.logger.info("Planning optimization strategy...")

        recipe = state["recipe"]
        context = state["context"]

        # Create planning message
        planning_message = HumanMessage(
            content=f"""
            Plan optimization strategy for recipe: {recipe}
            Context: {context}

            Determine:
            1. Which agents to use in what order
            2. Expected interactions between optimizations
            3. Risk assessment for each step
            4. Success criteria
            """
        )

        # Add planning insights to context
        context["optimization_plan"] = {
            "planned_agents": recipe.get("pipeline", []),
            "risk_level": "medium",
            "expected_duration": "30min",
            "success_criteria": {
                "accuracy_threshold": 0.95,
                "speedup_target": 2.0
            }
        }

        state["messages"].append(planning_message)
        state["context"] = context
        state["current_step"] = "planning_complete"

        return state

    def _quantization_node(self, state: WorkflowState) -> WorkflowState:
        """Execute quantization agent."""
        self.logger.info("Executing quantization optimization...")

        if "quantization" not in self.agents:
            return self._handle_missing_agent(state, "quantization")

        try:
            agent = self.agents["quantization"]
            recipe = state["recipe"]
            context = state["context"]

            # Execute quantization
            result = agent.execute(recipe, context)

            # Update state
            state["agent_results"]["quantization"] = result

            if result.success:
                # Update context with quantization artifacts
                if "artifacts" not in context:
                    context["artifacts"] = {}
                context["artifacts"]["quantization"] = result.artifacts

                # Add success message
                success_msg = AIMessage(
                    content=f"Quantization successful: {result.metrics.get('quantization_method', 'unknown')} "
                           f"with {result.metrics.get('bits', 'unknown')} bits"
                )
                state["messages"].append(success_msg)

                self.logger.info(f"Quantization completed successfully: {result.metrics}")
            else:
                state["error_count"] += 1
                error_msg = AIMessage(content=f"Quantization failed: {result.error}")
                state["messages"].append(error_msg)
                self.logger.error(f"Quantization failed: {result.error}")

            state["context"] = context
            state["current_step"] = "quantization_complete"

        except Exception as e:
            state["error_count"] += 1
            state["agent_results"]["quantization"] = AgentResult(
                success=False, metrics={}, artifacts={}, error=str(e)
            )
            self.logger.error(f"Quantization node failed: {e}")

        return state

    def _pruning_node(self, state: WorkflowState) -> WorkflowState:
        """Execute pruning agent."""
        self.logger.info("Executing pruning optimization...")

        if "pruning_sparsity" not in self.agents:
            return self._handle_missing_agent(state, "pruning_sparsity")

        try:
            agent = self.agents["pruning_sparsity"]
            recipe = state["recipe"]
            context = state["context"]

            # Execute pruning
            result = agent.execute(recipe, context)

            # Update state
            state["agent_results"]["pruning"] = result

            if result.success:
                # Update context with pruning artifacts
                if "artifacts" not in context:
                    context["artifacts"] = {}
                context["artifacts"]["pruning"] = result.artifacts

                # Add success message
                success_msg = AIMessage(
                    content=f"Pruning successful: {result.metrics.get('pruning_method', 'unknown')} "
                           f"with {result.metrics.get('sparsity_ratio', 'unknown')} sparsity"
                )
                state["messages"].append(success_msg)

                self.logger.info(f"Pruning completed successfully: {result.metrics}")
            else:
                state["error_count"] += 1
                error_msg = AIMessage(content=f"Pruning failed: {result.error}")
                state["messages"].append(error_msg)
                self.logger.error(f"Pruning failed: {result.error}")

            state["context"] = context
            state["current_step"] = "pruning_complete"

        except Exception as e:
            state["error_count"] += 1
            state["agent_results"]["pruning"] = AgentResult(
                success=False, metrics={}, artifacts={}, error=str(e)
            )
            self.logger.error(f"Pruning node failed: {e}")

        return state

    def _evaluation_node(self, state: WorkflowState) -> WorkflowState:
        """Evaluate optimization results."""
        self.logger.info("Evaluating optimization results...")

        try:
            results = state["agent_results"]
            context = state["context"]

            # Collect metrics from all completed agents
            all_metrics = {}
            for agent_name, result in results.items():
                if result.success:
                    for key, value in result.metrics.items():
                        all_metrics[f"{agent_name}_{key}"] = value

            # Calculate composite score
            weights = self.config.get("objective_weights", {
                "accuracy": 1.0,
                "latency": -1.0,
                "vram": -1.0
            })

            composite_score = sum(
                weight * all_metrics.get(metric, 0)
                for metric, weight in weights.items()
            )

            all_metrics["composite_score"] = composite_score

            # Create evaluation result
            evaluation_result = AgentResult(
                success=True,
                metrics=all_metrics,
                artifacts={"final_metrics": all_metrics}
            )

            state["agent_results"]["evaluation"] = evaluation_result
            state["current_step"] = "evaluation_complete"

            # Add evaluation message
            eval_msg = AIMessage(
                content=f"Evaluation complete. Composite score: {composite_score:.3f}"
            )
            state["messages"].append(eval_msg)

            self.logger.info(f"Evaluation completed: {all_metrics}")

        except Exception as e:
            state["error_count"] += 1
            state["agent_results"]["evaluation"] = AgentResult(
                success=False, metrics={}, artifacts={}, error=str(e)
            )
            self.logger.error(f"Evaluation failed: {e}")

        return state

    def _distillation_node(self, state: WorkflowState) -> WorkflowState:
        """Execute distillation agent."""
        self.logger.info("Executing distillation optimization...")

        if "distillation" not in self.agents:
            return self._handle_missing_agent(state, "distillation")

        try:
            agent = self.agents["distillation"]
            recipe = state["recipe"]
            context = state["context"]

            # Execute distillation
            result = agent.execute(recipe, context)

            # Update state
            state["agent_results"]["distillation"] = result

            if result.success:
                # Update context with distillation artifacts
                if "artifacts" not in context:
                    context["artifacts"] = {}
                context["artifacts"]["distillation"] = result.artifacts

                # Add success message
                success_msg = AIMessage(
                    content=f"Distillation successful: {result.metrics.get('distillation_method', 'unknown')} "
                           f"to {result.metrics.get('student_size', 'unknown')} size"
                )
                state["messages"].append(success_msg)

                self.logger.info(f"Distillation completed successfully: {result.metrics}")
            else:
                state["error_count"] += 1
                error_msg = AIMessage(content=f"Distillation failed: {result.error}")
                state["messages"].append(error_msg)
                self.logger.error(f"Distillation failed: {result.error}")

            state["context"] = context
            state["current_step"] = "distillation_complete"

        except Exception as e:
            state["error_count"] += 1
            state["agent_results"]["distillation"] = AgentResult(
                success=False, metrics={}, artifacts={}, error=str(e)
            )
            self.logger.error(f"Distillation node failed: {e}")

        return state

    def _kv_optimization_node(self, state: WorkflowState) -> WorkflowState:
        """Execute KV optimization agent."""
        self.logger.info("Executing KV cache optimization...")

        if "kv_longcontext" not in self.agents:
            return self._handle_missing_agent(state, "kv_longcontext")

        try:
            agent = self.agents["kv_longcontext"]
            recipe = state["recipe"]
            context = state["context"]

            # Execute KV optimization
            result = agent.execute(recipe, context)

            # Update state
            state["agent_results"]["kv_optimization"] = result

            if result.success:
                # Update context with KV optimization artifacts
                if "artifacts" not in context:
                    context["artifacts"] = {}
                context["artifacts"]["kv_optimization"] = result.artifacts

                # Add success message
                success_msg = AIMessage(
                    content=f"KV optimization successful: {result.metrics.get('optimization_method', 'unknown')}"
                )
                state["messages"].append(success_msg)

                self.logger.info(f"KV optimization completed successfully: {result.metrics}")
            else:
                state["error_count"] += 1
                error_msg = AIMessage(content=f"KV optimization failed: {result.error}")
                state["messages"].append(error_msg)
                self.logger.error(f"KV optimization failed: {result.error}")

            state["context"] = context
            state["current_step"] = "kv_optimization_complete"

        except Exception as e:
            state["error_count"] += 1
            state["agent_results"]["kv_optimization"] = AgentResult(
                success=False, metrics={}, artifacts={}, error=str(e)
            )
            self.logger.error(f"KV optimization node failed: {e}")

        return state

    def _performance_node(self, state: WorkflowState) -> WorkflowState:
        """Execute performance monitoring agent."""
        self.logger.info("Executing performance monitoring...")

        if "perf_carbon" not in self.agents:
            return self._handle_missing_agent(state, "perf_carbon")

        try:
            agent = self.agents["perf_carbon"]
            recipe = state["recipe"]
            context = state["context"]

            # Execute performance monitoring
            result = agent.execute(recipe, context)

            # Update state
            state["agent_results"]["performance"] = result

            if result.success:
                # Update context with performance artifacts
                if "artifacts" not in context:
                    context["artifacts"] = {}
                context["artifacts"]["performance"] = result.artifacts

                # Add success message
                success_msg = AIMessage(
                    content=f"Performance monitoring successful: {result.metrics.get('analysis_type', 'unknown')}"
                )
                state["messages"].append(success_msg)

                self.logger.info(f"Performance monitoring completed successfully: {result.metrics}")
            else:
                state["error_count"] += 1
                error_msg = AIMessage(content=f"Performance monitoring failed: {result.error}")
                state["messages"].append(error_msg)
                self.logger.error(f"Performance monitoring failed: {result.error}")

            state["context"] = context
            state["current_step"] = "performance_complete"

        except Exception as e:
            state["error_count"] += 1
            state["agent_results"]["performance"] = AgentResult(
                success=False, metrics={}, artifacts={}, error=str(e)
            )
            self.logger.error(f"Performance monitoring node failed: {e}")

        return state

    def _coordination_node(self, state: WorkflowState) -> WorkflowState:
        """Coordinate next steps in optimization."""
        self.logger.info("Coordinating optimization workflow...")

        results = state["agent_results"]
        pipeline = state["recipe"].get("pipeline", [])
        current_step = state["current_step"]

        # Determine what's been completed and what's next
        completed_agents = set(results.keys()) - {"evaluation"}
        remaining_agents = set(pipeline) - completed_agents

        state["optimization_progress"] = {
            "completed": list(completed_agents),
            "remaining": list(remaining_agents),
            "success_rate": len([r for r in results.values() if r.success]) / max(len(results), 1)
        }

        # Add coordination message
        coord_msg = AIMessage(
            content=f"Workflow coordination: {len(completed_agents)} agents completed, "
                   f"{len(remaining_agents)} remaining"
        )
        state["messages"].append(coord_msg)

        state["current_step"] = "coordination_complete"

        return state

    def _error_handler_node(self, state: WorkflowState) -> WorkflowState:
        """Handle errors in the workflow."""
        self.logger.warning("Handling workflow errors...")

        error_count = state["error_count"]

        # Create error summary
        failed_agents = [
            name for name, result in state["agent_results"].items()
            if not result.success
        ]

        error_msg = AIMessage(
            content=f"Workflow completed with {error_count} errors. "
                   f"Failed agents: {failed_agents}"
        )
        state["messages"].append(error_msg)

        state["success"] = error_count == 0
        state["current_step"] = "error_handling_complete"

        return state

    def _handle_missing_agent(self, state: WorkflowState, agent_name: str) -> WorkflowState:
        """Handle missing agent."""
        error_msg = f"Agent {agent_name} not available"
        state["agent_results"][agent_name] = AgentResult(
            success=False, metrics={}, artifacts={}, error=error_msg
        )
        state["error_count"] += 1

        missing_msg = AIMessage(content=error_msg)
        state["messages"].append(missing_msg)

        return state

    # Routing functions for conditional edges
    def _route_from_planner(self, state: WorkflowState) -> str:
        """Route from planner node."""
        pipeline = state["recipe"].get("pipeline", [])

        if "quantization" in pipeline:
            return "quantization"
        elif "pruning_sparsity" in pipeline or "pruning" in pipeline:
            return "pruning"
        else:
            return "evaluation"

    def _route_from_agent(self, state: WorkflowState) -> str:
        """Route from agent nodes."""
        pipeline = state["recipe"].get("pipeline", [])
        completed = set(state["agent_results"].keys())

        # Find next agent in pipeline
        for agent_name in pipeline:
            if agent_name not in completed:
                if agent_name in ["pruning_sparsity", "pruning"]:
                    return "pruning"
                elif agent_name == "quantization":
                    return "quantization"

        # All agents completed, go to evaluation
        return "evaluation"

    def _route_from_evaluation(self, state: WorkflowState) -> str:
        """Route from evaluation node."""
        error_count = state["error_count"]

        if error_count > 0:
            return "error"
        else:
            return "end"

    def _route_from_coordinator(self, state: WorkflowState) -> str:
        """Route from coordinator node."""
        progress = state.get("optimization_progress", {})
        remaining = progress.get("remaining", [])

        if not remaining:
            return "end"
        elif state["error_count"] > 3:  # Too many errors
            return "error"
        else:
            # Route to next agent
            next_agent = remaining[0]
            if "quantization" in next_agent:
                return "quantization"
            elif "pruning" in next_agent:
                return "pruning"
            else:
                return "evaluation"

    def execute_recipe(self, recipe: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a recipe using the LangGraph workflow."""
        recipe_id = recipe.get("id", f"recipe_{int(time.time())}")
        self.logger.info(f"Executing recipe with LangGraph: {recipe_id}")

        # Initial state
        initial_state = WorkflowState(
            messages=[],
            recipe=recipe,
            context={
                "config": self.config,
                "recipe_id": recipe_id,
                "artifacts": {},
                "model_path": self.config.get("model", {}).get("base_model", "google/gemma-3-4b-it"),
                "model_config": self.config.get("model", {}),
                "hardware_config": self.config.get("hardware", {})
            },
            agent_results={},
            current_step="start",
            optimization_progress={},
            error_count=0,
            success=False
        )

        # Execute workflow
        try:
            final_state = self.workflow.invoke(initial_state)

            # Collect final metrics
            final_metrics = self._collect_final_metrics(final_state["agent_results"])

            # Register experiment
            experiment_data = {
                "recipe": recipe,
                "results": final_state["agent_results"],
                "metrics": final_metrics,
                "messages": [msg.content for msg in final_state["messages"]],
                "timestamp": time.time()
            }
            self.registry.register_experiment(recipe_id, experiment_data)

            return {
                "recipe_id": recipe_id,
                "success": final_state["success"],
                "metrics": final_metrics,
                "results": final_state["agent_results"],
                "workflow_messages": final_state["messages"]
            }

        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            return {
                "recipe_id": recipe_id,
                "success": False,
                "error": str(e),
                "results": {}
            }

    def _collect_final_metrics(self, results: Dict[str, AgentResult]) -> Dict[str, Any]:
        """Collect and aggregate final metrics from all agents."""
        metrics = {}

        # Aggregate metrics from all agents
        for agent_name, result in results.items():
            if result.success:
                for key, value in result.metrics.items():
                    metrics[f"{agent_name}_{key}"] = value

        # Calculate composite score
        weights = self.config.get("objective_weights", {
            "accuracy": 1.0,
            "latency": -1.0,
            "vram": -1.0,
            "energy": -1.0
        })

        score = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                score += weight * metrics[metric]

        metrics["composite_score"] = score
        return metrics