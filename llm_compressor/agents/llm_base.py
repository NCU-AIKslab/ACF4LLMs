"""LLM-driven base agent class using LangChain and LangGraph."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type
import logging
import time
import os
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from .base import BaseAgent, AgentResult


class AgentDecision(BaseModel):
    """Structured output for agent decisions."""
    action: str = Field(description="The action to take")
    parameters: Dict[str, Any] = Field(description="Parameters for the action")
    reasoning: str = Field(description="Reasoning behind the decision")
    confidence: float = Field(description="Confidence score (0-1)")
    estimated_impact: Dict[str, float] = Field(description="Estimated impact on metrics")


class LLMBaseAgent(BaseAgent):
    """Base class for LLM-driven compression/optimization agents."""

    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)

        # LLM configuration
        self.llm_config = config.get("llm", {})
        self.provider = self.llm_config.get("provider", "openai")
        self.model_name = self.llm_config.get("model", "gpt-4")
        self.temperature = self.llm_config.get("temperature", 0.1)

        # Initialize LLM
        self.llm = self._initialize_llm()

        # Output parser
        self.output_parser = PydanticOutputParser(pydantic_object=AgentDecision)

        # Agent-specific knowledge base
        self.knowledge_base = self._load_knowledge_base()

        # System prompt template
        self.system_prompt = self._create_system_prompt()

        # Create the chain
        self.chain = self._create_chain()

    def _initialize_llm(self):
        """Initialize the LLM based on provider configuration."""
        if self.provider == "openai":
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        elif self.provider == "anthropic":
            return ChatAnthropic(
                model=self.model_name,
                temperature=self.temperature,
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        elif self.provider == "google":
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _load_knowledge_base(self) -> str:
        """Load agent-specific knowledge base."""
        return f"""
        You are a specialized {self.name} agent for LLM compression and optimization.

        Your expertise includes:
        {self._get_agent_expertise()}

        Available tools and methods:
        {self._get_available_tools()}

        Performance considerations:
        {self._get_performance_considerations()}
        """

    @abstractmethod
    def _get_agent_expertise(self) -> str:
        """Return agent-specific expertise description."""
        pass

    @abstractmethod
    def _get_available_tools(self) -> str:
        """Return available tools and methods for this agent."""
        pass

    @abstractmethod
    def _get_performance_considerations(self) -> str:
        """Return performance considerations for this agent."""
        pass

    def _create_system_prompt(self) -> SystemMessagePromptTemplate:
        """Create the system prompt for this agent."""
        return SystemMessagePromptTemplate.from_template(
            f"""
            {self.knowledge_base}

            Your task is to analyze the given recipe and context, then make an intelligent decision
            about how to optimize the target model.

            You must respond in the following JSON format:
            {{
                "action": "specific action to take",
                "parameters": {{"param1": "value1", "param2": "value2"}},
                "reasoning": "detailed reasoning for your decision",
                "confidence": 0.85,
                "estimated_impact": {{"accuracy": -0.02, "latency": -0.3, "vram": -0.5}}
            }}

            Consider:
            1. The recipe requirements and constraints
            2. Hardware limitations
            3. Target performance metrics
            4. Risk vs reward trade-offs
            5. Compatibility with other optimizations

            Format instructions:
            {{format_instructions}}
            """
        )

    def _create_chain(self):
        """Create the LangChain processing chain."""
        prompt = ChatPromptTemplate.from_messages([
            self.system_prompt,
            HumanMessagePromptTemplate.from_template(
                """
                Recipe: {recipe}
                Context: {context}
                Current Model State: {model_state}
                Hardware Config: {hardware_config}
                Constraints: {constraints}

                Please analyze this situation and provide your optimization decision.
                """
            )
        ])

        return (
            RunnablePassthrough.assign(
                format_instructions=lambda _: self.output_parser.get_format_instructions()
            )
            | prompt
            | self.llm
            | self.output_parser
        )

    def execute(self, recipe: Dict[str, Any], context: Dict[str, Any]) -> AgentResult:
        """Execute the agent using LLM reasoning."""
        try:
            start_time = time.time()

            # Prepare input for LLM
            model_state = self._analyze_model_state(context)
            hardware_config = context.get("hardware_config", {})
            constraints = context.get("config", {}).get("constraints", {})

            # Get LLM decision
            decision = self.chain.invoke({
                "recipe": recipe,
                "context": context,
                "model_state": model_state,
                "hardware_config": hardware_config,
                "constraints": constraints
            })

            self.logger.info(f"LLM Decision: {decision.action} (confidence: {decision.confidence})")
            self.logger.info(f"Reasoning: {decision.reasoning}")

            # Execute the decision
            result = self._execute_decision(decision, recipe, context)
            result.execution_time = time.time() - start_time

            return result

        except Exception as e:
            self.logger.error(f"LLM agent execution failed: {e}")
            return AgentResult(
                success=False,
                metrics={},
                artifacts={},
                error=str(e),
                execution_time=time.time() - start_time
            )

    def _analyze_model_state(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current model state."""
        model_path = context.get("model_path", "")
        artifacts = context.get("artifacts", {})

        return {
            "model_path": model_path,
            "previous_optimizations": list(artifacts.keys()),
            "estimated_size": self._estimate_model_size(model_path),
            "current_precision": self._detect_current_precision(artifacts)
        }

    def _estimate_model_size(self, model_path: str) -> str:
        """Estimate model size category."""
        if "270m" in model_path.lower() or "270M" in model_path:
            return "small"
        elif "4b" in model_path.lower() or "4B" in model_path:
            return "medium"
        elif "7b" in model_path.lower() or "7B" in model_path:
            return "large"
        elif "13b" in model_path.lower() or "13B" in model_path:
            return "very_large"
        else:
            return "unknown"

    def _detect_current_precision(self, artifacts: Dict[str, Any]) -> str:
        """Detect current model precision."""
        for agent_name, artifact in artifacts.items():
            if "quantization" in agent_name:
                return artifact.get("precision", "fp16")
        return "fp16"

    @abstractmethod
    def _execute_decision(self, decision: AgentDecision, recipe: Dict[str, Any],
                         context: Dict[str, Any]) -> AgentResult:
        """Execute the LLM's decision."""
        pass

    def get_search_space(self) -> Dict[str, Any]:
        """Return LLM-informed search space."""
        # Use LLM to determine optimal search space
        try:
            prompt = f"""
            As a {self.name} expert, define the optimal hyperparameter search space
            for this optimization technique. Consider:
            1. Most impactful parameters
            2. Reasonable value ranges
            3. Interaction effects

            Return as JSON with parameter names as keys and ranges/options as values.
            """

            response = self.llm.invoke([HumanMessage(content=prompt)])
            # Parse and return the search space
            return self._parse_search_space(response.content)

        except Exception as e:
            self.logger.warning(f"Failed to get LLM search space: {e}")
            return super().get_search_space()

    def _parse_search_space(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM response for search space."""
        # Simple fallback implementation
        return {
            "learning_rate": [1e-5, 1e-4, 1e-3],
            "batch_size": [1, 2, 4, 8],
            "epochs": [1, 2, 3, 5]
        }

    def estimate_cost(self, recipe: Dict[str, Any]) -> Dict[str, float]:
        """LLM-informed cost estimation."""
        try:
            prompt = f"""
            Estimate computational costs for this {self.name} recipe:
            {recipe}

            Provide estimates for:
            - time (hours)
            - memory (GB)
            - energy (kWh)

            Return as JSON with numerical values.
            """

            response = self.llm.invoke([HumanMessage(content=prompt)])
            return self._parse_cost_estimate(response.content)

        except Exception as e:
            self.logger.warning(f"Failed to get LLM cost estimate: {e}")
            return super().estimate_cost(recipe)

    def _parse_cost_estimate(self, llm_response: str) -> Dict[str, float]:
        """Parse LLM cost estimation response."""
        # Simple fallback implementation
        return {"time": 2.0, "memory": 8.0, "energy": 0.5}