# Workspace Directory

This directory serves as the **long-term memory** for the Compression Deep Agent.

## Structure

- `experiments/`: Stores experiment configurations and results (JSON)
- `knowledge/`: Accumulated best practices and learned insights (Markdown)
- `checkpoints/`: Model checkpoint metadata and references

## Usage

The Deep Agent automatically:
- Saves all experiments to `experiments/`
- Documents successful strategies in `knowledge/`
- Tracks model checkpoints in `checkpoints/`

This workspace enables the agent to:
1. Learn from past experiments
2. Avoid repeating failed configurations
3. Build domain knowledge over time
4. Resume long-running optimizations

## Deep Agent Memory

Based on the LangChain Deep Agents architecture, the file system acts as persistent memory, allowing the agent to maintain context across multiple optimization runs.
