"""
Command-line interface for the agentic compression framework.

This CLI allows running compression experiments (RQ1-RQ4) from the command line.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

# Import experiment functions
from .optimization.agent_driven import run_rq2_experiment
from .optimization.dynamic_vs_static import run_rq1_experiment
from .optimization.resource_adaptation import run_rq4_experiment
from .optimization.weighting import run_rq3_experiment

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print CLI banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║   Agentic Carbon-Efficient LLM Compression Framework v2.0    ║
║   Carbon-aware model compression with multi-agent systems    ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser"""
    parser = argparse.ArgumentParser(
        description="Agentic Carbon-Efficient LLM Compression Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run RQ2 experiment
  agentic-compress run-rq2 --model google/gemma-12b --carbon-budget 5.0

  # Run all experiments
  agentic-compress run-all --output results.json

  # Run RQ1 with custom parameters
  agentic-compress run-rq1 --accuracy-threshold 0.95 --max-iterations 30

For more information, see: https://github.com/your-org/agentic-compression
        """,
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # RQ1: Dynamic vs Static
    rq1_parser = subparsers.add_parser(
        "run-rq1", help="Run RQ1: Dynamic vs Static Compression Comparison"
    )
    rq1_parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-12b",
        help="Model identifier (default: google/gemma-12b)",
    )
    rq1_parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=0.93,
        help="Minimum acceptable accuracy (default: 0.93)",
    )
    rq1_parser.add_argument(
        "--carbon-budget", type=float, default=5.0, help="Carbon budget in kg CO2 (default: 5.0)"
    )
    rq1_parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum iterations for dynamic approach (default: 20)",
    )

    # RQ2: Agent-Driven Optimization
    rq2_parser = subparsers.add_parser("run-rq2", help="Run RQ2: Agent-Driven Optimization")
    rq2_parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-12b",
        help="Model identifier (default: google/gemma-12b)",
    )
    rq2_parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=0.93,
        help="Minimum acceptable accuracy (default: 0.93)",
    )
    rq2_parser.add_argument(
        "--carbon-budget", type=float, default=5.0, help="Carbon budget in kg CO2 (default: 5.0)"
    )

    # RQ3: Weighting Scheme Analysis
    rq3_parser = subparsers.add_parser("run-rq3", help="Run RQ3: Weighting Scheme Analysis")
    rq3_parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-12b",
        help="Model identifier (default: google/gemma-12b)",
    )
    rq3_parser.add_argument(
        "--carbon-budget", type=float, default=5.0, help="Carbon budget in kg CO2 (default: 5.0)"
    )

    # RQ4: Resource-Constrained Adaptation
    rq4_parser = subparsers.add_parser("run-rq4", help="Run RQ4: Resource-Constrained Adaptation")
    rq4_parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-12b",
        help="Model identifier (default: google/gemma-12b)",
    )
    rq4_parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=0.85,
        help="Minimum acceptable accuracy (default: 0.85)",
    )

    # Run all experiments
    all_parser = subparsers.add_parser("run-all", help="Run all experiments (RQ1-RQ4)")
    all_parser.add_argument(
        "--model",
        type=str,
        default="google/gemma-12b",
        help="Model identifier (default: google/gemma-12b)",
    )
    all_parser.add_argument(
        "--carbon-budget",
        type=float,
        default=5.0,
        help="Carbon budget in kg CO2 per experiment (default: 5.0)",
    )

    # Common arguments
    for p in [rq1_parser, rq2_parser, rq3_parser, rq4_parser, all_parser]:
        p.add_argument("--output", type=str, help="Output file for results (JSON format)")
        p.add_argument(
            "--format",
            choices=["json", "markdown", "summary"],
            default="summary",
            help="Output format (default: summary)",
        )
        p.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser


async def run_rq1(args: argparse.Namespace) -> dict[str, Any]:
    """Run RQ1 experiment"""
    logger.info("Running RQ1: Dynamic vs Static Comparison")

    results = await run_rq1_experiment(
        model=args.model,
        accuracy_threshold=args.accuracy_threshold,
        carbon_budget=args.carbon_budget,
        max_iterations=args.max_iterations,
    )

    return results


async def run_rq2(args: argparse.Namespace) -> dict[str, Any]:
    """Run RQ2 experiment"""
    logger.info("Running RQ2: Agent-Driven Optimization")

    results = await run_rq2_experiment(
        model=args.model,
        accuracy_threshold=args.accuracy_threshold,
        carbon_budget=args.carbon_budget,
    )

    return results


async def run_rq3(args: argparse.Namespace) -> dict[str, Any]:
    """Run RQ3 experiment"""
    logger.info("Running RQ3: Weighting Scheme Analysis")

    results = await run_rq3_experiment(model=args.model, carbon_budget=args.carbon_budget)

    return results


async def run_rq4(args: argparse.Namespace) -> dict[str, Any]:
    """Run RQ4 experiment"""
    logger.info("Running RQ4: Resource-Constrained Adaptation")

    results = await run_rq4_experiment(model=args.model, accuracy_threshold=args.accuracy_threshold)

    return results


async def run_all(args: argparse.Namespace) -> dict[str, Any]:
    """Run all experiments"""
    logger.info("Running all experiments (RQ1-RQ4)")

    all_results = {
        "framework": "Agentic Carbon-Efficient LLM Compression",
        "version": "2.0.0",
        "model": args.model,
        "experiments": {},
    }

    # Run each experiment
    logger.info("\n" + "=" * 60)
    logger.info("Experiment 1/4: Dynamic vs Static Comparison")
    logger.info("=" * 60)
    all_results["experiments"]["rq1"] = await run_rq1_experiment(
        model=args.model, accuracy_threshold=0.93, carbon_budget=args.carbon_budget
    )

    logger.info("\n" + "=" * 60)
    logger.info("Experiment 2/4: Agent-Driven Optimization")
    logger.info("=" * 60)
    all_results["experiments"]["rq2"] = await run_rq2_experiment(
        model=args.model, accuracy_threshold=0.93, carbon_budget=args.carbon_budget
    )

    logger.info("\n" + "=" * 60)
    logger.info("Experiment 3/4: Weighting Scheme Analysis")
    logger.info("=" * 60)
    all_results["experiments"]["rq3"] = await run_rq3_experiment(
        model=args.model, carbon_budget=args.carbon_budget
    )

    logger.info("\n" + "=" * 60)
    logger.info("Experiment 4/4: Resource-Constrained Adaptation")
    logger.info("=" * 60)
    all_results["experiments"]["rq4"] = await run_rq4_experiment(
        model=args.model, accuracy_threshold=0.85
    )

    logger.info("\n" + "=" * 60)
    logger.info("All experiments completed!")
    logger.info("=" * 60)

    return all_results


def format_json(results: dict[str, Any]) -> str:
    """Format results as JSON"""
    return json.dumps(results, indent=2)


def format_markdown(results: dict[str, Any]) -> str:
    """Format results as Markdown"""
    md = f"# {results.get('experiment', 'Experiment Results')}\n\n"

    if "model" in results:
        md += f"**Model**: {results['model']}\n\n"

    if "parameters" in results:
        md += "## Parameters\n\n"
        for key, value in results["parameters"].items():
            md += f"- **{key}**: {value}\n"
        md += "\n"

    if "key_findings" in results:
        md += "## Key Findings\n\n"
        for finding in results["key_findings"]:
            md += f"- {finding}\n"
        md += "\n"

    if "conclusion" in results:
        md += "## Conclusion\n\n"
        md += f"{results['conclusion']}\n\n"

    return md


def format_summary(results: dict[str, Any]) -> str:
    """Format results as console summary"""
    summary = "\n" + "=" * 70 + "\n"
    summary += f"  {results.get('experiment', 'Experiment Results')}\n"
    summary += "=" * 70 + "\n\n"

    if "model" in results:
        summary += f"Model: {results['model']}\n\n"

    if "parameters" in results:
        summary += "Parameters:\n"
        for key, value in results["parameters"].items():
            summary += f"  • {key}: {value}\n"
        summary += "\n"

    if "key_findings" in results:
        summary += "Key Findings:\n"
        for i, finding in enumerate(results["key_findings"], 1):
            summary += f"  {i}. {finding}\n"
        summary += "\n"

    if "conclusion" in results:
        summary += "Conclusion:\n"
        summary += f"  {results['conclusion']}\n"

    summary += "\n" + "=" * 70 + "\n"

    return summary


def save_results(results: dict[str, Any], output_path: str):
    """Save results to file"""
    output_file = Path(output_path)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {output_file}")


async def main_async(args: argparse.Namespace):
    """Main async entry point"""
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Execute command
    if args.command == "run-rq1":
        results = await run_rq1(args)
    elif args.command == "run-rq2":
        results = await run_rq2(args)
    elif args.command == "run-rq3":
        results = await run_rq3(args)
    elif args.command == "run-rq4":
        results = await run_rq4(args)
    elif args.command == "run-all":
        results = await run_all(args)
    else:
        logger.error("No command specified. Use --help for usage information.")
        return 1

    # Format output
    if args.format == "json":
        output = format_json(results)
    elif args.format == "markdown":
        output = format_markdown(results)
    else:  # summary
        output = format_summary(results)

    # Display output
    print(output)

    # Save to file if requested
    if args.output:
        save_results(results, args.output)

    return 0


def main():
    """Main entry point for CLI"""
    print_banner()

    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Run async main
    try:
        exit_code = asyncio.run(main_async(args))
        return exit_code
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
