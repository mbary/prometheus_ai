import os
import sys
from pathlib import Path
from collections import Counter
import asyncio
import random
import json
import argparse
from datetime import datetime
from typing import Optional, List, Dict

import instructor
import weave
from datasets import load_dataset
from tqdm.asyncio import tqdm
from rich import print
from rich.table import Table
from rich.console import Console

sys.path.append(str(Path(__file__).parent.parent))
from prometheus_ai import Agent
from utils.project_types import Scenario, Trajectory
from utils.utils_types import load_scenarios, score_action  # <-- use shared utils

# -----------------------------
# run_agent_and_score
# -----------------------------
@weave.op()
async def run_agent_and_score(
    scenario: Scenario,
    semaphore: asyncio.Semaphore,
    agent: Agent,
) -> Trajectory:
    async with semaphore:
        action = await agent.action(scenario.full_command)

        if isinstance(action, dict) and 'error' in action:
            # simple print so we don't depend on loggers; trace captured by weave op return
            # print(f"[red]Error in action:[/red] {action['error']}\nScenario: {scenario.full_command}")
            trajectory = Trajectory(
                scenario=scenario,
                action=None,
                total_score=None,
                error=action['error'],
                error_type=action['error_type']
            )
            return trajectory
        else:
            # scoring comes from shared utils
            score = score_action(action, scenario)

            normalized_total_score = sum(score.values()) / len(score) if len(score) > 0 else 0
            success_rate = normalized_total_score

            trajectory = Trajectory(
                scenario=scenario,
                action=action,
                total_score=normalized_total_score,
                success_rate=success_rate,
                correct_tool=score['correct_tool'],
                correct_zone=score['correct_zone'],
                correct_scene=score['correct_scene'],
                correct_light=score['correct_light'],
                correct_temperature=score['correct_temperature'],
                correct_brightness=score['correct_brightness'],
                correct_brightness_relative=score['correct_brightness_relative'],
                correct_brightness_up_down=score['correct_brightness_up_down']
            )
            # light console note; detailed data is in the weave trace (inputs/outputs)
            # print(f"[dim]Scenario {scenario.id} scored {normalized_total_score:.3f}[/dim]")

        return trajectory

# -----------------------------
# benchmark orchestrator
# -----------------------------
@weave.op()
async def benchmark(
    model: str,
    provider: str,
    max_concurrent_requests: int = 8,
    num_scenarios: int = None,
    # exclude_actions: Optional[List[str]] = None,
    max_retries: int = 1,
    seed: Optional[int] = None,
    # mode: Optional[str] = None,
) -> List[Trajectory]:
    scenarios = load_scenarios(
        'mbary/hue_commands_synth_5k_v3',
        split='test',
        limit=num_scenarios,
        # exclude_actions=exclude_actions,
        seed=seed
    )

    print(f"Loaded {len(scenarios)} scenarios after filtering (seed: {seed})")

    # instructor_mode = None
    # if mode:
    #     if hasattr(instructor.Mode, mode.upper()):
    #         instructor_mode = getattr(instructor.Mode, mode.upper())
    #     else:
    #         raise ValueError(
    #             f"Invalid mode: {mode}. "
    #             f"Available modes: {[attr for attr in dir(instructor.Mode) if not attr.startswith('_')]}"
            # )

    agent = Agent(
        benchmarking=True,
        max_retries=max_retries,
        model=model,
        provider=provider,
        # mode=instructor_mode
    )
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    trajectories = await tqdm.gather(
        *[
            run_agent_and_score(
                scenario=scenario,
                semaphore=semaphore,
                agent=agent
            )
            for scenario in scenarios
        ],
        desc=f"Benchmarking {model} yo"
    )
    return trajectories

def display_summary_table(summary_dict: Dict):
    console = Console()
    results = summary_dict["benchmark_results"]

    table = Table(title="Benchmark Summary")

    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Total Scenarios", style="magenta", justify="right")
    table.add_column("Successful Scenarios", style="green", justify="right")
    table.add_column("Errors", style="red", justify="right")
    table.add_column("Success Rate", style="green", justify="right")
    table.add_column("Score (no errors)", style="green", justify="right")
    table.add_column("Score (with errors)", style="green", justify="right")

    table.add_row(
        summary_dict.get("model", "Unknown"),
        str(results["total_scenarios"]),
        str(results["successful_scenarios"]),
        str(results["failed_scenarios"]),
        f"{results['success_rate']:.2%}",
        f"{results['score_no_errors']:.3f}",
        f"{results['score_with_errors']:.3f}"
    )
    console.print(table)

    if "detailed_scores" in results:
        detailed_table = Table(title="Detailed Scoring Breakdown")
        detailed_table.add_column("Metric", style="cyan")
        detailed_table.add_column("Accuracy", style="green", justify="right")
        detailed_table.add_column("Total Scenarios", style="magenta", justify="right")

        for metric, data in results["detailed_scores"].items():
            detailed_table.add_row(
                metric.replace("_", " ").title(),
                f"{data:.2%}",
                str(results["successful_scenarios"])
            )
        console.print(detailed_table)

        if "error_details" in results:
            error_table = Table(title="Error Details")
            error_table.add_column("Error Type", style="red")
            error_table.add_column("Count", style="red", justify="right")
            for error_type, count in results["error_details"].items():
                error_table.add_row(error_type, str(count))
            console.print(error_table)

def main():
    """Main function that coordinates the benchmarking process with command-line arguments."""
    parser = argparse.ArgumentParser(description="Run benchmarking for Prometheus AI agent")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible results (default: 42)")
    parser.add_argument("--concurrent", type=int, default=8,
                        help="Maximum number of concurrent requests (default: 8)")
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of scenarios to benchmark (default: 10)")
    parser.add_argument("--model-name", type=str, default="Qwen3-0.6B",
                        help="Model name to use for benchmarking (default: Qwen3-0.6B)")
    # parser.add_argument("--skip-actions", nargs="*",
    #                     help="Actions to skip during benchmarking (default: set_color dim)")
    parser.add_argument("--provider", type=str, default="local",
                        help="Provider for the model API (default: local)")
    # parser.add_argument("--mode", type=str, default=None,
    #                     help="Instructor mode to use (e.g., TOOLS, JSON, ANTHROPIC_TOOLS)")
    parser.add_argument("--extras", type=str, default=None,
                        help="Additional key-value pairs for configuration.")
    # parser.add_argument("--temperature", type=float, default=None,
    #                     help="Temperature for the model")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_PATH = Path(__file__).parent.resolve() / "logs"
    LOG_PATH.mkdir(parents=True, exist_ok=True)

    # Initialize Weave (project can be overridden with env var)
    weave.init("benchmarking_new")

    print(f"   Starting benchmark with configuration:")
    print(f"   Model: {args.model_name}")
    print(f"   Provider: {args.provider}")
    print(f"   Samples: {args.samples}")
    print(f"   Concurrent requests: {args.concurrent}")
    print(f"   Seed: {args.seed}")
    # print(f"   Skipping actions: {args.skip_actions}")
    print(f"   Running benchmark with rate limiting (max {args.concurrent} concurrent requests)...\n")

    results = asyncio.run(benchmark(
        num_scenarios=args.samples,
        max_concurrent_requests=args.concurrent,
        # exclude_actions=args.skip_actions,
        model=args.model_name,
        seed=args.seed,
        provider=args.provider,
        # mode=args.mode,
    ))

    final_score_list_no_errors = [t.total_score for t in results if t.total_score is not None]
    final_score_list_w_errors = [t.total_score if t.total_score else 0 for t in results]

    final_score_no_errors = sum(final_score_list_no_errors) / len(final_score_list_no_errors) if final_score_list_no_errors else 0
    final_score_with_errors = sum(final_score_list_w_errors) / len(final_score_list_w_errors) if final_score_list_w_errors else 0

    successful_trajectories = [t for t in results if not t.error]
    error_trajectories = [t for t in results if t.error]

    correct_tool_final_score = sum(t.correct_tool for t in successful_trajectories) / len(successful_trajectories) if successful_trajectories else 0
    correct_zone_final_score = sum(t.correct_zone for t in successful_trajectories) / len(successful_trajectories) if successful_trajectories else 0
    correct_scene_final_score = sum(t.correct_scene for t in successful_trajectories) / len(successful_trajectories) if successful_trajectories else 0
    correct_light_final_score = sum(t.correct_light for t in successful_trajectories) / len(successful_trajectories) if successful_trajectories else 0
    correct_temperature_final_score = sum(t.correct_temperature for t in successful_trajectories) / len(successful_trajectories) if successful_trajectories else 0
    correct_brightness_final_score = sum(t.correct_brightness for t in successful_trajectories) / len(successful_trajectories) if successful_trajectories else 0
    correct_brightness_relative_final_score = sum(t.correct_brightness_relative for t in successful_trajectories) / len(successful_trajectories) if successful_trajectories else 0
    correct_brightness_up_down_final_score = sum(t.correct_brightness_up_down for t in successful_trajectories) / len(successful_trajectories) if successful_trajectories else 0

    detailed_scores = {
        "correct_tool": correct_tool_final_score,
        "correct_zone": correct_zone_final_score,
        "correct_scene": correct_scene_final_score,
        "correct_light": correct_light_final_score,
        "correct_temperature": correct_temperature_final_score,
        "correct_brightness": correct_brightness_final_score,
        "correct_brightness_relative": correct_brightness_relative_final_score,
        "correct_brightness_up_down": correct_brightness_up_down_final_score
    }
    with open(LOG_PATH / f"benchmark_results_{args.model_name.replace('/', '_').replace('-','_')}_{len(results)}_{timestamp}.json", 'a') as f:
        metadata = {
            "samples": len(results),
            "seed": args.seed,
            "model": args.model_name,
            # "excluded_actions": args.skip_actions,
            "concurrent_requests": args.concurrent,
            "provider": args.provider,
            "timestamp": timestamp,
            "end_time": datetime.now().isoformat()
        }
        if args.provider == "local":
            metadata["locally_served"] = True

        f.write(json.dumps({"metadata": metadata}) + "\n")

        benchmark_results = {
            "benchmark_results": {
                "total_scenarios": len(results),
                "successful_scenarios": len(successful_trajectories),
                "failed_scenarios": len(error_trajectories),
                "success_rate": len(successful_trajectories) / len(results) if results else 0,
                "score_no_errors": final_score_no_errors,
                "score_with_errors": final_score_with_errors,
                "error_rate": len(error_trajectories) / len(results) if results else 0,
                "detailed_scores": detailed_scores,
            },
            "model": args.model_name
        }
        if len(error_trajectories) > 0:
            error_dict = Counter([t.error for t in error_trajectories])
            benchmark_results["benchmark_results"]["error_details"] = error_dict

        f.write(json.dumps(benchmark_results) + "\n")

        for trajectory in results:
            f.write(trajectory.model_dump_json() + "\n")

    print(f"\nBenchmark Results:")
    print(f"Total scenarios: {len(results)}")
    print(f"Successfully completed: {len(successful_trajectories)}/{len(results)} scenarios")
    print(f"Error rate: {len(error_trajectories)}/{len(results)} scenarios")
    print(f"Benchmark score (no errors): {final_score_no_errors:.3f}")
    print(f"Benchmark score (with errors): {final_score_with_errors:.3f}")

    display_summary_table(benchmark_results)

if __name__ == '__main__':
    main()
