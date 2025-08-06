import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import asyncio
import random
import json
import argparse
from datetime import datetime
from typing import Optional, List, Dict

import instructor
from prometheus_ai import Agent
from utils.project_types import Scenario, Trajectory

import logfire
from datasets import load_dataset
from tqdm.asyncio import tqdm
from rich import print
from rich.table import Table
from rich.console import Console

##TODOs
""" 
- [] Implement scoring functions
   Example ideas:
       - [] Check if the action is valid vs command action
       - [] Check if the command details are correct:
            - [] MUST contain zone
            - [] MUST contain action
            - [] if brightness, check it's correct value vs command
            - [] if brightness relative, check that:
                - [] is correct %
                - [] is correct relative (up or down)
            - [] if temperature check it's correct value vs command
            - [] if light, check it's correct value vs command
                - [] ensure the light is correct i.e. it exists
            - [] if scene, check it's correct value vs command
                - [] ensure scene value exists
                - [] ensure that scene actually belongs to the zone
"""

@logfire.instrument('load_scenarios', extract_args=True, record_return=True)
def load_scenarios(
    dataset_name: str = "mbary/hue_commands_synth_3k", 
    split: str = "train", 
    limit: Optional[int] = None,
    exclude_actions: Optional[List[str]] = None,
    seed: Optional[int] = None
) -> List[Scenario]:
    """
    Load scenarios from a Hugging Face dataset and convert them to Scenario objects.
    
    Args:
        dataset_name: The name of the Hugging Face dataset to load
        split: The split to load (train, test, validation)
        limit: Maximum number of scenarios to return. If None, returns all scenarios.
        exclude_actions: List of action names to filter out from the dataset
        seed: Random seed for reproducible sampling. If None, no shuffling is applied.
    
    Returns:
        List of Scenario objects
    """
    dataset = load_dataset(dataset_name, split=split
                        #    split='train'
                        )
    all_items = list(dataset)
    
    if seed is not None:
        random.seed(seed)
        random.shuffle(all_items)
    
    scenarios = []
    if exclude_actions is None:
        exclude_actions = []
    
    processed_count = 0
    for i, item in enumerate(all_items):
        if limit is not None and processed_count >= limit:
            break
            
        if item['action'] in exclude_actions:
            continue
            
        scenario = Scenario(
            id=item.get('id', i),
            full_command=item['full_command'],
            wakeword_phrase=item['wakeword_phrase'],
            action_type=item['action'],
            zone=item['zone'],
            scene=item.get('scene'),
            light=item.get('light'),
            temperature=item.get('temperature'),
            brightness=item.get('brightness'),
            brightness_relative=item.get('brightness_relative', False),
            brightness_up_down=item.get('brightness_up_down'),
            color=item.get('color'),
            split=item.get('split', split)
        )
        scenarios.append(scenario)
        processed_count += 1
    
    logfire.info(f"Loaded {len(scenarios)} scenarios from {dataset_name} ({split} split) with limit={limit}, seed={seed}")
    return scenarios


@logfire.instrument('score_action', extract_args=True, record_return=True)
def tool_usage(action, scenario) -> int:
    """
    Scores the action based on the command.
    """
    score = 0
    if action and action.action_type == scenario.action_type:
        score = 1
    return score

@logfire.instrument('correct_zone', extract_args=True, record_return=True)
def correct_zone(action, scenario) -> int:
    """
    Checks if the action's zone matches the scenario's zone.
    """
    score = 0
    if action and action.zone == scenario.zone:
        score = 1
    return score

@logfire.instrument('correct_scene', extract_args=True, record_return=True)
def correct_scene(action, scenario) -> int:
    """
    Checks if the action's scene matches the scenario's scene.
    """
    score = 0
    if action and action.scene == scenario.scene:
        score = 1
    return score

@logfire.instrument('correct_light', extract_args=True, record_return=True)
def correct_light(action, scenario) -> int:
    """
    Checks if the action's light matches the scenario's light.
    """
    score = 0
    if action and action.light == scenario.light:
        score = 1
    return score

@logfire.instrument('correct_temperature', extract_args=True, record_return=True)
def correct_temperature(action, scenario) -> int:
    """
    Checks if the action's temperature matches the scenario's temperature.
    """
    score = 0
    if action and action.temperature == scenario.temperature:
        score = 1
    return score

@logfire.instrument('correct_brightness', extract_args=True, record_return=True)
def correct_brightness(action, scenario) -> int:
    """
    Checks if the action's brightness matches the scenario's brightness.
    """
    score = 0
    if action and action.brightness == scenario.brightness:
        score = 1
    return score

@logfire.instrument('correct_brightness_relative', extract_args=True, record_return=True)
def correct_brightness_relative(action, scenario) -> int:
    """
    Checks if the action's brightness relative matches the scenario's brightness relative.
    """
    score = 0
    if action and action.brightness_relative == scenario.brightness_relative:
        score = 1
    return score

@logfire.instrument('correct_brightness_up_down', extract_args=True, record_return=True)
def correct_brightness_up_down(action, scenario) -> int:
    """
    Checks if the action's brightness up/down matches the scenario's brightness up/down.
    """
    score = 0
    if action and action.brightness_up_down == scenario.brightness_up_down:
        score = 1
    return score

@logfire.instrument('score_action', extract_args=True, record_return=True)
def score_action(action,scenario) -> Dict[str, float]:
    """this will run all the scoring functions on the action and scenario"""
    correct_tool_score = tool_usage(action, scenario)
    correct_zone_score = correct_zone(action, scenario)
    correct_scene_score = correct_scene(action, scenario)
    correct_light_score = correct_light(action, scenario)
    correct_temperature_score = correct_temperature(action, scenario)
    correct_brightness_score = correct_brightness(action, scenario)
    correct_brightness_relative_score = correct_brightness_relative(action, scenario)
    correct_brightness_up_down_score = correct_brightness_up_down(action, scenario)

    total_score = (
        correct_tool_score +
        correct_zone_score +
        correct_scene_score +
        correct_light_score +
        correct_temperature_score +
        correct_brightness_score +
        correct_brightness_relative_score +
        correct_brightness_up_down_score
    )
    score_dict = {
        "total_score": total_score,
        "correct_tool": correct_tool_score,
        "correct_zone": correct_zone_score,
        "correct_scene": correct_scene_score,
        "correct_light": correct_light_score,
        "correct_temperature": correct_temperature_score,
        "correct_brightness": correct_brightness_score,
        "correct_brightness_relative": correct_brightness_relative_score,
        "correct_brightness_up_down": correct_brightness_up_down_score,
    }
    return score_dict


@logfire.instrument('run_agent_and_score', extract_args=True, record_return=True)
async def run_agent_and_score(
    scenario: Scenario,
    semaphore: asyncio.Semaphore,
    agent: Agent,
    )->List[Trajectory]:
    
    async with semaphore:
        action = await agent.action(scenario.full_command)
        if isinstance(action, dict) and 'error' in action:
            logfire.error(f"Error in action: {action['error']}\nScenario: {scenario.full_command}")
            trajectory = Trajectory(
                scenario=scenario,
                action=None,
                score=None,
                error=action['error']
            )
            return trajectory
        else:
            score = score_action(action, scenario)
            trajectory = Trajectory(
                scenario=scenario,
                action=action,
                total_score=score['total_score'],
                success_rate=score['total_score'] / sum([score[key] for key in score.keys() if key != 'total_score' and score[key]] ),
                correct_tool=score['correct_tool'],
                correct_zone=score['correct_zone'],
                correct_scene=score['correct_scene'],
                correct_light=score['correct_light'],
                correct_temperature=score['correct_temperature'],
                correct_brightness=score['correct_brightness'],
                correct_brightness_relative=score['correct_brightness_relative'],
                correct_brightness_up_down=score['correct_brightness_up_down']
            )
            logfire.info(f"Scenario ID: {scenario.id}\nFull command: {scenario.full_command}\n Action: {action}\n Scores: {score}")

        return trajectory

@logfire.instrument('benchmarking', extract_args=True, record_return=True)
async def benchmark(
    model:str,
    provider: str,
    max_concurrent_requests: int = 5,
    num_scenarios: int = None, 
    exclude_actions: Optional[List[str]] = None,
    max_retries: int = 1,
    seed: Optional[int] = None,
    mode: Optional[str] = None,
) -> List[Trajectory]:
    scenarios = load_scenarios(
        'mbary/hue_commands_synth_3k', 
        split='test', 
        limit=num_scenarios,
        exclude_actions=exclude_actions,
        seed=seed
    )
    
    print(f"Loaded {len(scenarios)} scenarios after filtering (seed: {seed})")
    
    # Convert string mode to instructor.Mode if provided
    instructor_mode = None
    if mode:
        print(f"MODE IN KURWA BENCHMARK: {mode}")
        if hasattr(instructor.Mode, mode.upper()):
            instructor_mode = getattr(instructor.Mode, mode.upper())
        else:
            raise ValueError(f"Invalid mode: {mode}. Available modes: {[attr for attr in dir(instructor.Mode) if not attr.startswith('_')]}")
    
    agent = Agent(benchmarking=True,
                  max_retries=max_retries,
                  model=model,
                  provider=provider,
                  mode=instructor_mode)
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    trajectories = await tqdm.gather(*[run_agent_and_score(scenario=scenario,
                                                      semaphore=semaphore, 
                                                      agent=agent) for scenario in scenarios], desc=f"Benchmarking {model} yo")
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

def main():
    """Main function that coordinates the benchmarking process with command-line arguments."""
    parser = argparse.ArgumentParser(description="Run benchmarking for Prometheus AI agent")
    
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducible results (default: 42)")
    parser.add_argument("--concurrent", type=int, default=5, 
                        help="Maximum number of concurrent requests (default: 5)")
    parser.add_argument("--samples", type=int, default=10, 
                        help="Number of scenarios to benchmark (default: 10)")
    parser.add_argument("--model-name", type=str, default="Qwen3-0.6B", 
                        help="Model name to use for benchmarking (default: Qwen3-0.6B)")
    parser.add_argument("--skip-actions", nargs="*", default=["set_color", "dim"], 
                        help="Actions to skip during benchmarking (default: set_color dim)")
    parser.add_argument("--provider", type=str, default="local",
                        help="Provider for the model API (default: local)")
    parser.add_argument("--mode", type=str, default=None,
                        help="Instructor mode to use (e.g., TOOLS, JSON, ANTHROPIC_TOOLS) (default: None - uses provider default)")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_PATH = Path("./logs/")
    LOG_PATH.mkdir(parents=True, exist_ok=True)
    
    logfire.configure(token=os.environ.get("LOGFIRE_WRITE_TOKEN_PROMETHEUS"), console=False)
    logfire.instrument_openai()

    print(f"   Starting benchmark with configuration:")
    print(f"   Model: {args.model_name}")
    print(f"   Provider: {args.provider}")
    print(f"   Samples: {args.samples}")
    print(f"   Concurrent requests: {args.concurrent}")
    print(f"   Seed: {args.seed}")
    print(f"   Skipping actions: {args.skip_actions}")
    
    with logfire.span(f'benchmarking: {args.model_name}'):
        logfire.info("Starting benchmarking...")
        logfire.info(f"Start at: {datetime.now().isoformat()}")
        logfire.info(f"Provider: {args.provider}")
        logfire.info(f"Model: {args.model_name}")
        logfire.info(f"Mode: {args.mode}")
        logfire.info(f"Samples: {args.samples}")
        logfire.info(f"Concurrent requests: {args.concurrent}")
        logfire.info(f"Seed: {args.seed}")
        logfire.info(f"Skipping actions: {args.skip_actions}")

        print(f"   Running benchmark with rate limiting (max {args.concurrent} concurrent requests)...\n")
        
        results = asyncio.run(benchmark(
            num_scenarios=args.samples,
            max_concurrent_requests=args.concurrent,
            exclude_actions=args.skip_actions,
            model=args.model_name,
            seed=args.seed,
            provider=args.provider,
            mode=args.mode,
        ))

        final_score_list_no_errors = [t.score for t in results if t.score is not None]
        final_score_list_w_errors = [t.score if t.score else 0 for t in results]
        
        final_score_no_errors = sum(final_score_list_no_errors) / len(final_score_list_no_errors) if final_score_list_no_errors else 0
        final_score_with_errors = sum(final_score_list_w_errors) / len(final_score_list_w_errors) if final_score_list_w_errors else 0
        
        successful_trajectories = [t for t in results if not t.error]
        error_trajectories = [t for t in results if t.error]

        with open(LOG_PATH / f"benchmark_results_{args.model_name.replace('/', '_').replace("-","_")}_{len(results)}_{timestamp}.jsonl", 'a') as f:
            metadata = {
                "samples": len(results),
                "seed": args.seed,
                "model": args.model_name,
                "excluded_actions": args.skip_actions,
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
                    "error_rate": len(error_trajectories) / len(results) if results else 0
                }
            }
            f.write(json.dumps(benchmark_results) + "\n")

            for trajectory in results:
                f.write(trajectory.model_dump_json() + "\n")
        
        print(f"\nBenchmark Results:")
        print(f"Total scenarios: {len(results)}")
        print(f"Successfully completed: {len(successful_trajectories)}/{len(results)} scenarios")
        print(f"Error rate: {len(error_trajectories)}/{len(results)} scenarios")
        print(f"Benchmark score (no errors): {final_score_no_errors:.3f}")
        print(f"Benchmark score (with errors): {final_score_with_errors:.3f}")
        
        logfire.info("Benchmark Results:")
        logfire.info(f"Successfully completed: {len(successful_trajectories)}/{len(results)} scenarios.")
        logfire.info(f"Error rate: {len(error_trajectories)}/{len(results)} scenarios.")
        logfire.info(f"Completed (no errors) with result: {final_score_no_errors}")
        logfire.info(f"Completed (with errors) with result: {final_score_with_errors}")

        display_summary_table(benchmark_results)

if __name__ == '__main__':
    main()