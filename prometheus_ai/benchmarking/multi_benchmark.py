import argparse
from collections import Counter
import os
import sys
import asyncio
import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import logfire
from tqdm.asyncio import tqdm
from rich import print
from rich.table import Table
from rich.console import Console

sys.path.append(str(Path(__file__).parent.parent))
from utils.project_types import Trajectory

from benchmarking import benchmark

console = Console()

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'models' not in config:
        raise ValueError("Configuration must contain 'models' field")
    
    if not isinstance(config['models'], list) or len(config['models']) == 0:
        raise ValueError("'models' must be a non-empty list")

    global_defaults = {
        "seed": 42,
        "samples": 10,
        "skip_actions": ["set_color", "dim"]
    }
    
    config.setdefault('global', {})
    for key, default_value in global_defaults.items():
        config['global'].setdefault(key, default_value)

    model_defaults = {
        "max_concurrent_requests": 5,
        "max_retries": 1
    }

    required_model_fields = ['model', 'provider']
    
    for model_config in config['models']:
        for field in required_model_fields:
            if field not in model_config:
                raise ValueError(f"Each model configuration must contain '{field}' field")

        for key, default_value in model_defaults.items():
            model_config.setdefault(key, default_value)
    
    return config

@logfire.instrument('multi_model_benchmarking', extract_args=True, record_return=True)
async def multi_benchmark(
    config: Dict[str, Any]
) -> Dict[str, List[Trajectory]]:

    all_results = {}
    global_config = config['global']
    models_config = config['models']
    
    for model_config in models_config:
        model = model_config['model']
        provider = model_config['provider']
        
        console.print(f"\n[bold blue]Starting benchmark for {model} ({provider})[/bold blue]")
        
        with logfire.span(f'multi_model_benchmarking: {model}'):
            logfire.info(f"Starting benchmarking for model: {model}")
            logfire.info(f"Provider: {provider}")
            logfire.info(f"Mode: {model_config.get('mode')}")
            logfire.info(f"Samples: {global_config['samples']}")
            logfire.info(f"Concurrent requests: {model_config['max_concurrent_requests']}")
            logfire.info(f"Max retries: {model_config['max_retries']}")
            logfire.info(f"Seed: {global_config['seed']}")
            logfire.info(f"Skipping actions: {global_config['skip_actions']}")
            
            try:
                results = await benchmark(
                    model=model,
                    provider=provider,
                    max_concurrent_requests=model_config['max_concurrent_requests'],
                    num_scenarios=global_config['samples'],
                    exclude_actions=global_config['skip_actions'],
                    max_retries=model_config['max_retries'],
                    seed=global_config['seed'],
                    mode=model_config.get('mode'),
                )
                all_results[(model, provider)] = results

                successful_trajectories = [t for t in results if not t.error]
                error_trajectories = [t for t in results if t.error]
                
                logfire.info(f"Completed benchmarking for {model}")
                logfire.info(f"Successfully completed: {len(successful_trajectories)}/{len(results)} scenarios")
                logfire.info(f"Error rate: {len(error_trajectories)}/{len(results)} scenarios")
                
                console.print(f"[green]✓ Completed {model}: {len(successful_trajectories)}/{len(results)} successful[/green]")
                
            except Exception as e:
                logfire.error(f"Error benchmarking {model}: {str(e)}")
                console.print(f"[red]✗ Failed {model}: {str(e)}[/red]")
                all_results[(model, provider)] = []
    
    return all_results

def display_summary_table(all_summaries: List[Dict[str, Any]]):
    """Display a summary table of all model results."""
    table = Table(title="Multi-Model Benchmark Summary")
    
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Provider", style="magenta")
    table.add_column("Total", justify="right")
    table.add_column("Success", justify="right", style="green")
    table.add_column("Errors", justify="right", style="red")
    table.add_column("Success Rate", justify="right")
    table.add_column("Score (No Errors)", justify="right", style="blue")
    table.add_column("Score (With Errors)", justify="right", style="blue")
    
    for summary in all_summaries:
        table.add_row(
            summary["model"],
            summary["provider"],
            str(summary["total_scenarios"]),
            str(summary["successful_scenarios"]),
            str(summary["failed_scenarios"]),
            f"{summary['success_rate']:.3f}",
            f"{summary['score_no_errors']:.3f}",
            f"{summary['score_with_errors']:.3f}"
        )
    console.print(table)

    if all_summaries and "detailed_scores" in all_summaries[0]:
        console.print("\n[bold]Detailed Scoring Breakdown by Metric:[/bold]")

        metrics = list(all_summaries[0]["detailed_scores"].keys())
        
        for metric in metrics:
            metric_table = Table(title=f"{metric.replace('_', ' ').title()} Accuracy")
            metric_table.add_column("Model", style="cyan", no_wrap=True)
            metric_table.add_column("Provider", style="magenta")
            metric_table.add_column("Accuracy", justify="right", style="green")
            
            for summary in all_summaries:
                if metric in summary["detailed_scores"]:
                    data = summary["detailed_scores"][metric]
                    metric_table.add_row(
                        summary["model"],
                        summary["provider"],
                        f"{data:.2%}",
                        )  
                if "error_details" in summary:
                    is_errors = True
                    error_table = Table(title=f"{metric.replace('_', ' ').title()} Error Details")
                    error_table.add_column("Error Type", style="red")
                    error_table.add_column("Count", style="red", justify="right")
                    for error_type, count in summary['error_details'].items():
                        error_table.add_row(error_type, str(count))
                              
            console.print(metric_table)
            # if is_errors:
            #     console.print(error_table)

def main():

    parser = argparse.ArgumentParser(description="Run benchmarking for Prometheus AI agent")
    parser.add_argument('--config-file', type=str, required=True, help="Path to the yaml configuration file")

    args = parser.parse_args()
    try:
        config = load_config(args.config_file)
    except FileNotFoundError:
        console.print(f"[red]Error: Configuration file '{args.config_file}' not found[/red]")
        return
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_PATH = Path(__file__).parent / "./multi_model_logs/"
    LOG_PATH.mkdir(parents=True, exist_ok=True)

    logfire.configure(token=os.environ.get("LOGFIRE_WRITE_TOKEN_PROMETHEUS"), console=False)
    logfire.instrument_openai()

    models_config = config['models']
    global_config = config['global']

    console.print(f"[bold]Starting multi-model benchmark with configuration:[/bold]")
    console.print(f"Config file: {args.config_file}")
    console.print(f"Models: {len(models_config)} models")
    console.print(f"Samples: {global_config['samples']}")
    console.print(f"Seed: {global_config['seed']}")
    console.print(f"Skipping actions: {global_config['skip_actions']}")

    for model_config in models_config:
        console.print(f"  - {model_config['model']} ({model_config['provider']}) "
                     f"- concurrent: {model_config['max_concurrent_requests']}, "
                     f"retries: {model_config['max_retries']}")
    
    with logfire.span(f'multi_model_benchmarking: {len(models_config)} models'):
        logfire.info("Starting multi-model benchmarking...")
        logfire.info(f"Start at: {datetime.now().isoformat()}")
        logfire.info(f"Config file: {args.config_file}")
        logfire.info(f"Models: {len(models_config)} models")
        logfire.info(f"Samples: {global_config['samples']}")
        logfire.info(f"Seed: {global_config['seed']}")
        logfire.info(f"Skipping actions: {global_config['skip_actions']}")

        all_results = asyncio.run(multi_benchmark(config))

        all_summaries = []
        
        for model_provider, results in all_results.items():
            if results: 
                model, provider = model_provider

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

                summary = {
                    "model": model,
                    "provider": provider,
                    "total_scenarios": len(results),
                    "successful_scenarios": len(successful_trajectories),
                    "failed_scenarios": len(error_trajectories),
                    "success_rate": len(successful_trajectories) / len(results) if results else 0,
                    "score_no_errors": final_score_no_errors,
                    "score_with_errors": final_score_with_errors,
                    "error_rate": len(error_trajectories) / len(results) if results else 0,
                    "detailed_scores": detailed_scores
                }
                if len(error_trajectories) > 0:
                    error_dict = Counter([t.error_type for t in error_trajectories])
                    summary["error_details"] = error_dict

                all_summaries.append(summary)

        total_scenarios = sum(len(results) for results in all_results.values() if results)
        multi_benchmark_filename = f"multi_benchmark_results_{len(models_config)}_models_{timestamp}.json"

        metadata = {
            "config_file": args.config_file,
            "samples": global_config['samples'],
            "seed": global_config['seed'],
            "models": [{"model": m['model'], "provider": m['provider'], 
                       "max_concurrent_requests": m['max_concurrent_requests'],
                       "max_retries": m['max_retries']} for m in models_config],
            "excluded_actions": global_config['skip_actions'],
            "timestamp": timestamp,
            "end_time": datetime.now().isoformat(),
            "total_models": len(models_config),
            "total_scenarios": total_scenarios
        }

        results_summary = {
            "total_models": len(models_config),
            "total_scenarios": total_scenarios,
            "model_summaries": all_summaries
        }
 
        detailed_results = {}
        for model_provider, results in all_results.items():
            if results:
                model, provider = model_provider
                model_key = f"{model}"
                
                detailed_results[model_key] = {
                    "scenarios": {}
                }
                
                for trajectory in results:
                    scenario_id = str(trajectory.scenario.id)
                    detailed_results[model_key]["scenarios"][scenario_id] = {
                        "scenario": trajectory.scenario.model_dump_json(),
                        "action": trajectory.action.model_dump_json() if trajectory.action else None,
                        "total_score": trajectory.total_score,
                        "success_rate": trajectory.success_rate,
                        "correct_tool": trajectory.correct_tool,
                        "correct_zone": trajectory.correct_zone,
                        "correct_scene": trajectory.correct_scene,
                        "correct_light": trajectory.correct_light,
                        "correct_temperature": trajectory.correct_temperature,
                        "correct_brightness": trajectory.correct_brightness,
                        "correct_brightness_relative": trajectory.correct_brightness_relative,
                        "correct_brightness_up_down": trajectory.correct_brightness_up_down,
                        "error": trajectory.error
                    }

        final_output = {
            "metadata": metadata,
            "results": results_summary,
            "detailed_results": detailed_results
        }

        with open(LOG_PATH / multi_benchmark_filename, 'w') as f:
            json.dump(final_output, f, indent=2)


        console.print(f"\n[bold green]Multi-Model Benchmark Complete![/bold green]")
        display_summary_table(all_summaries)

        logfire.info("Multi-model benchmark comparison results")
        logfire.info(f"Total models compared: {len(all_summaries)}")

        with logfire.span('Individual model summaries'):
            for summary in all_summaries:
                logfire.info(f"Model: {summary['model']} ({summary['provider']})")
                logfire.info(f"  - Total scenarios: {summary['total_scenarios']}")
                logfire.info(f"  - Success rate: {summary['success_rate']:.3f}")
                logfire.info(f"  - Score (no errors): {summary['score_no_errors']:.3f}")
                logfire.info(f"  - Score (with errors): {summary['score_with_errors']:.3f}")
                logfire.info(f"  - Error rate: {summary['error_rate']:.3f}")

        if len(all_summaries) > 1:
            best_success_model = max(all_summaries, key=lambda x: x['success_rate'])
            logfire.info(f"Best success rate: {best_success_model['model']} ({best_success_model['provider']}) - {best_success_model['success_rate']:.3f}")

            best_score_model = max(all_summaries, key=lambda x: x['score_no_errors'])
            logfire.info(f"Best score (no errors): {best_score_model['model']} ({best_score_model['provider']}) - {best_score_model['score_no_errors']:.3f}")

            avg_success_rate = sum(s['success_rate'] for s in all_summaries) / len(all_summaries)
            avg_score = sum(s['score_no_errors'] for s in all_summaries) / len(all_summaries)
            logfire.info(f"Average success rate across all models: {avg_success_rate:.3f}")
            logfire.info(f"Average score across all models: {avg_score:.3f}")

            min_success = min(s['success_rate'] for s in all_summaries)
            max_success = max(s['success_rate'] for s in all_summaries)
            logfire.info(f"Success rate range: {min_success:.3f} - {max_success:.3f}")

        logfire.info("Multi-model benchmark completed")
        logfire.info(f"Total models benchmarked: {len(all_summaries)}")

if __name__ == '__main__':
    main()