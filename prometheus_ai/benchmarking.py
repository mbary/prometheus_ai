import os
import asyncio
import random
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union, List

from pydantic import BaseModel
from prometheus_ai import Agent

import logfire
from datasets import load_dataset, Dataset
from tqdm.asyncio import tqdm
from rich import print, print_json

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
class Scenario(BaseModel):
    id:int
    full_command: str
    wakeword_phrase: str
    action_type: str
    zone: str
    light: Optional[str]
    temperature: Optional[int]
    brightness: Optional[Union[float,int]]
    brightness_relative: Optional[bool]
    brightness_up_down: Optional[str]
    color: Optional[str]
    split: str
class Trajectory(BaseModel):
    """Trajectory of a scenario."""
    scenario: Scenario
    action: Any = None
    score: Union[int,None] = None
    error: Optional[str] = None

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
def score_action(action, scenario):
    """
    Scores the action based on the command.
    """
    score = 0
    if action and action.action_type == scenario.action_type:
        score = 1
    return score

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
                score=score
            )
            logfire.info(f"Scenario ID: {scenario.id}\nFull command: {scenario.full_command}\n Action: {action}\n Score: {score}")

        return trajectory

@logfire.instrument('benchmarking', extract_args=True, record_return=True)
async def benchmark(
    model:str,
    base_url: str,
    max_concurrent_requests: int = 5,
    num_scenarios: int = None, 
    exclude_actions: Optional[List[str]] = None,
    max_retries: int = 1,
    seed: Optional[int] = None,
) -> int:
    scenarios = load_scenarios(
        'mbary/hue_commands_synth_3k', 
        split='test', 
        limit=num_scenarios,
        exclude_actions=exclude_actions,
        seed=seed
    )
    
    print(f"Loaded {len(scenarios)} scenarios after filtering (seed: {seed})")
    
    agent = Agent(benchmarking=True,
                  max_retries=max_retries,
                  model=model,
                  base_url=base_url)
    
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    trajectories = await tqdm.gather(*[run_agent_and_score(scenario=scenario,
                                                      semaphore=semaphore, 
                                                      agent=agent) for scenario in scenarios], desc="Benchmarking yo")
    
    

    

    return trajectories

if __name__ == '__main__':
    import asyncio
    from rich import print
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_PATH = Path(__file__).parent/"benchmarking/logs/"
    LOG_PATH.mkdir(parents=True, exist_ok=True)

    logfire.configure(token=os.environ.get("LOGFIRE_WRITE_TOKEN_PROMETHEUS"), console=False)
    logfire.instrument_openai()
    # model = "qwen/qwen3-30b-a3b"
    # model = "qwen/qwen3-14b"
    # model ="qwen/qwen3-32b"
    model='Qwen3-0.6B'
    base_url = "http://localhost:8000/v1"
    # model="deepseek/deepseek-chat-v3-0324"
    # model ="qwen/qwen3-235b-a22b"
    # base_url = os.getenv("OPENROUTER_BASE_URL")
    max_concurrent_requests = 5
    benchmark_seed = 42  # Use same seed for all model comparisons
    
    with logfire.span(f'benchmarking: {model}'):
        logfire.info("Starting benchmarking...")

        print(f"Running benchmark with rate limiting (max {max_concurrent_requests} concurrent requests)...")

        unimplemented_actions = [
            "set_color",
            "dim"
        ]
        ## TODO mve the saving of logs to file outsite the benchmark
        ## include it here and appent at the beginning a metadata class
        ## where all the shit is included like num of samples/seed/model/excluded etc
        print(f"Excluding actions: {unimplemented_actions}")
        print(f"Using seed: {benchmark_seed} for reproducible results")
        results = asyncio.run(benchmark(
            num_scenarios=50,
            max_concurrent_requests=max_concurrent_requests,
            exclude_actions=unimplemented_actions,
            model=model,
            seed=benchmark_seed,
            base_url=base_url,
        ))

        with open(LOG_PATH / f"benchmark_results_{model}_{len(results)}_{timestamp}.jsonl", 'a') as f:
            metadata = {
                "num_samples": len(results),
                "seed": benchmark_seed,
                "model": model,
                "excluded_actions": unimplemented_actions
            }
            if "localhost" in base_url:
                metadata["locally_served"] = True

            f.write(json.dumps({"metadata": metadata}) + "\n")
            for trajectory in results:
                f.write(trajectory.model_dump_json() + "\n")

        final_score_list_no_errors = [t.score for t in results if t.score is not None]
        final_score_list_w_errors = [t.score if t.score else 0 for t in results]
        
        final_score_no_errors = sum(final_score_list_no_errors) / len(final_score_list_no_errors) if final_score_list_no_errors else 0
        final_score_with_errors = sum(final_score_list_w_errors) / len(final_score_list_w_errors) if final_score_list_w_errors else 0

        successful_trajectories = [t for t in results if not t.error]
        error_trajectories = [t for t in results if t.error]

        print(f"Total scenarios: {len(results)}")
        print(f"Benchmarking succesfully completed for {len(successful_trajectories)}/ {len(results)} scenarios.")
        print(f"Benchmarking failed for {len(error_trajectories)}/ {len(results)} scenarios.")
        print(f"Benchmark result (no errors): {final_score_no_errors}")
        print(f"Benchmark result (with errors): {final_score_with_errors}")

        logfire.info(f"Benchmarking succesfully completed for {len(successful_trajectories)}/ {len(results)} scenarios.")
        logfire.info(f"Benchmarking failed for {len(error_trajectories)}/ {len(results)} scenarios.")
        logfire.info(f"Benchmark completed (no errors) with result: {final_score_no_errors}")
        logfire.info(f"Benchmark completed (with errors) with result: {final_score_with_errors}")