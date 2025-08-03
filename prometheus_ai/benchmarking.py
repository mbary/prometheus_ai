import os
from typing import Any, Optional, Union

from pydantic import BaseModel
from prometheus_ai import Agent

import logfire
# logfire.configure()
from datasets import load_dataset, Dataset
from tqdm.asyncio import tqdm
from rich import print, print_json
dataset = load_dataset("mbary/hue_commands_synth_3k", split="train")

len(dataset)
dataset[:10]
print([scenario['full_command'] for scenario in dataset[:10]])
dataset[0]

print([x for x in dataset[:10]])

# ag = Agent(benchmarking=True)
# a = ag.action(dataset[0]['full_command'])
# print(a)

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
    action: str
    zone: str
    light: Optional[str]
    temperature: Optional[int]
    brightness: Optional[Union[float,int]]
    brightness_relative: bool
    brightness_up_down: Optional[str]
    color: Optional[str]
    split: str



@logfire.instrument('score_action', extract_args=True, record_return=True)
async def score_action(action, command):
    """
    Scores the action based on the command.
    """
    # Placeholder for scoring logic
    # This should be replaced with actual scoring logic
    score = 0
    if action and action.command == command:
        score = 1  # Example score for a correct action
    return score

@logfire.instrument('run_agent_and_score', extract_args=True, record_return=True)
async def run_agent_and_score(
    scenario:Any,
    benchmarking: bool = True):
    print("runnin' and scorin'")
    agent= Agent(benchmarking=benchmarking)

    action = await agent.action(scenario['full_command'])
    score = await score_action(action, scenario['action'])

    return score

@logfire.instrument('benchmarking', extract_args=True, record_return=True)
async def benchmark(num_scenarios: int) ->int:
    scenarios = load_dataset('mbary/hue_commands_synth_3k', split='test')[:num_scenarios]

    results = await tqdm.gather(*[run_agent_and_score(scenario) for scenario in scenarios])

    return sum(results)/len(results) if results else 0

if __name__ == '__main__':
    import asyncio
    from rich import print
    print('running?')
    logfire.configure(token=os.environ.get("LOGFIRE_TOKEN"), console=False)
    print(asyncio.run(benchmark(num_scenarios=10)))
