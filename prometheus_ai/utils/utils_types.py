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

# import weave
import instructor
import logfire
from datasets import load_dataset
from tqdm.asyncio import tqdm
from rich import print
from rich.table import Table
from rich.console import Console


sys.path.append(str(Path(__file__).parent.parent))
from utils.project_types import Scenario

# @weave.op()
def load_scenarios(
    dataset_name: str = "mbary/hue_commands_synth_5k_v7", 
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
            
        if item['action_type'] in exclude_actions:
            continue
        
        scenario = Scenario(
            id=item.get('id', i),
            full_command=item['full_command'],
            wakeword_phrase=item['wakeword_phrase'],
            action_type=item['action_type'],
            zone=item['zone'],
            scene=item['scene'],
            light=item['light'],
            temperature=item['temperature'],
            brightness_value=item['brightness_value'] ,
            brightness_mode=item['brightness_mode'],
            brightness_direction=item['brightness_direction'],
            split=item['split']
        )
        scenarios.append(scenario)
        processed_count += 1
    
    return scenarios


# @weave.op()
def tool_usage(action, scenario) -> int:
    """
    Scores the action based on the command.
    """
    score = 0
    if action and action.action_type == scenario.action_type:
        score = 1
    return score

# @weave.op()
def correct_zone(action, scenario) -> int:
    """
    Checks if the action's zone matches the scenario's zone.
    """
    score = 0
    if action and action.command.zone == scenario.zone:
        score = 1
    return score

# @weave.op()
def correct_scene(action, scenario) -> int:
    score = 0
    if action and action.command.scene == scenario.scene:
        score = 1
    return score

# @weave.op()
def correct_light(action, scenario) -> int:
    score = 0
    if action and action.command.light == scenario.light:
        score = 1
    return score

# @weave.op()
def correct_temperature(action, scenario) -> int:
    score = 0
    if action and action.command.temperature == scenario.temperature:
        score = 1
    return score

# @weave.op()
def correct_brightness_value(action, scenario) -> int:

    score = 0
    if action and action.command.brightness_value == scenario.brightness_value:
        score = 1
    return score

# @weave.op()
def correct_brightness_mode(action, scenario) -> int:
    score = 0
    if action and action.command.brightness_mode == scenario.brightness_mode:
        score = 1
    return score

# @weave.op()
def correct_brightness_direction(action, scenario) -> int:
    score = 0
    if action and action.command.brightness_direction == scenario.brightness_direction:
        score = 1
    return score

# @weave.op()
def score_action(action,scenario) -> Dict[str, float]:
    correct_tool_score = tool_usage(action, scenario)
    correct_zone_score = correct_zone(action, scenario)
    correct_scene_score = correct_scene(action, scenario)
    correct_light_score = correct_light(action, scenario)
    correct_temperature_score = correct_temperature(action, scenario)
    correct_brightness_score = correct_brightness_value(action, scenario)
    correct_brightness_mode_score = correct_brightness_mode(action, scenario)
    correct_brightness_direction_score = correct_brightness_direction(action, scenario)

    score_dict = {
    "correct_tool": correct_tool_score,
    "correct_zone": correct_zone_score,
    "correct_scene": correct_scene_score,
    "correct_light": correct_light_score,
    "correct_temperature": correct_temperature_score,
    "correct_brightness_value": correct_brightness_score,
    "correct_brightness_mode": correct_brightness_mode_score,
    "correct_brightness_direction": correct_brightness_direction_score,
    }
    
    return score_dict