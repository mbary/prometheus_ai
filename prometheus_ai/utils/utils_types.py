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
import logfire
from datasets import load_dataset
from tqdm.asyncio import tqdm
from rich import print
from rich.table import Table
from rich.console import Console


sys.path.append(str(Path(__file__).parent.parent))
from utils.project_types import Scenario

@logfire.instrument('load_scenarios', extract_args=True, record_return=True)
def load_scenarios(
    dataset_name: str = "mbary/hue_commands_synth_5k_v3", 
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
            brightness=item['brightness'] ,
            split=item['split']
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
    if action and action.command.zone == scenario.zone:
        score = 1
    return score

@logfire.instrument('correct_scene', extract_args=True, record_return=True)
def correct_scene(action, scenario) -> int:
    score = 0
    if action and action.command.scene == scenario.scene:
        score = 1
    return score

@logfire.instrument('correct_light', extract_args=True, record_return=True)
def correct_light(action, scenario) -> int:
    score = 0
    if action and action.command.light == scenario.light:
        score = 1
    return score

@logfire.instrument('correct_temperature', extract_args=True, record_return=True)
def correct_temperature(action, scenario) -> int:
    score = 0
    if action and action.command.temperature == scenario.temperature:
        score = 1
    return score


@logfire.instrument('correct_brightness', extract_args=True, record_return=True)
def correct_brightness(action, scenario) -> int:

    score = 0
    if action and action.command.brightness.brightness == scenario.brightness.brightness:
        score = 1
    return score

@logfire.instrument('correct_brightness_relative', extract_args=True, record_return=True)
def correct_brightness_relative(action, scenario) -> int:
    score = 0
    if action and action.command.brightness.relative == scenario.brightness.relative:
        score = 1
    return score

@logfire.instrument('correct_brightness_up_down', extract_args=True, record_return=True)
def correct_brightness_up_down(action, scenario) -> int:
    score = 0
    if action and action.command.brightness.up_down == scenario.brightness.up_down:
        score = 1
    return score

@logfire.instrument('score_action', extract_args=True, record_return=True)
def score_action(action,scenario) -> Dict[str, float]:
    correct_tool_score = tool_usage(action, scenario)
    correct_zone_score = correct_zone(action, scenario)
    correct_scene_score = correct_scene(action, scenario)
    correct_light_score = correct_light(action, scenario)
    correct_temperature_score = correct_temperature(action, scenario)
    correct_brightness_score = correct_brightness(action, scenario)
    correct_brightness_relative_score = correct_brightness_relative(action, scenario)
    correct_brightness_up_down_score = correct_brightness_up_down(action, scenario)

    score_dict = {
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