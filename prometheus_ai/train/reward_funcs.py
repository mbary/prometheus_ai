import json
from typing import List, Optional, Union, Any
from collections.abc import Callable
from pathlib import Path

import numpy as np
# import weave

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.utils_types import Scenario


def _clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))

def _is_none(x) -> bool:
    return x is None

def _eq(a, b) -> bool:
    return a == b

def _all_none_brightness(b) -> bool:
    if b is None:
        return True
    return getattr(b, "brightness", None) is None \
        and getattr(b, "relative", None) is None \
        and getattr(b, "up_down", None) is None


class RewardFunctions:
    """Strict/equality-based rewards. No tolerance. Extra params are penalized."""
    def __init__(self, parse_completion_to_action: Callable[[str], Union[Any, None]]) -> None:
        self.parse_completion_to_action = parse_completion_to_action

    # @weave.op()
    def json_validity_reward(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards = []
        for c in completions:
            try:
                s = c.strip()
                if s.startswith("```json"):
                    s = s.replace("```json", "").replace("```", "").strip()
                data = json.loads(s)
                ok = isinstance(data, dict) and \
                     "think" in data and "action_type" in data and isinstance(data.get("command"), dict)
                rewards.append(0.35 if ok else -1.0)
            except Exception:
                rewards.append(-1.0)
        return rewards

    # @weave.op()
    def action_selection_reward(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards = []
        scenarios = [Scenario(**json.loads(s)) for s in kwargs.get("scenarios", [])]
        for i, c in enumerate(completions):
            if i >= len(scenarios):
                rewards.append(0.0); continue
            sc = scenarios[i]
            try:
                act = self.parse_completion_to_action(c)
                if act is None:
                    rewards.append(-1.0)
                elif act.action_type == sc.action_type:
                    rewards.append(1.0)
                else:
                    rewards.append(-1.0)
            except Exception:
                rewards.append(-1.0)
        return rewards

    # @weave.op()
    def zone_selection_reward(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards = []
        scenarios = [Scenario(**json.loads(s)) for s in kwargs.get("scenarios", [])]
        for i, c in enumerate(completions):
            if i >= len(scenarios):
                rewards.append(0.0); continue
            sc = scenarios[i]
            try:
                act = self.parse_completion_to_action(c)
                if act is None or act.command is None:
                    rewards.append(-0.8); continue

                if sc.zone is not None:
                    if act.command.zone is None:
                        rewards.append(-1.0)
                    elif _eq(act.command.zone, sc.zone):
                        rewards.append(0.9)
                    else:
                        rewards.append(-1.0)
                else:
                    rewards.append(0.2 if act.command.zone is None else -0.5)
            except Exception:
                rewards.append(-0.8)
        return rewards

    # @weave.op()
    def parameter_accuracy_reward(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards = []
        scenarios = [Scenario(**json.loads(s)) for s in kwargs.get("scenarios", [])]

        for i, c in enumerate(completions):
            if i >= len(scenarios):
                rewards.append(0.0); continue
            sc = scenarios[i]
            try:
                act = self.parse_completion_to_action(c)
                if act is None or act.command is None:
                    rewards.append(-1.0); continue

                score = 0.0

                if sc.action_type == "set_scene":
                    if sc.scene is None:
                        score += 0.2 if act.command.scene is None else -0.7
                    else:
                        if act.command.scene is None:
                            score += -1.0
                        elif _eq(act.command.scene, sc.scene):
                            score += 1.0
                        else:
                            score += -1.0
                    if not _is_none(act.command.temperature): score += -0.5

                elif sc.action_type == "set_temperature":
                    if sc.temperature is None:
                        score += 0.2 if act.command.temperature is None else -0.7
                    else:
                        if act.command.temperature is None:
                            score += -1.0
                        elif _eq(act.command.temperature, sc.temperature):
                            score += 1.0
                        else:
                            score += -1.0
                    if act.command.scene is not None: score += -0.5

                elif sc.action_type in {"turn_on", "turn_off"}:
                    if act.command.scene is not None:        score += -0.5
                    if act.command.temperature is not None:  score += -0.5

                if sc.light is not None and sc.action_type in {"turn_on","turn_off","set_brightness","set_temperature"}:
                    if act.command.light is None:
                        score += -1.0
                    elif _eq(act.command.light, sc.light):
                        score += 0.8
                    else:
                        score += -1.0
                else:
                    if act.command.light is not None:
                        score += -0.5
                    else:
                        score += 0.2

                rewards.append(_clamp(score))
            except Exception:
                rewards.append(-0.8)
        return rewards

    # @weave.op()
    def brightness_control_reward(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards = []
        scenarios = kwargs.get("scenarios", [])
        if scenarios and isinstance(scenarios[0], str):
            scenarios = [Scenario(**json.loads(s)) for s in scenarios]

        for i, c in enumerate(completions):
            if i >= len(scenarios):
                rewards.append(0.0); continue
            sc = scenarios[i]
            try:
                act = self.parse_completion_to_action(c)
                if act is None or act.command is None:
                    rewards.append(-1.0); continue

                b = getattr(act.command, "brightness", None)
                sb = getattr(sc, "brightness", None)

                score = 0.0

                if sc.action_type == "set_brightness":
                    if b is None:
                        score += -1.0
                    else:
                        if sb and sb.brightness is not None:
                            if b.brightness is None:
                                score += -1.0
                            elif _eq(b.brightness, sb.brightness):
                                score += 1.0
                            else:
                                score += -1.0
                        else:
                            score += -0.7 if b and b.brightness is not None else 0.2

                        if sb and sb.relative is not None:
                            score += 0.6 if _eq(b.relative, sb.relative) else -0.8
                        else:
                            score += -0.5 if (b and b.relative is not None) else 0.2

                        if sb and sb.up_down is not None:
                            score += 0.6 if _eq(b.up_down, sb.up_down) else -0.8
                        else:
                            score += -0.5 if (b and b.up_down is not None) else 0.2
                else:
                    score += 0.4 if _all_none_brightness(b) else -0.8

                rewards.append(_clamp(score))
            except Exception:
                rewards.append(-1.0)
        return rewards

    # @weave.op()
    def extraneous_param_penalty(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards = []
        scenarios = [Scenario(**json.loads(s)) for s in kwargs.get("scenarios", [])]
        for i, c in enumerate(completions):
            if i >= len(scenarios):
                rewards.append(0.0); continue
            sc = scenarios[i]
            try:
                act = self.parse_completion_to_action(c)
                if act is None or act.command is None:
                    rewards.append(-0.6); continue

                cmd = act.command
                score = 0.0

                if sc.action_type == "set_scene":
                    if cmd.scene is None: score += -0.8
                    if cmd.temperature is not None: score += -0.6
                    if not _all_none_brightness(cmd.brightness): score += -0.6

                elif sc.action_type == "set_temperature":
                    if cmd.temperature is None: score += -0.8
                    if cmd.scene is not None: score += -0.6
                    if not _all_none_brightness(cmd.brightness): score += -0.6

                elif sc.action_type in {"turn_on", "turn_off"}:
                    if cmd.scene is not None: score += -0.6
                    if cmd.temperature is not None: score += -0.6
                    if not _all_none_brightness(cmd.brightness): score += -0.6

                elif sc.action_type == "set_brightness":
                    if cmd.brightness is None: score += -0.8
                    if cmd.scene is not None: score += -0.6
                    if cmd.temperature is not None: score += -0.6

                rewards.append(_clamp(score))
            except Exception:
                rewards.append(-0.6)
        return rewards


def create_grpo_reward_functions(agent):
    rf = RewardFunctions(agent.parse_completion_to_action)
    return [
        rf.json_validity_reward,
        rf.action_selection_reward,
        rf.zone_selection_reward,
        rf.parameter_accuracy_reward,
        rf.brightness_control_reward,
        rf.extraneous_param_penalty,
    ]
