from __future__ import annotations

from typing import Any, List, Optional, Literal, Union, Dict
from pydantic import Field, BaseModel

from Prometheus import Bridgette
from Prometheus.device import HueResource
from dotenv import load_dotenv
load_dotenv()

class CustomBaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed=True
        ignored_types = (Bridgette, HueResource)

##TODO consider if this is even required
class StateManager(CustomBaseModel):
    bridge_state: Dict[str, Dict[str, Any]] = Field(description="A representation of the current bridge state")

class DependenciesManager(CustomBaseModel):
    client: Any = Field(description="The instructor client used for LLM interactions.",)
    max_retries: int = Field(default=3, description="The maximum number of retries for LLM calls.",)
    bridge: Optional[Bridgette] = Field(description="The Bridgette instance used for interacting with the Hue ecosystem.",)
    model: str = Field(description="The model used for LLM interactions.",default=None)
    benchmarking: bool = Field(default=False, description="Whether the agent is running in benchmarking mode. If True, the agent will return the action instead of executing it.")


class Brightness(BaseModel):
    brightness: Union[int,float] = Field(description="""The user's desired brightness level.
                                         Can be expressed in absolute values (int) or percentages (float).
                                         If a percentage is given, the absolute value will be based on the current light state.
                                         If user says 'set brightness to 50' return 50, if the user says 'decrease brightness to 30%', return 0.3. If the user says 'increase brightness by 10%' return 0.1. If the user says 'decrease brightness by 40%' return -0.4.""",
                                         examples=[30,0.5, 0.75, 100, 50, 0.25,
                                                   -40,-0.2,-100],
                                         ge=-100, le=100,
                                         )
    relative: bool = Field(description="Whether the brightness level is changed in absolute or relative terms. 'Change brightness by 20'-> relative terms; 'Set brightness to 30'->absolute terms", examples=[True, False])

    up_down: Literal['up','down'] = Field(description="Whether the brightness is increased or decreased. 'Increase brightness by 20'-> up; 'Decrease brightness by 30'-> down")

class Command(BaseModel):
    # thinking: str = Field(description="Think about the action to be executed. What action does the user want to perform?")

    zone: Literal["office", "lounge", "lounge floor lights", "bedroom", "all", "tv"] = Field(
        description="The name of the zone where the command will be executed.")
    
    light: Optional[str] = Field(
        description="The name of the light where the command will be executed")
    
    scene: Optional[Literal['natural light', 'relax', 'bloodbath', 'rest', 'disturbia', 
                            'relax', 'energize ', 'concentrate', 'read', 'warm embrace', 
                            'galaxy', 'phthalocyanine green love', 'starlight', 'tri colour',
                              'shrexy', 'nightlight', 'energize', 'vapor wavey', 'dimmed', 'valley dawn', 'soho']] = Field(description="The scene to be set in the specified zone. A scene can be set only on an entire zone, not on a specific light.")
    brightness: Optional[Brightness]
    temperature: Optional[int] = Field(description="The warmth of the light",
                                       ge=153, le=500,
                                       examples=[153, 200, 300, 400, 500]) 


class Scenario(BaseModel):
    id:int
    full_command: str
    wakeword_phrase: str
    action_type: str
    zone: str
    scene: Optional[str]
    light: Optional[str]
    temperature: Optional[int]
    brightness: Optional[Union[float,int]]
    brightness_relative: Optional[bool]
    brightness_up_down: Optional[str]
    color: Optional[str]
    split: Literal["train", "test"]

class Trajectory(BaseModel):
    """Trajectory of a scenario."""
    scenario: Scenario
    action: Any = None
    error: Optional[str] = None
    total_score: Union[int,None] = None
    success_rate: Optional[Union[int,None]] = None
    correct_tool: Optional[Union[int,None]] = None
    correct_zone: Optional[Union[int,None]] = None
    correct_scene: Optional[Union[int,None]] = None
    correct_light: Optional[Union[int,None]] = None
    correct_temperature: Optional[Union[int,None]] = None
    correct_brightness: Optional[Union[int,None]] = None
    correct_brightness_relative: Optional[Union[int,None]] = None
    correct_brightness_up_down: Optional[Union[int,None]] = None