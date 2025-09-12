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
    training: bool = Field(default=False, description="Whether the agent is running in training mode. If True, the agent will return the action instead of executing it.")
    max_tokens: int = Field(description="The maximum number of tokens for LLM responses.")

class Brightness(BaseModel):
    brightness: Union[int,float, None] = Field(description="""The user's desired brightness level.
                                         Can be expressed in absolute values (int) or percentages (float).
                                         If a percentage is given, the absolute value will be based on the current light state.
                                         If user says 'set brightness to 50' return 50, if the user says 'decrease brightness to 30%', return 0.3. If the user says 'increase brightness by 10%' return 0.1. If the user says 'decrease brightness by 40%' return -0.4.""",
                                         ge=0, le=100,
                                         default=None)
    
    # relative: Union[bool, None] = Field(description="Whether the brightness level is changed in absolute or relative terms. 'Change brightness by 20'-> relative terms; 'Set brightness to 30'->absolute terms", 
    #                                     examples=[True, False],
    #                                     default=None)

    relative: bool = Field(description="Whether the brightness level is changed in absolute or relative terms. 'Change brightness by 20'-> relative terms; 'Set brightness to 30'->absolute terms", 
                                        examples=[True, False])

    up_down: Union[Literal['up','down'], None] = Field(description="Whether the brightness is increased or decreased. 'Increase brightness by 20'-> up; 'Decrease brightness by 30'-> down",
                                                       default=None,)

class Command(BaseModel):
    zone: Literal["office", "lounge", "lounge floor lights", "bedroom", "all", "tv"] = Field(
        description="The name of the zone where the command will be executed.")
    
    light: Union[str, None] = Field(
        description="The name of the light where the command will be executed",
        default=None,)

    scene: Union[str, None] = Field(description="The scene to be set in the specified zone. A scene can be set only on an entire zone, not on a specific light.",
                                    default=None,)
    temperature: Union[int, None] = Field(description="The user's desired light temperature. May be expressed in Kelvin units.",
                                       ge=153, le=500,
                                       examples=[153, 200, 300, 400, 500],
                                       default=None,) 
    brightness_value: Optional[Union[int, float, None]] = None
    brightness_mode: Optional[Literal["absolute", "relative", None]] = None
    brightness_direction: Optional[Literal["up", "down", None]] = None



class Scenario(BaseModel):
    """A scenario for the light controlling system."""
    id: int
    full_command: str
    wakeword_phrase: str
    action_type: str
    zone: str
    scene: Union[str, None]
    light: Union[str, None]
    temperature: Union[int, None]
    brightness_value: Optional[Union[int, float, None]]
    brightness_mode: Optional[Literal["absolute", "relative", None]]
    brightness_direction: Optional[Literal["up", "down", None]]
    split: Literal["train", "test"]

class Trajectory(BaseModel):
    """Trajectory of a scenario."""
    scenario: Scenario
    action: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    total_score: Union[int,float,None] = None
    success_rate: Optional[Union[float,None]] = None
    correct_tool: Optional[Union[int,None]] = None
    correct_zone: Optional[Union[int,None]] = None
    correct_scene: Optional[Union[int,None]] = None
    correct_light: Optional[Union[int,None]] = None
    correct_temperature: Optional[Union[int,None]] = None
    correct_brightness_value: Optional[Union[int,None]] = None
    correct_brightness_mode: Optional[Union[int,None]] = None
    correct_brightness_direction: Optional[Union[int,None]] = None
