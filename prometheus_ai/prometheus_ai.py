from __future__ import annotations


import os
from pathlib import Path
from typing import Any, List, Optional, Literal, Union, Dict
from openai import AsyncOpenAI, OpenAI
from pydantic import Field, create_model, ConfigDict, BaseModel
from rich import print
import instructor

from Prometheus import Bridgette
from Prometheus.device import HueZone, HueResource, HueLight, HueRoom
"""
This file is the agent responsible for managing and controlling the Philips Hue ecosystem.
(The base Bridgette class will likely require refactoring as it's quite old now)

It will consist of the following:

State/Deps managers
- StateManager
    Will contain all hue resource information, store their current satate
    available scenes/actions per resource
- DependenciesManager


Tools:
- set_scene (zone)
- turn_off (lights/room/zone)
- turn_on (lights/room/zone)
- set_brightness (zone/light)
- set_temperature
- set_color (zone/light) - experimental
- dim (zone/light) - experimental - automatically decrease the light (set_brightness to 50% of current level)


Zones:
- office
- lounge
- Lounge floor lights
- bedroom
- all
- tv

Scenes:
Scenes will be per zone/room


The agent will operate mostly on zone/light level and ignore rooms
As it seems like this is The HueWay. Rooms are mere light containers and allow for little flexibility when it comes to orgainising lights into groups (zones) and creaging custom scenes etc

To Do's:
- [] Implement StateManager
- [] Implement DependenciesManager
- [] Implement tools:
    - [] set_scene
        - [] execute
        - [] summarise
    - [] turn_off
        - [] execute
        - [] summarise
    - [] turn_on
        - [] execute
        - [] summarise
    - [] set_brightness
        - [] execute
        - [] summarise
    - [] set_color (experimental)
        - [] execute
        - [] summarise
    - [] dim
        - [] execute
        - [] summarise
    - [] set_temperature
        - [] execute
        - [] summarise
- [] step loop (is that even necessary if there are no steps per se and we're running continuously?)
- [] firebase logging
- [] implement event stream (no idea whether in the agent or Bridgette)

Testing:
- []  tools:
    - [] set_scene
    - [] turn_off
    - [] turn_on
    - [] set_brightness
    - [] set_color (experimental)
    - [] dim
"""

###################################################################
################### TOOL AND OUTPUT DEFINITIONS ###################
###################################################################

#############################################################
######### STATE MANAGER AND DEPENDENCIES DEFINITION #########
#############################################################

bridge = Bridgette()
print(dir(bridge.zones["office"]))
print(bridge.zones["office"].children)


class CustomBaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed=True
        ignored_types = (Bridgette, HueResource)


class StateManager(CustomBaseModel):
    zones: Dict[str,HueZone] = bridge.zones
    rooms: Dict[str,HueRoom] = bridge.rooms
    lights: Dict[str,HueLight] = bridge.lights

class DependenciesManager(CustomBaseModel):
    client: Any = Field(description="The instructor client used for LLM interactions.",)
    max_retries: int = Field(default=3, description="The maximum number of retries for LLM calls.",)
    bridge: Bridgette = Field(default=bridge, description="The Bridgette instance used for interacting with the Hue ecosystem.",)


class Command(BaseModel):
    thinking: str = Field(description="Think about the command to be executed. What action does the user want to perform?")

    zone: Literal["office", "lounge", "lounge floor lights", "bedroom", "all", "tv"] = Field(
        description="The name of the zone where the command will be executed.")
    
    light: Optional[str] = Field(
        description="The name of the light where the command will be executed")
    
    scene: Optional[Literal['natural light', 'relax, ', 'bloodbath', 'rest', 'disturbia', 
                            'relax', 'energize ', 'concentrate', 'read', 'warm embrace', 
                            'galaxy', 'phthalocyanine green love', 'starlight', 'tri colour',
                              'shrexy', 'nightlight', 'energize', 'vapor wavey', 'dimmed', 'valley dawn', 'soho ']] = Field(description="The scene to be set in the specified zone. A scene can be set only on an entire zone, not on a specific light.")

class TurnOn(BaseModel):
    thinking: str = Field(description="Think about the command to turn on the lights in the specified zone.")
    action_type: Literal["turn_on"] = "turn_on"
    command: Command = Field(description="Details where to perform the action")

    def execute(self, state: StateManager, deps: DependenciesManager, command: Command) -> None:
        ##TODO implement the execution and return the updated light state
        ## or maybe just update in the func and return outcome for the summary? 
        ## if thats even needed? not sure if I have to even 
        ## track the history conversation since it doesn't really matter

        """
        First I have to check what exactly is being turned on
        - if it's a zone, I can just turn on all lights in that zone
        - if it's a light, I can just turn on that light in a zone
        - zone MUST be specified, otherwise I can't turn on anything
        """
        if not command.zone:
            return 

        pass

    def summarise(self, state: StateManager):
        pass

class TurnOff(BaseModel):
    thinking: str = Field(description="Think about the command to turn off the lights in the specified zone.")
    action_type: Literal["turn_off"] = "turn_off"
    command: Command = Field(description="Details where to perform the action")

class SetScene(BaseModel):
    thinking: str = Field(description="What scene does the user what to set?")
    action_type: Literal["set_scene"] = "set_scene"
    command: Command = Field(description="Details where to perform the action")

class SetBrightness(BaseModel):
    thinking: str = Field(description="What brightness does the user want to set?")
    action_type: Literal["set_brightness"] = "set_brightness"
    command: Command = Field(description="Details where to perform the action")

class Dim(BaseModel):
    thinking: str = Field(description="Where does the user want to dim the lights?")
    action_type: Literal["dim"] = "dim"
    command: Command = Field(description="Details where to perform the action")

