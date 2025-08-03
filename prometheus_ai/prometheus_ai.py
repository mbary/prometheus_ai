from __future__ import annotations


import json
import os
import textwrap
from pathlib import Path
from typing import Any, List, Optional, Literal, Union, Dict
from openai import AsyncOpenAI, OpenAI
from pydantic import Field, create_model, ConfigDict, BaseModel
from rich import print
import instructor
import logfire


from Prometheus import Bridgette
from Prometheus.device import HueZone, HueResource, HueLight, HueRoom
from dotenv import load_dotenv
load_dotenv()
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
- [] ??step loop (is that even necessary if there are no steps per se and we're running continuously?)
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

    
Tool planning:
- Each tool will have an execute() method that will perform the action and update the state on the go
    - How to deal with async execution? How to ensure that the state is:
            - updated and read correctly?
                is that even a good idea? I doubt the model will be executing multiple commands at once,
                it might be a good idea however, to allow for command chaining like "turn lights on in the office, set the scene to relax and dim the lights to 50%"

                This could be achieved in the following ways:
                    - split the command into sub-commands and execute them one by one
                    - update Bridgette to allow for doing both i.e. turning on AND setting brightness
                        - basically tuning the functions to make them more robust, handling multiple action at once
                        - 'set scene' should automatically turn on the lights in the zone if they are off (it might do so now?)
                                                                                                          - checked, it does
                                                                                                        

Considerations:
- Is summarise() even necessary considering that this is not a chat agent and we're not keeping track of the conversation history? It'll solely operate based on commands given to it so no history is needed
    - this might allow me to decrease the complexity of the agent and focus on executing commands rather than keeping track of the conversation history and theoretically, the model shouldn't get as confused as the others.

- Should I implement a step loop? I don't think so, as the agent will operate on commands given to it it'll
  The model will run continuously and will execute commands as they come in
"""

###################################################################
################### TOOL AND OUTPUT DEFINITIONS ###################
###################################################################

#############################################################
######### STATE MANAGER AND DEPENDENCIES DEFINITION #########
#############################################################
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
    thinking: str = Field(description="Think about the action to be executed. What action does the user want to perform?")

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


class Action(BaseModel):
    thinking: str = Field(description="Think about the action to be executed. What action does the user want to perform?")

    selected_action: Union[TurnOn, TurnOff, SetScene, SetBrightness, Dim, SetTemperature] = Field(description="The type of action to be performed.")
    command: Command = Field(description="The details of the command to be executed.")


class TurnOn(BaseModel):
    """Turn on the light or a zone"""
    thinking: str = Field(description="Think about the command to turn on the lights in the specified zone.")
    action_type: Literal["turn_on"] = "turn_on"
    command: Command = Field(description="Details of the action to be performed")

    def execute(self, state: StateManager, deps: DependenciesManager, command: Command) -> None:
        if not command.zone:
            raise ValueError("Zone must be specified to turn on the lights.") 

        if command.light:
            # Turn on a specific light in the specified zone
            deps.bridge.zones[command.zone].devices[command.light].turn_on()
        elif command.scene == "all":
            deps.bridge.turn_on_all_lights()
        else:
            deps.bridge.zones[command.zone].turn_on()
        
        state.bridge_state = deps.bridge.get_current_state()

class TurnOff(BaseModel):
    """Turn off the light or a zone"""
    thinking: str = Field(description="Think about the command to turn off the lights in the specified zone.")
    action_type: Literal["turn_off"] = "turn_off"
    command: Command = Field(description="Details of the action to be performed")

    @logfire.instrument('turn_off', extract_args=True, record_return=True)
    def execute(self, state: StateManager, deps: DependenciesManager, command: Command) -> None:
        if not command.zone:
            raise ValueError("Zone must be specified to turn off the lights.")
        if command.light:
            # Turn off a specific light in the specified zone
            deps.bridge.zones[command.zone].devices[command.light].turn_off()
        elif command.scene == "all":
            deps.bridge.turn_off_all_lights()
        else:
            deps.bridge.zones[command.zone].turn_off()
        
        state.bridge_state = deps.bridge.get_current_state()


class SetScene(BaseModel):
    """Set the selected scene in the specified zone"""
    thinking: str = Field(description="What scene does the user what to set?")
    action_type: Literal["set_scene"] = "set_scene"
    command: Command = Field(description="Details of the action to be performed")

    @logfire.instrument('set_scene', extract_args=True, record_return=True)
    def execute(self, state: StateManager, deps: DependenciesManager, command: Command) -> None:
        if not command.zone:
            raise ValueError("Zone must be specified to set the scene.")
        if not command.scene:
            raise ValueError("Scene must be specified to set the scene.")
        
        # Set the scene in the specified zone
        deps.bridge.zones[command.zone].set_scene(command.scene)
        
        state.bridge_state = deps.bridge.get_current_state()

class SetBrightness(BaseModel):
    """Set the brightness of the specified zone or light.
    Can be set in absolute terms (e.g. 50) or relative terms (e.g. increase by 20%)."""
    thinking: str = Field(description="What brightness does the user want to set?")
    action_type: Literal["set_brightness"] = "set_brightness"
    command: Command = Field(description="Details of the action to be performed")

    @logfire.instrument('set_brightness', extract_args=True, record_return=True)
    def execute(self, state: StateManager, deps: DependenciesManager, command: Command) -> None:
        if not command.zone:
            raise ValueError("Zone must be specified to set the brightness.")
        if not command.brightness:
            raise ValueError("Brightness must be specified to set the brightness.")
        
        # Set the brightness in the specified zone or light
        if command.light:
            current_brightness = state.bridge_state['zones'][command.zone]['devices'][command.light]["brightness"]
            if isinstance(command.brightness.brightness, float):
                if command.brightness.relative:
                    new_brightness = current_brightness + (current_brightness * command.brightness)
                else:
                    new_brightness = command.brightness.brightness

            elif isinstance(command.brightness.brightness, int):
                if command.brightness.relative:
                    new_brightness = current_brightness + command.brightness.brightness
                else:
                    new_brightness = command.brightness.brightness
            
            deps.bridge.zones[command.zone].devices[command.light].change_brightness(new_brightness)
        else:
            on_devices = {name:state['brightness'] for name, state in state.bridge_state['zones'][command.zone]['devices'].items() if state['state']=='on'}

            avg_brightness = sum(on_devices.values()) / len(on_devices)

            if isinstance(command.brightness.brightness, float):
                if command.brightness.relative:
                    if command.brightness.up_down == 'down':
                        delta_brightness = -command.brightness.brightness
                    new_brightness = avg_brightness + (avg_brightness * delta_brightness)
                else:
                    new_brightness = command.brightness.brightness
            elif isinstance(command.brightness.brightness, int):
                if command.brightness.relative:
                    if command.brightness.up_down == 'down':
                        delta_brightness = -command.brightness.brightness
                    new_brightness = avg_brightness + delta_brightness
                else:
                    new_brightness = command.brightness.brightness

            deps.bridge.zones[command.zone].change_brightness(new_brightness)

        state.bridge_state = deps.bridge.get_current_state()


class SetTemperature(BaseModel):
    """Set the temperature of the specified zone or light."""
    thinking: str = Field(description="How warm does the user want the lights to be?")
    action_type: Literal["set_temperature"] = "set_temperature"
    command: Command = Field(description="Details of the action to be performed")
    
    @logfire.instrument('set_temperature', extract_args=True, record_return=True)   
    def execute(self, state: StateManager, deps: DependenciesManager, command: Command) -> None:
        if not command.zone:
            raise ValueError("Zone must be specified to set the temperature.")
        if not command.temperature:
            raise ValueError("Temperature must be specified to set the temperature.")
        
        # Set the temperature in the specified zone or light
        if command.light:
            deps.bridge.zones[command.zone].devices[command.light].change_temp(command.temperature)
        else:
            deps.bridge.zones[command.zone].change_temp(command.temperature)
        
        state.bridge_state = deps.bridge.get_current_state()

class Dim(BaseModel):
    """Dim the specified zone or light by 50% of its current brightness."""
    thinking: str = Field(description="Where does the user want to dim the lights?")
    action_type: Literal["dim"] = "dim"
    command: Command = Field(description="Details of the action to be performed")

    @logfire.instrument('dim', extract_args=True, record_return=True)
    def execute(self, state: StateManager, deps: DependenciesManager, command: Command) -> None:
        if not command.zone:
            raise ValueError("Zone must be specified to dim the lights.")
        
        if command.light:
            # Dim a specific light in the specified zone
            current_brightness = state.bridge_state['zones'][command.zone]['devices'][command.light]["brightness"]
            new_brightness = current_brightness * 0.5
            deps.bridge.zones[command.zone].devices[command.light].change_brightness(new_brightness)
        else:
            current_brightness
        
        state.bridge_state = deps.bridge.get_current_state()


        
AGENTACTIONS = Union[TurnOn, 
                     TurnOff, 
                     SetScene, 
                     SetBrightness, 
                     SetTemperature, 
                     Dim
                     ]

TEST_MODEL_LARGE_FREE = "qwen/qwen3-235b-a22b:free"


class Agent:

    @logfire.instrument('agent_initialisation', extract_args=True, record_return=True)
    def __init__(self, 
                 api_key: str = os.getenv("OPENROUTER_API_KEY"),
                 base_url: str = os.getenv("OPENROUTER_BASE_URL"),
                 max_retries: int = 3,
                 model=TEST_MODEL_LARGE_FREE,
                 benchmarking: bool = False) -> None:

        if "localhost" in base_url:
            api_key = "EMPTY"
            base_url = "http://localhost:8000/v1"
        
        # if benchmarking:
        #     self.bridge = None
        #     self.state = StateManager(bridge_state={})
        # else:
        self.bridge: Bridgette = Bridgette()
        self.state: StateManager = StateManager(bridge_state=self.bridge.get_current_state())
        self.deps: DependenciesManager = DependenciesManager(client=instructor.from_openai(OpenAI(api_key=api_key,
                                                                                                  base_url=base_url)),
                                                            max_retries=max_retries,
                                                            bridge=self.bridge,
                                                            benchmarking=benchmarking)
        if "localhost" in base_url:
            self.deps.model = self.deps.client.models.list().data[0].id
        else:
            self.deps.model = model


        self.SYS_PROMPT = self._build_sys_prompt(self.deps, self.state)
        logfire.instrument_openai()
        logfire.info(f"Model: {self.deps.model}")

    @logfire.instrument('executing_action', extract_args=True, record_return=True)
    async def action(self, user_prompt: str) -> Union[None, AGENTACTIONS]:
        try:
            print("I'm trying action yo")
            logfire.info(f"User prompt: {user_prompt}")
            action = self.deps.client.chat.completions.create(
                model=self.deps.model,
                response_model=AGENTACTIONS,
                messages=[
                    {"role":"system", "content": self.SYS_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_retries=self.deps.max_retries,
                temperature=0.1)
            
            if self.deps.benchmarking:
                return action.model_dump()
            
            action.execute(self.state, self.deps, action.command)
            
        except Exception as e:
            logfire.error(f"Error executing action: {e}")
            return None
    
    def format_sections(self, data: dict) -> str:
        """Return a bullet-formatted string from a dict[str, list[str]]."""
        lines = []
        for section, items in data.items():
            lines.append(f"* {section}:")
            for item in items:
                lines.append(f"    - {item}")
        return "\n".join(lines)
    
    def _build_sys_prompt(self, deps: DependenciesManager, state:StateManager) -> str:
        zones = ",\n".join(deps.bridge.zones.keys())
        zone_devices = {zone:list(val['devices'].keys()) for zone, val in state.bridge_state['zones'].items()}

        zone_devices = self.format_sections(zone_devices)


        SYS_PROMPT_COMMAND = f"""You are an assistant parsing a command to be executed by a light controlling system.
        You have access to the following tools:
        - turn_on - Turns on the selected lights or zone
        - turn_off - Turns off the selected lights or zone
        - set_scene - Sets the scene in the selected zone
        - set_brightness - Sets the brightness of the selected lights or zone
        - set_temperature - Sets the temperature of the selected lights or zone

        Each tool requires:
        - thinking: Your reasoning for selecting this action
        - action_type: The exact name of the selected tool
        - command: A structured command containing the details of the action to be performed.

        The available zones are: 
        {zones}
        The available devices in each zone are: {zone_devices}

        Keep the replies short and concise, focusing on the action to be performed.
        """
        return textwrap.dedent(SYS_PROMPT_COMMAND)
        

if __name__ == "__main__":
    user_query = ''
    logfire.configure(token=os.environ.get("LOGFIRE_TOKEN"), console=False)
    with logfire.span("Agent Run"):
        base_url = "http://localhost:8000/v1"
        agent = Agent(base_url=base_url)
        while True and user_query.lower() != "exit":
            try:
                user_query = input("Enter your command: ")
                agent.action(user_query)
            except KeyboardInterrupt:
                print("\nExiting...")
                break