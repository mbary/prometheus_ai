from __future__ import annotations

from typing import Literal, Union
from pydantic import Field, BaseModel
import logfire

from utils.project_types import StateManager, DependenciesManager, Command


class Action(BaseModel):
    # thinking: str = Field(description="Think about the action to be executed. What action does the user want to perform?")

    selected_action: Union[turn_on, turn_off, set_scene, set_brightness, Dim, set_temperature] = Field(description="The type of action to be performed.")
    command: Command = Field(description="The details of the command to be executed.")


class turn_on(BaseModel):
    """Turn on the light or a zone"""
    thinking: str = Field(description="Think about the command to turn on the lights in the specified zone.")
    action_type: Literal["turn_on"] = "turn_on"
    command: Command = Field(description="Details of the action to be performed")

    @logfire.instrument('turn_on', extract_args=True, record_return=True)
    def execute(self, state: StateManager, deps: DependenciesManager, command: Command) -> None:
        if not command.zone:
            raise ValueError("Zone must be specified to turn on the lights.") 

        if command.light:
            deps.bridge.zones[command.zone].devices[command.light].turn_on()
        elif command.scene == "all":
            deps.bridge.turn_on_all_lights()
        else:
            deps.bridge.zones[command.zone].turn_on()
        
        state.bridge_state = deps.bridge.get_current_state()

class turn_off(BaseModel):
    """Turn off the light or a zone"""
    thinking: str = Field(description="Think about the command to turn off the lights in the specified zone.")
    action_type: Literal["turn_off"] = "turn_off"
    command: Command = Field(description="Details of the action to be performed")

    @logfire.instrument('turn_off', extract_args=True, record_return=True)
    def execute(self, state: StateManager, deps: DependenciesManager, command: Command) -> None:
        if not command.zone:
            raise ValueError("Zone must be specified to turn off the lights.")
        if command.light:
            deps.bridge.zones[command.zone].devices[command.light].turn_off()
        elif command.scene == "all":
            deps.bridge.turn_off_all_lights()
        else:
            deps.bridge.zones[command.zone].turn_off()
        
        state.bridge_state = deps.bridge.get_current_state()


class set_scene(BaseModel):
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

        deps.bridge.zones[command.zone].set_scene(command.scene)
        
        state.bridge_state = deps.bridge.get_current_state()

class set_brightness(BaseModel):
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


class set_temperature(BaseModel):
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
            current_brightness = state.bridge_state['zones'][command.zone]['devices'][command.light]["brightness"]
            new_brightness = current_brightness * 0.5
            deps.bridge.zones[command.zone].devices[command.light].change_brightness(new_brightness)
        else:
            current_brightness
        
        state.bridge_state = deps.bridge.get_current_state()