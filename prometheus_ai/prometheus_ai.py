from __future__ import annotations

import os
import textwrap
import asyncio
from typing import Union

from openai import AsyncOpenAI
from rich import print
import instructor
import logfire
from dotenv import load_dotenv
load_dotenv()

from Prometheus import Bridgette
from utils.project_types import StateManager, DependenciesManager
from utils.agent_tools import turn_off, turn_on, set_scene, set_brightness, set_temperature

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
AGENTACTIONS = Union[turn_on, 
                     turn_off, 
                     set_scene, 
                     set_brightness, 
                     set_temperature, 
                    #  Dim
                     ]

class Agent:

    @logfire.instrument('agent_initialisation', extract_args=True, record_return=True)
    def __init__(self, 
                 provider: str = "openrouter",
                 max_retries: int = 3,
                 model: str = None,
                 benchmarking: bool = False,
                 mode: instructor.Mode = None,
                 max_tokens: int = 200) -> None:

        PROVIDERS = {
            "local": {
                "base_url": "http://localhost:8000/v1",
                "api_key_env": None,
                "default_api_key": "EMPTY"
            },
            "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
                "default_api_key": None
            },
            "anthropic": {
                "base_url": "https://api.anthropic.com/v1/",
                "api_key_env": "ANTHROPIC_API_KEY",
                "default_api_key": None
            },
            "deepinfra": {
                "base_url": "https://api.deepinfra.com/v1/openai",
                "api_key_env": "DEEPINFRA_API_KEY",
                "default_api_key": None
            },
            "openai": {
                "base_url": "",
                "api_key_env": "OPENAI_API_KEY",
                "default_api_key": None
            }
        }
        if provider not in PROVIDERS:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: {list(PROVIDERS.keys())}")
        
        provider_config = PROVIDERS[provider]

        base_url = provider_config["base_url"]
        
        if provider_config["api_key_env"]:
            api_key = os.getenv(provider_config["api_key_env"])
            if api_key is None:
                raise ValueError(f"API key not found in environment variable {provider_config['api_key_env']} for provider {provider}")
        else:
            api_key = provider_config["default_api_key"]
        
        if benchmarking:
            self.bridge = None
            self.state = StateManager(bridge_state={})
        else:
            self.bridge: Bridgette = Bridgette()
            self.state: StateManager = StateManager(bridge_state=self.bridge.get_current_state())

        try:
            if provider == "openai":
                openai_client = AsyncOpenAI()
            else:
                openai_client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url)
        except Exception as e:
            logfire.error(f"Error initializing OpenAI client: {e}")
            raise e
        
        # Use provided mode if given, otherwise use default logic
        if mode is not None:
            selected_mode = mode
        elif provider == 'local':
            selected_mode = instructor.Mode.TOOLS
        else:
            selected_mode = instructor.Mode.JSON
            
        self.deps: DependenciesManager = DependenciesManager(client=instructor.from_openai(openai_client, 
                                                                                        mode=selected_mode
                                                                                           ),
                                                            max_retries=max_retries,
                                                            bridge=self.bridge,
                                                            benchmarking=benchmarking,
                                                            model=model,
                                                            max_tokens=max_tokens)

        self.SYS_PROMPT = self._build_sys_prompt(self.deps, self.state)
        logfire.instrument_openai()
        logfire.info(f"Model: {self.deps.model}")
        logfire.info(f"Mode: {selected_mode}")
        logfire.info(f"Deps: {self.deps.model_dump()}")

    @logfire.instrument('executing_action', extract_args=True, record_return=True)
    async def action(self, user_prompt: str) -> Union[dict, AGENTACTIONS]:
        try:
            logfire.info(f"User prompt: {user_prompt}")
            action = await self.deps.client.chat.completions.create(
                    model=self.deps.model,
                    response_model=AGENTACTIONS,
                    messages=[
                        {"role":"system", "content": self.SYS_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_retries=self.deps.max_retries,
                    temperature=0.2,
                    max_tokens=self.deps.max_tokens,
                    presence_penalty=1.5,
                    top_p=0.9,
                    extra_body={
                                "repetition_penalty": 1.05,
                                "top_k": 20, 
                                # "chat_template_kwargs": {"enable_thinking": False},
                                "min_p":0
                                }
                    )
            if self.deps.benchmarking:
                return action
            
            action.execute(self.state, self.deps, action.command)
            
        except Exception as e:
            logfire.error(f"Error executing action: {e}")
            return {"error": str(e), 'error_type': type(e).__name__}

    def format_sections(self, data: dict) -> str:
        """Return a bullet-formatted string from a dict[str, list[str]]."""
        lines = []
        for section, items in data.items():
            lines.append(f"* {section}:")
            for item in items:
                lines.append(f"    - {item}")
        return "\n".join(lines)
    
    def _build_sys_prompt(self, deps: DependenciesManager, state:StateManager) -> str:
        if deps.benchmarking:
            zones = "office, lounge, lounge floor lights, bedroom, all, tv"
            zone_devices = {
                "office": ["desk", "ceiling", "floor"],
                "lounge": ["standing", "flartsy", "tv1", "tv2"],
                "bedroom": ["ceiling", "ceiling"],
                "tv": ["sub"],
                "lounge floor lamps":["standing", "flartsy"],

            }
            zone_devices = self.format_sections(zone_devices)
        else:
            zones = ",\n".join(deps.bridge.zones.keys())
            zone_devices = {zone:list(val['devices'].keys()) for zone, val in state.bridge_state['zones'].items()}
            zone_devices = self.format_sections(zone_devices)

        SYS_PROMPT_COMMAND = f"""You are an assistant parsing a command to be executed by a light controlling system.
        #You have access to the following tools:
        - turn_on - Turns on the selected lights or zone
        - turn_off - Turns off the selected lights or zone
        - set_scene - Sets the scene in the selected zone
        - set_brightness - Sets the brightness of the selected lights or zone
        - set_temperature - Sets the temperature of the selected lights or zone

        #Each tool requires:
        - think: Your reasoning for selecting this action
        - action_type: The exact name of the selected tool
        - command: A structured command containing the details of the action to be performed.

        #Instructions:
        * Keep your responses concise and focused on the action to be performed.
        * If set_temperature, do not set brightness level

        #The available zones are: 
        {zones}
        #The available devices in each zone are: {zone_devices}
        """
        return textwrap.dedent(SYS_PROMPT_COMMAND)
        

async def main():
    user_query = ''
    logfire.configure(token=os.environ.get("LOGFIRE_WRITE_TOKEN_PROMETHEUS"), console=False)
    with logfire.span("Agent Run"):
        agent = Agent(provider="local", model="Qwen3-0.6B")
        while True and user_query.lower() != "exit":
            try:
                user_query = input("Enter your command: ")
                if user_query.lower() == "exit":
                    break
                _ = await agent.action(user_query)
            except KeyboardInterrupt:
                print("\nExiting...")
                break

if __name__ == "__main__":
    asyncio.run(main())