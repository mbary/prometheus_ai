from __future__ import annotations

import argparse
import ast
import json
import os
import textwrap
import asyncio
from typing import Union
import re
from openai import AsyncOpenAI
from rich import print
import instructor
import weave
import logfire
from pydantic import ValidationError
from dotenv import load_dotenv
load_dotenv()

from Prometheus import Bridgette
from utils.project_types import StateManager, DependenciesManager, Command, Brightness
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

@weave.op()
def _extract_first_json_block(text: str) -> str | None:
    # Strip fences (```json / ``` JSON / bare ```), anywhere top/bottom
    s = (text or "").strip()
    s = re.sub(r"^```[a-zA-Z]*\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s, flags=re.IGNORECASE)

    # Already looks like a sole JSON object
    if s.startswith("{") and s.rstrip().endswith("}"):
        return s

    # Find first balanced {...} block inside any surrounding prose
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i, c in enumerate(s[start:], start=start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None

@weave.op()
def _loads_jsonish(block: str):
    # 1) strict JSON
    try:
        return json.loads(block)
    except Exception:
        pass
    # 2) double-encoded JSON (a JSON string containing JSON)
    try:
        tmp = json.loads(block)
        if isinstance(tmp, str):
            return json.loads(tmp)
    except Exception:
        pass
    # 3) permissive python literal (single quotes / True/False/None)
    try:
        return ast.literal_eval(block)
    except Exception:
        return None

class AgentStruct:

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
                                                            max_tokens=max_tokens,
                                                            mode=selected_mode)

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
                    # reasoning_effort="minimal",
                    # verbosity="low"
                    presence_penalty=1.5,
                    top_p=0.8,
                    extra_body={
                                "repetition_penalty": 1.05,
                                # "top_k": 20, 
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
                "lounge floor lights":["standing", "flartsy"],

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
    

class Agent:
    def __init__(self, 
                 provider: str = "local",
                 max_retries: int = 3,
                 model: str = None,
                 benchmarking: bool = False,
                 max_tokens: int = 258) -> None:
        self.provider = provider
        self.max_retries = max_retries
        self.model = model

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
                client = AsyncOpenAI()
            else:
                client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url)
        except Exception as e:            
            raise e
        
        self.deps: DependenciesManager = DependenciesManager(client=client,
                                                             bridge=self.bridge,
                                                             benchmarking=benchmarking,
                                                             model=model,
                                                             max_tokens=max_tokens,
                                                             max_retries=max_retries,) 

        self.SYS_PROMPT = self._build_sys_prompt(self.deps, self.state)

    def __format_sections(self, data: dict) -> str:
        """Return a bullet-formatted string from a dict[str, list[str]]."""
        lines = []
        for section, items in data.items():
            line = f"{section}: {', '.join(items)}"
            lines.append(line)
            # for item in items:
            #     lines.append(f"    - {item}")
        return "\n".join(lines)

    def _build_sys_prompt(self, deps: DependenciesManager, state:StateManager) -> str:
        if deps.benchmarking:
            zones = "office, lounge, lounge floor lights, bedroom, all, tv"
            zone_devices = {
                "office": ["desk", "ceiling", "floor"],
                "lounge": ["standing", "flartsy", "tv1", "tv2"],
                "bedroom": ["ceiling", "ceiling"],
                "tv": ["sub"],
                "lounge floor lights":["standing", "flartsy"],

            }
            zone_devices = self.__format_sections(zone_devices)
            zone_scenes = """\n*Lounge: Nightlight, Warm embrace, Galaxy, Starlight, Valley dawn\n*Office: Energize, Concentrate, Natural light, tri colour, vapor wavey, Shrexy, Soho, Relax, Rest, Dimmed, Read, Nightlight, Disturbia, Bloodbath, phthalocyanine green love\n*Bedroom: Natural light, Energize, Read, Concentrate, Nightlight, Rest, Relax""".lower()
        else:
            zones = ",\n".join(deps.bridge.zones.keys())
            zone_devices = {zone:list(val['devices'].keys()) for zone, val in state.bridge_state['zones'].items()}
            zone_devices = self.__format_sections(zone_devices)
            zone_scenes = {zone:list(val.scenes.keys()) for zone,val in deps.bridge.zones.items()}
            # zone_scenes = {zone:list(val['scenes'].keys()) for zone, val in state.bridge_state['zones'].items()}
            zone_scenes = self.__format_sections(zone_scenes)

        SYS_PROMPT = """You are an assistant parsing commands for a smart lighting system.

        Always respond with valid JSON in this exact format:
        {
        "think": "Your reasoning for this action",
        "action_type": "turn_on|turn_off|set_scene|set_brightness|set_temperature",
        "command": {
            "zone": "office|lounge|lounge floor lights|bedroom|all|tv",
            "light": "light_name or null",
            "scene": "scene_name or null", 
            "temperature": number_or_null,
            "brightness": {
            "brightness": number_or_null,
            "relative": true_or_false_or_null,
            "up_down": "up|down or null"
            }
        }
        }"""
        SYS_PROMPT += f"""\n\nAvailable zones: {zones}\n\nAvailable devices per zone:\n{zone_devices}\n\nAvailable scenes per zone:\n{zone_scenes}\n\n#Rules:
        - Only include relevant parameters for each action
        - Set irrelevant parameters to null
        - Temperature range: 153-500
        - Brightness range: 1-100
        - For relative brightness, "brightness" MUST be a decimal between 0 and 1 (e.g., 0.2 for 20%).
  Do NOT output 20 or 20% for relative deltas.
        - Output only a single JSON object. No prose."""

        # SYS_PROMPT += """
        # # Brightness rules:
        # - If relative: true -> `brightness` is a fractional delta in (0,1]; e.g., 0.2 means "+20%", 0.1 means "+10%".
        # - If relative: false -> `brightness` is an absolute level 1-100.
        # - Never mix units. Do not output 80 when relative=true; do not output 0.2 when relative=false.
        # #Examples:\n{"action_type":"set_brightness","command":{"zone":"office","light":null,"scene":null,
        # "temperature":null,"brightness":{"brightness":0.2,"relative":true,"up_down":"up"}}}
        # {"action_type":"set_brightness","command":{"zone":"office","light":null,"scene":null,"temperature":null,
        # "brightness":{"brightness":75,"relative":false,"up_down":null}}}
        # """

        SYS_PROMPT+="""
        # Brightness rules:
        - If relative: true -> `brightness` is a fractional delta in (0,1]; e.g., 0.2 means "+20%", 0.1 means "+10%".
        - If relative: false -> `brightness` is an absolute level 1-100.
        - Never mix units. Do not output 80 when relative=true; do not output 0.2 when relative=false.

        # Examples:
        {"action_type":"set_brightness","command":{"zone":"lounge","light":"standing","scene":null,"temperature":null,"brightness":{"brightness":0.15,"relative":true,"up_down":"up"}}}
        {"action_type":"set_brightness","command":{"zone":"bedroom","light":null,"scene":null,"temperature":null,"brightness":{"brightness":0.30,"relative":true,"up_down":"down"}}}
        {"action_type":"set_brightness","command":{"zone":"tv","light":null,"scene":null,"temperature":null,"brightness":{"brightness":42,"relative":false,"up_down":null}}}
        {"action_type":"set_scene","command":{"zone":"office","light":null,"scene":"Energize","temperature":null,"brightness":{"brightness":null,"relative":false,"up_down":null}}}"""

        return textwrap.dedent(SYS_PROMPT)

    @weave.op()
    async def generate_response(self, query: str):
        response = await self.deps.client.chat.completions.create(
            model=self.deps.model,
            messages=[
                {"role":"system", "content": self.SYS_PROMPT},
                {"role": "user", "content": query}
            ],
            temperature=0.2,
            top_p=0.9,
            max_tokens=256)
        return response.choices[0].message.content or ""

    # @weave.op()
    # def parse_completion_to_action(self, completion: str): ##TODO Add return type once determined
        # try:
        #     completion = completion.strip()
        #     if completion.startswith("```json"):
        #         completion = completion.replace("```json","").replace("```","").strip()
        #     data = json.loads(completion)

        #     action_type = data["action_type"]

        #     brightness_data = data["command"].get("brightness", {})
        #     brightness = None
        #     if brightness_data:
        #         brightness = Brightness(
        #             brightness=brightness_data.get("brightness"),
        #             relative=brightness_data.get("relative"),
        #             up_down=brightness_data.get("up_down"),
        #         )
        #     command = Command(
        #         zone=data["command"]["zone"],
        #         light=data["command"]["light"],
        #         scene=data["command"]["scene"],
        #         temperature=data["command"]["temperature"],
        #         brightness=brightness,
        #     )
        #     action_map = {
        #         "turn_on": turn_on, "turn_off": turn_off, "set_scene": set_scene,
        #         "set_brightness": set_brightness, "set_temperature": set_temperature,
        #     }
        #     return action_map[action_type](think=data["think"], command=command) if action_type in action_map else None
        # except (json.JSONDecodeError, KeyError, ValidationError, TypeError, AttributeError):
        #     return None
        
        # try:
        #     s = completion.strip()
        #     if s.startswith("```json"):
        #         s = s.replace("```json", "").replace("```", "").strip()
        #     data = json.loads(s)
        #     a = data.get("action_type")
        #     cmd = data.get("command", {}) or {}

        #     b_raw = cmd.get("brightness")
        #     brightness = None
        #     if isinstance(b_raw, dict):
        #         brightness = Brightness(
        #             brightness=b_raw.get("brightness"),
        #             relative=b_raw.get("relative"),
        #             up_down=b_raw.get("up_down"),
        #         )

        #     command = Command(
        #         zone=cmd.get("zone"),
        #         light=cmd.get("light"),
        #         scene=cmd.get("scene"),
        #         temperature=cmd.get("temperature"),
        #         brightness=brightness,
        #     )
        #     action_map = {
        #         "turn_on": turn_on, "turn_off": turn_off, "set_scene": set_scene,
        #         "set_brightness": set_brightness, "set_temperature": set_temperature,
        #     }
        #     if a in action_map:
        #         return action_map[a](think=data.get("think",""), command=command)
        #     return None
        # except Exception:
        #     return None

    @weave.op()  # <-- captures parse failures as exceptions with raw content attached
    def parse_completion_to_action(self, completion: str):
        ACTION_MAP = {
            "turn_on": turn_on,
            "turn_off": turn_off,
            "set_scene": set_scene,
            "set_brightness": set_brightness,
            "set_temperature": set_temperature,
        }
        block = _extract_first_json_block(completion)
        if not block:
            raise ValueError(f"No JSON block found. preview={ (completion or '')[:200] }")

        data = _loads_jsonish(block)
        if not isinstance(data, dict):
            raise ValueError(f"Parsed non-dict JSON. preview={ block[:200] }")

        action_type = data.get("action_type") or data.get("tool_name")
        if action_type not in ACTION_MAP:
            raise ValueError(f"Unknown action_type={action_type}. preview={ str(data)[:200] }")

        cmd = data.get("command") or data.get("arguments") or {}

        b = cmd.get("brightness")
        brightness = None
        if isinstance(b, dict):
            brightness = Brightness(
                brightness=b.get("brightness"),
                relative=b.get("relative"),
                up_down=b.get("up_down"),
            )

        try:
            command = Command(
                zone=cmd.get("zone"),
                light=cmd.get("light"),
                scene=cmd.get("scene"),
                temperature=cmd.get("temperature"),
                brightness=brightness,
            )
        except ValidationError as ve:
            # Let Weave capture the exception + inputs
            raise ValueError(f"Pydantic validation failed: {ve}") from ve

        return ACTION_MAP[action_type](think=data.get("think", ""), command=command)
        
    @weave.op()
    async def action(self, user_prompt: str) -> None:
        try:
            raw_response = await self.generate_response(user_prompt)
            action = self.parse_completion_to_action(raw_response)
            
            if self.deps.benchmarking:
                return action
            
            action.execute(self.state, self.deps, action.command)        
        except Exception as e:
            return {"error": str(e), "error_type": type(e).__name__}
    

async def main():

    parser = argparse.ArgumentParser(description="Run the Philips Hue Agent.")
    parser.add_argument("--provider", type=str, default="local", choices=["local", "openrouter", "anthropic", "deepinfra", "openai"], help="The AI provider to use.")
    parser.add_argument("--model", type=str, default=None, required=True ,help="The model to use.")
    args = parser.parse_args()

    user_query = ""


    weave.init("prometheus_ai_agent")
    while True and user_query.lower() != "exit":
        try:
            user_query = input("Enter your command: ")
            if user_query.lower() == "exit":
                break
            agent = Agent(provider=args.provider, model=args.model)
            _ = await agent.action(user_query)
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        finally:
            try:
                weave.finish()
            except Exception:
                pass            

# async def main():
#     user_query = ''
#     # logfire.configure(token=os.environ.get("LOGFIRE_WRITE_TOKEN_PROMETHEUS"), console=False)
#     with logfire.span("Agent Run"):
#         agent = Agent(provider=args["provider"], model=args["model"])
#         while True and user_query.lower() != "exit":
#             try:
#                 user_query = input("Enter your command: ")
#                 if user_query.lower() == "exit":
#                     break
#                 _ = await agent.action(user_query)
#             except KeyboardInterrupt:
#                 print("\nExiting...")
#                 break

if __name__ == "__main__":
    asyncio.run(main())
