from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Literal, Union
import itertools
from pydantic import BaseModel, Field
import instructor
import asyncio
import json
from rich import print
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.project_types import Brightness

from dotenv import load_dotenv
load_dotenv()

BASE_URL_OPENROUTER = os.getenv("OPENROUTER_BASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_FREE = "deepseek/deepseek-chat-v3-0324:free"
MODEL = "deepseek/deepseek-chat-v3-0324"
DATA_DIR = Path("./synth_commands")

semaphore = asyncio.Semaphore(10)
  

class SynthCommand(BaseModel):
    full_command: str = Field(description="A command, including both wakeword phrase and an action, a user would give to the light controlling system.",
                         examples=["Hey Bridgette, turn on the lights in the office",
                                   "Heeey Bridgette!! Turn off the lights in the lounge", 
                                   "Hi Bridgette, dim the lights in the lounge room", 
                                   "Bridgette, set the lights to blue",
                                   "Bridgette please set the brightness to 50%",
                                   "Hiiiii Bridgette set scene to natural light",
                                   "Bridgette, decrease the temperature in the office by 30%"]
                                   )
    wakeword_phrase: str = Field(description="The wakeword phrase to trigger the command.",
                                 examples=["Hey Bridgette", "Heeey Bridgette!!", "Hi Bridgette", "Bridgette"],)
    
    action: Literal["turn_on", "turn_off",
                    #  "dim", 
                    #  "set_color", 
                     "set_brightness", "set_scene", "set_temperature"] = Field(description="The action to be performed in the specified zone.")
    
    zone: Literal["office", "lounge","lounge floor lights", "bedroom", "all","tv"] = Field(description="The name of the zone where the command will be executed.")

    light: Union[str, None] = Field(
        description="The name of the light where the command will be executed", default=None,)                

    scene: Union[Literal['natural light', 'relax, ', 'bloodbath', 'rest', 'disturbia', 'relax', 'energize ', 
                             'concentrate', 'read', 'warm embrace', 'galaxy', 'phthalocyanine green love', 'starlight',
                             'tri colour', 'shrexy', 'nightlight', 'energize', 'vapor wavey', 'dimmed', 'valley dawn', 'soho '], None] = Field(description="The scene to be set in the specified zone. A scene can be set only on an entire zone, not on a specific light.",
                                                                                                                                         default=None)
    temperature: Union[int, None] = Field(description="The warmth of the light",
                                       ge=153, le=500,
                                       examples=[153, 200, 300, 400, 500],
                                       default=None)
    brightness: Union[Brightness, None] = Field(default=None,)

    color: Union[str, None] = Field(description="The color to be set in the specified zone.",
                                 examples=["red", "blue", "green", "yellow", "purple", "orange", "pink", "white"],
                                 default=None)

class SynthResponse(BaseModel):
    commands: List[SynthCommand] = Field(description="A list of commands that can be executed by the light controlling system.")


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
    brightness: Union[Brightness, None]
    split: Literal["train", "test"]


async def generate_commands(n_queries:int = 10,
                        # model: str = MODEL_FREE
                        model: str = MODEL
                        ) -> List[SynthCommand]:
    """Generate data for the light controlling system."""
    client = instructor.from_openai(AsyncOpenAI(base_url=BASE_URL_OPENROUTER, api_key=OPENROUTER_API_KEY))
    system_prompt = f"""You are an assistant that generates realistic commands a user would give to a light controlling system.

    #Commands per device:
    - Light:
      - turn_on
      - turn_off
      - set_brightness
      - set_temperature
    - Plug:
      - turn_on
      - turn_off
    
    #Available commands:
    - turn_on
    - turn_off
    - set_brightness
    - set_scene
    - set_temperature
    
    #Available zones:
    - office
    - lounge
    - lounge floor lights
    - bedroom
    - all

    #Scenes per zone:
    - office:
      - 'phthalocyanine green love', 
      - 'soho' 
      - 'shrexy'
      - 'tri colour'
      - 'bloodbath'
      - 'relax', 
      - 'read'
      - 'energize' 
      - 'rest'
      - 'vapor wavey'
      - 'nightlight'
      - 'disturbia'
      - 'dimmed'
      - 'concentrate'
      - 'natural light'
    - lounge:
      - 'galaxy'
      - 'starlight'
      - 'valley dawn'
      - 'nightlight'
      - 'warm embrace'
    - bedroom:
      - 'energize'
      - 'nightlight'
      - 'read'
      - 'rest'
      - 'relax'
      - 'concentrate'
      - 'natural light'
    
    #Lights per zone:
    - office:
      - 'desk' (light)
      - 'ceiling' (light)
      - 'floor' (light)
    - lounge:
      - 'TV1' (light)
      - 'TV2' (light)
      - 'Flartsy' (plug)
      - 'standing' (plug)
    - lounge floor lights:
      - 'standing' (plug)
      - 'flartsy' (plug)
    - bedroom:
      - 'ceiling1' (light)
      - 'ceiling2' (light)
    - tv:
      - 'sub' (plug)

    #Wakeword phrase variation:
    - The wakeword phrase **MUST** be a variation of `Hey Bridgette`
    - You can be creative and use any variation of the wakeword phrase you can think off
    - The commands should vary from simple to complex, but should not exceed 20 words in total.

    #Command creation rules:
    - A command can be **ONLY** for one zone at a time
      - You **CANNOT** create chained commands e.g. "hey bridgette, turn on all lights and dim them to 40%"
    - Scenes can **ONLY** be set for zones with listed scenes
      - Zones without scenes support only on/off functionality
    - Plugs can **ONLY** be turned on or off

    Respond with a JSON object with the following structure:
    {SynthResponse.model_json_schema()}
    """

    user_prompt = f"""Generate {n_queries} diverse commands"""
    commands = await client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system", "content":system_prompt},
            {"role":"user", "content":user_prompt}
        ],
        response_model=Union[SynthResponse],
        max_tokens=8000,
        max_retries=5
    )

    return commands.commands

async def generate_all_commands():
    async with semaphore:
        return await generate_commands()
    
async def main(n_runs:int = 500):
    
    tasks = [generate_all_commands() for _ in range(n_runs)]
    results = await tqdm.gather(*tasks)
    results = list(itertools.chain.from_iterable(results))
    print(len(results))
    return results

results = await main(n_runs=400)

print(results)
# len(results)
# results[0].model_dump_json
serialised_results=[res.model_dump_json() for res in results]

list_scenarios=[]
id=0
with open(DATA_DIR/"synth_commands_4k.jsonl", "a", encoding="utf-8") as file:
  for res in results:
        file.write(
            Scenario(
                id=id,
                full_command=res.full_command,
                wakeword_phrase=res.wakeword_phrase,
                action_type=res.action,
                zone=res.zone,
                light=res.light,
                scene=res.scene,
                temperature=res.temperature,
                brightness=res.brightness,#.brightness if res.brightness else None,
                # brightness_relative=res.brightness.relative if res.brightness else None,
                # brightness_up_down=res.brightness.up_down if res.brightness else None,
                color=res.color,
                split="train" if id < 2500 else "test"
            ).model_dump_json() + "\n"
        )
        list_scenarios.append(
            Scenario(
                id=id,
                full_command=res.full_command,
                wakeword_phrase=res.wakeword_phrase,
                action_type=res.action,
                zone=res.zone,
                light=res.light,
                scene=res.scene,
                temperature=res.temperature,
                brightness=res.brightness,#.brightness if res.brightness else None,
                # brightness_relative=res.brightness.relative if res.brightness else None,
                # brightness_up_down=res.brightness.up_down if res.brightness else None,
                color=res.color,
                split="train" if id < 3200 else "test"
            ).model_dump_json()
        )
        id += 1
type(list_scenarios[0])
dict_scenarios = [json.loads(scenario) for scenario in list_scenarios]
train_scenarios = [scenario for scenario in dict_scenarios if scenario['split'] == 'train']
test_scenarios = [scenario for scenario in dict_scenarios if scenario['split'] == 'test']
from datasets import Dataset
hf_ds_train = Dataset.from_list(train_scenarios)
hf_ds_test = Dataset.from_list(test_scenarios)

# hf_ds_train.push_to_hub("mbary/hue_commands_synth_4k", private=True, split="train")
# hf_ds_test.push_to_hub("mbary/hue_commands_synth_4k", split="test")

# with open(DATA_DIR/"synth_commands3_3k.jsonl","w",encoding="utf-8") as file:
#     file.write(json.dumps(serialised_results, indent=4))
    # json.dump(serialised_results, file)

# if __name__=="__main__":
#     results = asyncio.run(main())
#     print(results)