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
from dotenv import load_dotenv
load_dotenv()

BASE_URL_OPENROUTER = os.getenv("OPENROUTER_BASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_FREE = "deepseek/deepseek-chat-v3-0324:free"
MODEL = "deepseek/deepseek-chat-v3-0324"
DATA_DIR = Path("./synth_commands")

semaphore = asyncio.Semaphore(5)

class SynthCommand(BaseModel):
    full_command: str = Field(description="A command, including both wakeword phrase and an action, a user would give to the light controlling system.",
                         examples=["Hey Bridgette, turn on the lights in the office",
                                   "Heeey Bridgette!! Turn off the lights in the kitchen", 
                                   "Hi Bridgette, dim the lights in the living room", 
                                   "Bridgette, set the lights to blue", "Change the light color to red",
                                   "Bridgette please set the brightness to 50%",
                                   "Hiiiii Bridgette set scene to natural light"]
                                   )
    wakeword_phrase: str = Field(description="The wakeword phrase to trigger the command.",
                                 examples=["Hey Bridgette", "Heeey Bridgette!!", "Hi Bridgette", "Bridgette"],)
    
    zone: Literal["office", "kitchen", "lounge","lounge floor lights", "bedroom", "all","tv"] = Field(description="The name of the zone where the command will be executed.")
                
    action: Literal["turn on", "turn off", "dim", "set color", "set brightness", "set scene", "set lights"] = Field(description="The action to be performed in the specified zone.",
                        examples=["turn on", "turn off", "dim", "set color", "set brightness", "set scene", "set lights"],
                        )

    scenes: Optional[Literal['natural light', 'relax, ', 'bloodbath', 'rest', 'disturbia', 'relax', 'energize ', 'concentrate', 'read', 'warm embrace', 'galaxy', 'phthalocyanine green love', 'starlight', 'tri colour', 'shrexy', 'nightlight', 'energize', 'vapor wavey', 'dimmed', 'valley dawn', 'soho ']] = Field(
        description="The scene to be set in the specified zone. A scene can be set only on an entire zone, not on a specific light.",
        default=None
        )

class SynthResponse(BaseModel):
    commands: List[SynthCommand] = Field(description="A list of commands that can be executed by the light controlling system.")



async def generate_commands(n_queries:int = 10,
                        model: str = MODEL_FREE
                        # model: str = MODEL
                        ) -> List[SynthCommand]:
    """Generate data for the light controlling system."""
    client = instructor.from_openai(AsyncOpenAI(base_url=BASE_URL_OPENROUTER, api_key=OPENROUTER_API_KEY))
    system_prompt = f"""You are an assistant that generates realistic commands a user would give to a light controlling system.
    
    Available commands:
    - dim
    - turn on
    - turn off
    - set brightness
    - set scene

    Available zones:
    - office
    - lounge
    - lounge floor lights
    - bedroom
    - all

    Scenes per zone:
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

    wakeword phrase variation:
    - The wakeword phrase **MUST** be a variation of `Hey Bridgette`

    - You can be creative and use any variation of the wakeword phrase you can think off

    commands:
    - A command can be **ONLY** for one zone at a time
      - You **CANNOT** create chained commands e.g. "hey bridgette, turn on all lights and dim them to 40%"
    - Scenes can **ONLY** be set for zones with listed scenes
      - Zones without scenes support only on/off functionality

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
        response_model=SynthResponse,
        max_tokens=4096,
        max_retries=5
    )

    return commands.commands

async def generate_all_commands():
    async with semaphore:
        return await generate_commands()
    
async def main(n_runs:int = 10):
    
    tasks = [generate_all_commands() for _ in range(n_runs)]
    results = await tqdm.gather(*tasks)
    results = list(itertools.chain.from_iterable(results))
    print(len(results))
    return results

results = await main(n_runs=100)
len(results)
results[0].model_dump_json
serialised_results=[res.model_dump_json() for res in results]
serialised_results[0]
with open(DATA_DIR/"synth_commands.json","w",encoding="utf-8") as file:
    json.dump(serialised_results, file)

# if __name__=="__main__":
#     results = asyncio.run(main())
#     print(results)