from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Literal, Union
import itertools
from pydantic import BaseModel, Field
import instructor
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from dotenv import load_dotenv
load_dotenv()

BASE_URL_OPENROUTER = os.getenv("OPENROUTER_BASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_FREE = "deepseek/deepseek-chat-v3-0324:free"
MODEL = "deepseek/deepseek-chat-v3-0324"

class Zone(BaseModel):
    name: Literal["office", "kitchen", "living room", "bedroom", "all"] = Field(
        description="The name of the zone where the command will be executed.")
    

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
    
    zone: Zone = Field(description="The zone where the command will be executed.",)
    action: str = Field(description="The action to be performed in the specified zone.",
                        examples=["turn on", "turn off", "dim", "set color", "set brightness", "set scene", "set lights"],
                        )


class SynthResponse(BaseModel):
    commands: List[SynthCommand] = Field(description="A list of commands that can be executed by the light controlling system.")



async def generate_data(n_queries:int = 10,
                        model: str = MODEL_FREE) -> List[SynthCommand]:
    """Generate data for the light controlling system."""
    client = instructor.from_openai(AsyncOpenAI(base_url=BASE_URL_OPENROUTER, api_key=OPENROUTER_API_KEY))
    system_prompt = f"""You are an assistant that generates realistic commands a user would give to a light controlling system.
    
    Available commands:
    - dim
    - turn on
    - turn off

    Available zones:
    - kitchen
    - office
    - living room
    - bedroom
    - all

    wakeword phrase variation:
    - The wakeword phrase **MUST** be a variation of `Hey Bridgette`
        - You can do `Heeey Bridgette`; `Heeyy Bridgette`; `Bridgette` `Hiii Bridgette`etc
    - You can be creative and use any variation of the wakeword phrase you can think off

    commands:
    - A command can be **ONLY** for one zone
    - You **CANNOT** create chained commands e.g. "hey bridgette, turn on all lights and dim them to 40%"
    
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
        response_model=SynthResponse
    )

    return commands.commands

async def main(n_runs:int = 10):
    """
    Run generate_data n times
    Each run generates n_runs*10 commands
    """
    tasks = [generate_data() for _ in range(n_runs)]
    results = await tqdm.gather(*tasks)
    results = list(itertools.chain.from_iterable(results))
    print(len(results))
    # commands = [x.command for x in results]
    # return commands
    return results

results = await main()



from rich import print
print(results)
len(results)

type(results[0][0])
results[0][0]

if __name__=="__main__":
    results = asyncio.gather(main())
    print(results)

    