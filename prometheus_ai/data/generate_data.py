from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Literal, Union
import itertools
from pydantic import BaseModel, Field, field_validator, model_validator
import instructor
import asyncio
import json
from rich import print
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
from datasets import Dataset, load_dataset
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.project_types import Brightness, Scenario

from dotenv import load_dotenv
load_dotenv()

BASE_URL_OPENROUTER = os.getenv("OPENROUTER_BASE_URL")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_FREE = "deepseek/deepseek-chat-v3-0324:free"
MODEL = "deepseek/deepseek-chat-v3-0324"
DATA_DIR = Path("./synth_commands")

semaphore = asyncio.Semaphore(20)
  

# class SynthCommand(BaseModel):
#     full_command: str = Field(description="A command, including both wakeword phrase and an action, a user would give to the light controlling system.",
#                          examples=["Hey Bridgette, turn on the lights in the office",
#                                    "Heeey Bridgette!! Turn off the lights in the lounge", 
#                                    "Hi Bridgette, dim the lights in the lounge room", 
#                                    "Bridgette, set the lights to blue",
#                                    "Bridgette please set the brightness to 50%",
#                                    "Hiiiii Bridgette set scene to natural light",
#                                    "Bridgette, decrease the temperature in the office by 30%"]
#                                    )
#     wakeword_phrase: str = Field(description="The wakeword phrase to trigger the command.",
#                                  examples=["Hey Bridgette", "Heeey Bridgette!!", "Hi Bridgette", "Bridgette"],)
    
#     action_type: Literal["turn_on", "turn_off",
#                      "set_brightness", "set_scene", "set_temperature"] = Field(description="The action to be performed in the specified zone.")
    
#     zone: Literal["office", "lounge","lounge floor lights", "bedroom", "all","tv"] = Field(description="The name of the zone where the command will be executed.")

#     light: Union[str, None] = Field(
#         description="The name of the light where the command will be executed", default=None,)                

#     scene: Union[Literal['natural light', 'bloodbath', 'rest', 'disturbia', 'relax', 
#                              'concentrate', 'read', 'warm embrace', 'galaxy', 'phthalocyanine green love', 'starlight',
#                              'tri colour', 'shrexy', 'nightlight', 'energize', 'vapor wavey', 'dimmed', 'valley dawn', 'soho '], None] = Field(description="The scene to be set in the specified zone. A scene can be set only on an entire zone, not on a specific light.",
#                                                                                                                                          default=None)
#     temperature: Union[int, None] = Field(description="The warmth of the light",
#                                        ge=153, le=500,
#                                        examples=[153, 200, 300, 400, 500],
#                                        default=None)
#     brightness_value: Optional[Union[int, float, None]] = Field(description="The brightness level. A float in (0,1] if relative; an int in [1,100] if absolute; None otherwise.",
#                                        default=None, ge=0, le=100)
#     brightness_mode: Optional[Literal["absolute", "relative", None]] = None
#     brightness_direction: Optional[Literal["up", "down", None]] = None

# class SynthResponse(BaseModel):
#     commands: List[SynthCommand] = Field(description="A list of commands that can be executed by the light controlling system.")


PLUGS = {"standing", "flartsy", "sub"}
ZONES = ("office","lounge","lounge floor lights","bedroom","all","tv")
SCENES = (
    "natural light","relax","bloodbath","rest","disturbia","energize",
    "concentrate","read","warm embrace","galaxy","phthalocyanine green love",
    "starlight","tri colour","shrexy","nightlight","vapor wavey","dimmed",
    "valley dawn","soho"
)

class SynthCommand(BaseModel):
    full_command: str = Field(...)
    wakeword_phrase: str = Field(...)

    action_type: Literal["turn_on","turn_off","set_brightness","set_scene","set_temperature"]
    zone: Literal[*ZONES]
    light: Optional[str] = None
    scene: Optional[Literal[*SCENES]] = None
    temperature: Optional[int] = Field(default=None, ge=153, le=500)

    # flattened brightness
    brightness_value: Optional[Union[int, float]] = None
    brightness_mode: Optional[Literal["absolute","relative"]] = None
    brightness_direction: Optional[Literal["up","down"]] = None

    @field_validator("light")
    @classmethod
    def strip_light(cls, v):
        return v.strip() if isinstance(v, str) else v

    @model_validator(mode="after")
    def check_consistency(self):
        a = self.action_type

        # scene & temperature exclusivity
        if a == "set_scene":
            if self.scene is None:
                raise ValueError("set_scene requires `scene`.")
            if self.temperature is not None:
                raise ValueError("set_scene must not include `temperature`.")
        else:
            if self.scene is not None and a != "set_scene":
                raise ValueError("`scene` only allowed for set_scene.")

        if a == "set_temperature":
            if self.temperature is None:
                raise ValueError("set_temperature requires `temperature`.")
            if self.scene is not None:
                raise ValueError("set_temperature must not include `scene`.")
        else:
            if self.temperature is not None and a != "set_temperature":
                raise ValueError("`temperature` only allowed for set_temperature.")

        # plugs: only on/off
        if self.light in PLUGS:
            if a not in {"turn_on","turn_off"}:
                raise ValueError(f"Plug `{self.light}` supports only turn_on/turn_off.")

        # brightness flattened rules
        if a == "set_brightness":
            if self.brightness_mode not in {"absolute","relative"}:
                raise ValueError("set_brightness requires brightness_mode {'absolute','relative'}.")

            if self.brightness_mode == "absolute":
                if not isinstance(self.brightness_value, int) or not (1 <= self.brightness_value <= 100):
                    raise ValueError("absolute mode requires brightness_value int 1–100.")
                if self.brightness_direction is not None:
                    raise ValueError("absolute mode must have brightness_direction = None.")

            if self.brightness_mode == "relative":
                if not (isinstance(self.brightness_value, float) and 0.0 < self.brightness_value <= 1.0):
                    raise ValueError("relative mode requires brightness_value float in (0,1].")
                if self.brightness_direction not in {"up","down"}:
                    raise ValueError("relative mode requires brightness_direction in {'up','down'}.")

        else:
            # non brightness actions: all brightness fields must be None
            if any(x is not None for x in (self.brightness_value, self.brightness_mode, self.brightness_direction)):
                raise ValueError("brightness_* must be None unless action_type is set_brightness.")

        # 'all' zone rules (optional): disallow specific light when zone == all
        if self.zone == "all" and self.light is not None:
            raise ValueError("When zone == 'all', `light` must be None.")

        return self

class SynthResponse(BaseModel):
    commands: List[SynthCommand] = Field(description="A list of commands that can be executed by the light controlling system.")

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
    - tv

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
      - 'tv1' (light)
      - 'tv2' (light)
      - 'flartsy' (plug)
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
    
# async def main(n_runs:int = 100):
    
#     tasks = [generate_all_commands() for _ in range(n_runs)]
#     results = await tqdm.gather(*tasks)
#     results = list(itertools.chain.from_iterable(results))
#     print(len(results))
#     return results


# keep your existing `generate_commands(n_queries: int, model: str) -> List[SynthCommand]`

async def _worker(sema: asyncio.Semaphore, n_queries: int, model: str):
    async with sema:
        try:
            return await generate_commands(n_queries=n_queries, model=model)
        except Exception as e:
            print(f"[warn] batch failed: {e}")
            return []  # keep going

async def main(
    n_runs: int = 500,
    n_queries: int = 10,
    concurrency: int = 8,
    model: str = MODEL,
    sink_path: str | None = None,
):
    sema = asyncio.Semaphore(concurrency)
    tasks = [asyncio.create_task(_worker(sema, n_queries, model)) for _ in range(n_runs)]

    results = []
    f = open(sink_path, "a", encoding="utf-8") if sink_path else None
    try:
        for coro in tqdm.as_completed(tasks, total=n_runs, desc="Generating"):
            batch = await coro  # [] if failed
            if not batch:
                continue
            results.extend(batch)
            if f:
                for cmd in batch:
                    f.write(json.dumps(cmd.model_dump(), ensure_ascii=False) + "\n")
                f.flush()
    finally:
        if f:
            f.close()

    print(f"total commands: {len(results)}")
    return results


results = await main(n_runs=500)

print(results)
print(len(results))
# len(results)
# results[0].model_dump_json
serialised_results=[res.model_dump_json() for res in results]

list_scenarios=[]
id=0
with open(DATA_DIR/"synth_commands_5k_v5.jsonl", "a", encoding="utf-8") as file:
  for res in results:
        file.write(
            Scenario(
                id=id,
                full_command=res.full_command,
                wakeword_phrase=res.wakeword_phrase,
                action_type=res.action_type,
                zone=res.zone,
                light=res.light,
                scene=res.scene,
                temperature=res.temperature,
                brightness_value=res.brightness_value,
                brightness_mode=res.brightness_mode,
                brightness_direction=res.brightness_direction,
                split="train" if id < 4000 else "test"
            ).model_dump_json() + "\n"
        )
        list_scenarios.append(
            Scenario(
                id=id,
                full_command=res.full_command,
                wakeword_phrase=res.wakeword_phrase,
                action_type=res.action_type,
                zone=res.zone,
                light=res.light,
                scene=res.scene,
                temperature=res.temperature,
                brightness_value=res.brightness_value,
                brightness_mode=res.brightness_mode,
                brightness_direction=res.brightness_direction,
                split="train" if id < 4000 else "test"
            ).model_dump_json()
        )
        id += 1
type(list_scenarios[0])
dict_scenarios = [json.loads(scenario) for scenario in list_scenarios]
len(dict_scenarios)
dict_no_bright = [x for x in dict_scenarios if x['action_type'] != 'set_brightness']
fixed_bright = [x for x in dict_scenarios if x['action_type'] == 'set_brightness' and x['brightness_value'] is not None]
len(fixed_bright), len(dict_no_bright)

new_dict_scenarios = fixed_bright + dict_no_bright
cleaned_new_dict_scenarios = []

for scenario in new_dict_scenarios:
    if scenario['action_type'] != "set_brightness":
        scenario['brightness_value'] = None
        scenario['brightness_mode'] = None
        scenario['brightness_direction'] = None
        cleaned_new_dict_scenarios.append(scenario)
    else:
        cleaned_new_dict_scenarios.append(scenario)


fixed_relative_bright =[]

for scenario in cleaned_new_dict_scenarios:
    if scenario['action_type'] == "set_brightness" and scenario['brightness_mode'] == 'relative' and scenario['brightness_value'] > 1.0:
        scenario['brightness_value'] = scenario['brightness_value'] / 100
        fixed_relative_bright.append(scenario)
    else:
        fixed_relative_bright.append(scenario)

fixed_absolute_bright = []

for scenario in fixed_relative_bright:
    if scenario['action_type'] == "set_brightness" and scenario['brightness_mode'] == 'absolute' and scenario['brightness_direction'] is not None:
        scenario['brightness_direction'] = None
        fixed_absolute_bright.append(scenario)
    else:
        fixed_absolute_bright.append(scenario)
        


len(fixed_absolute_bright)

train_scenarios = [scenario for scenario in fixed_absolute_bright if scenario['split'] == 'train']
test_scenarios = [scenario for scenario in fixed_absolute_bright if scenario['split'] == 'test']

print(len(train_scenarios), len(test_scenarios))

len([x for x in dict_scenarios if x['brightness_value'] ==0   and x['action_type'] == 'set_brightness'])

[x for x in dict_scenarios if x['id']==94]

len([x for x in fixed_absolute_bright if x['action_type'] == 'set_brightness' and x['brightness_mode']=='relative' and x['brightness_direction'] is None])

print([x for x in fixed_relative_bright if x['action_type'] == 'set_brightness' and x['brightness_mode']=='absolute' and x['brightness_direction'] is not None])

len([x for x in fixed_absolute_bright if x['action_type'] != 'set_brightness' and (x['brightness_value'] is not None or x['brightness_mode'] is not None or x['brightness_direction'] is not None)] )

print([x for x in fixed_absolute_bright if x['action_type'] != 'set_brightness' and (x['brightness_value'] is not None or x['brightness_mode'] is not None or x['brightness_direction'] is not None)])


hf_ds_train = Dataset.from_list(train_scenarios)
hf_ds_test = Dataset.from_list(test_scenarios)

hf_ds_train.push_to_hub("mbary/hue_commands_synth_5k_v6", private=True, split="train")
hf_ds_test.push_to_hub("mbary/hue_commands_synth_5k_v6", split="test")

# with open(DATA_DIR/"synth_commands3_3k.jsonl","w",encoding="utf-8") as file:
#     file.write(json.dumps(serialised_results, indent=4))
    # json.dump(serialised_results, file)

# if __name__=="__main__":
#     results = asyncio.run(main())
#     print(results)


#############################
##### DATA CLEANING FFS #####
#############################

data_train = load_dataset("mbary/hue_commands_synth_5k_v2", split="train") 
data_test = load_dataset("mbary/hue_commands_synth_5k_v2", split="test")
len(data_train),len(data_test)

all_scenarios = list(data_train) + list(data_test)
print(f"Total scenarios: {len(all_scenarios)}")


no_bright = [x for x in all_scenarios if x['brightness'] is None]
bright = [x for x in all_scenarios if x['brightness'] is not None]
len(no_bright), len(bright)
"""
So now, due to how the eval will be set up, I need to 
ensure that each scenario MUST HAVE a brightness object, but everything else can be None!


1. [X] Scenarios no brightness:
- So I have to check how many scnearios DO NOT HAVE brightness and then append it 
  with hte brightness object (Brightness(none, none, none))

- [X] I probably should ensure that the no bright objs are correct i.e. that the full command 
  CERTAINLY does NOT contain 'brightness' or something like this in the body


2. Scenarios with brightness:
- I have to ensure that the ones that have brightness are correct i.e. there are no 
    missing fields (where they're not supposed to be missing)
    Rules:
    - if 'increase' in the full command:
      - relative = True
      - up_down = 'up'
    - if 'decrease' in the full command:
      - relative = True
      - up_down = 'down'
    - if 'set' in the full command:
      - relative = False
      - up_down = None
    - ensure that all brightness levels are 0<=level<=100


- ensure that in the scenarios with brightness:
  - the temperature is None as that often was incorrect in the agent answers


3. All scnerrios:
All scnearios should contain the following fields:
- id
- full_command
- wakeword_phrase
- action_type
- zone
"""
#################
## 1 No bright ##
#################
cleaned_no_bright = []
for scenario in no_bright:
    scen = Scenario(**scenario)
    scen.brightness = Brightness(
        brightness=None, relative=None, up_down=None
    )
    cleaned_no_bright.append(scen)

type(cleaned_no_bright[0].model_dump_json())
len([x for x in cleaned_no_bright if 'brightness' in  x.full_command.lower()])
len(cleaned_no_bright)


##################
#### 2 Bright ####
##################
len([x for x in bright if 'increase' in x['full_command'].lower() and x['brightness']['relative'] is False] )
print([x for x in bright if 'increase' in x['full_command'].lower() and x['brightness']['relative'] is False])

len([x for x in bright if 'increase' in x['full_command'].lower() and x['brightness']['relative'] is None] )

len([x for x in bright if 'decrease' in x['full_command'].lower() and x['brightness']['relative'] is False] )
print([x for x in bright if 'decrease' in x['full_command'].lower() and x['brightness']['relative'] is False] )

# examples where relative SHOULD be "True"  but it's "False"
len([x for x in bright if 'by' in x['full_command'].lower() and x['brightness']['relative'] is False] )
print([x for x in bright if 'by' in x['full_command'].lower() and x['brightness']['relative'] is False] )
brightness_rel_true_but_false=[x for x in bright if 'by' in x['full_command'].lower() and x['brightness']['relative'] is False] 
for i in bright:
    if 'by' in i['full_command'].lower() and i['brightness']['relative'] is False:
        i['brightness']['relative'] = True

len([x for x in bright if 'set' in x['full_command'].lower() and x['brightness']['relative'] is True] )
len([x for x in bright if  x['brightness']['brightness'] >100] )
len([x for x in bright if  x['brightness']['brightness'] <0] )


type(bright[0])
cleaned_bright = [Scenario(**x) for x in bright]
type(cleaned_no_bright[0])

type(cleaned_bright[0])
type(cleaned_no_bright)
print(cleaned_bright[0])

##### Setting all relative to 'false' if null
cleaned_all = []

for scenario in all_scenarios:
    scen = Scenario(**scenario)
    if scen.brightness.relative is None:
        scen.brightness.relative = False

        cleaned_all.append(scen)
    else:
      cleaned_all.append(scen)
  
len(cleaned_all), len(all_scenarios)
type(cleaned_all[0])

len([x for x in cleaned_all if x.brightness.relative is None])

clean_train = [x.model_dump_json() for x in cleaned_all if x.split == "train"]
clean_test = [x.model_dump_json() for x in cleaned_all if x.split == "test"]

len(clean_train), len(clean_test)

hf_ds_train = Dataset.from_list([json.loads(x) for x in clean_train])
hf_ds_test = Dataset.from_list([json.loads(x) for x in clean_test])

# hf_ds_train.push_to_hub("mbary/hue_commands_synth_5k_v3", private=True, split="train")
# hf_ds_test.push_to_hub("mbary/hue_commands_synth_5k_v3", split="test")

###########################################################
len([scenario for scenario in all_scenarios if scenario['brightness']['relative'] is None])
len([scenario for scenario in all_scenarios if scenario['brightness']['relative'] is False])
len([scenario for scenario in all_scenarios if scenario['brightness']['relative'] is True])
print(all_scenarios[2]['brightness']['relative'])


cleanded_data_all = cleaned_no_bright + cleaned_bright
len(cleanded_data_all), len(all_scenarios)

len([x for x in cleanded_data_all if x.temperature and (x.temperature < 153 or x.temperature > 500)])
len([x for x in cleanded_data_all if x.light == "TV1"])
# normalise all values - to lower

for i in cleanded_data_all:
    i.light = i.light.lower() if i.light else None
    i.zone = i.zone.lower()
    i.scene = i.scene.lower() if i.scene else None

len([x for x in cleanded_data_all if 'color' in x.full_command.lower()])

len(cleanded_data_all)


clean_train = [json.loads(x.model_dump_json()) for x in cleanded_data_all if x.split == "train"]
clean_test = [json.loads(x.model_dump_json()) for x in cleanded_data_all if x.split == "test"]

hf_ds_train = Dataset.from_list(clean_train)
hf_ds_test = Dataset.from_list(clean_test)


# hf_ds_train.push_to_hub("mbary/hue_commands_synth_5k_v2", private=True, split="train")
# hf_ds_test.push_to_hub("mbary/hue_commands_synth_5k_v2", split="test")



#####################
### CHANGES TO V6 ###
#####################


ds_train = load_dataset("mbary/hue_commands_synth_5k_v6", split="train")
ds_test = load_dataset("mbary/hue_commands_synth_5k_v6", split="test")

all_data = list(ds_train) + list(ds_test)
print(all_data[:10])

import re
regex = r"\s+(\d+)"
regex2 = r"^[^\d]*$"
pat = re.compile(regex)
pat2 = re.compile(regex2)
command = "'Hi Bridgette, set the brightness of the ceiling light in the bedroom to 75%'"
print(pat.search(command).group(1))
print(len(pat.search(command).group(0)))

print(pat.findall(command).group(1))

all_data[0]['full_command']

print(re.match(regex, all_data[0]['full_command']))

len(all_data)
len([x for x in all_data if x['action_type']=='set_temperature' and pat2.search(x['full_command'])])

temp__val = []
for i in all_data:
    if i['action_type']=='set_temperature' and pat2.search(i['full_command']):
        pass
    else:
        temp__val.append(i)

temp__val2 = []
for i in all_data:
    if i['action_type']=='set_temperature' and not pat.search(i['full_command']):
        pass
    else:
        temp__val2.append(i)


len(temp__val)
len(temp__val2)
len([x for x in temp__val2 if x['action_type']=='set_temperature' and pat.search(x['full_command'])])
len([x for x in temp__val2 if x['action_type']=='set_temperature' and pat2.search(x['full_command'])])
len([x for x in temp__val2 if x['action_type']=='set_temperature' and not pat.search(x['full_command'])])
print([x for x in temp__val2 if x['action_type']=='set_temperature' and not pat.search(x['full_command'])])
len([x for x in temp__val2 if x['action_type']=='set_temperature' and int(re.search(regex, x['full_command']).group(1))>500])
len([x for x in temp__val2 if x['action_type']=='set_temperature' and int(re.search(regex, x['full_command']).group(1))<153])
len([x for x in temp__val2 if x['action_type']=='set_temperature' and (int(re.search(regex, x['full_command']).group(1))<153 or int(re.search(regex, x['full_command']).group(1))>500)])

print([x for x in temp__val2 if x['action_type']=='set_temperature' and int(re.search(regex, x['full_command']).group(1))>500])
print([x for x in temp__val2 if x['action_type']=='set_temperature' and int(re.search(regex, x['full_command']).group(1))<153])

final_clean = []
for i in temp__val2:
    if i['action_type']=='set_temperature' and (int(re.search(regex, i['full_command']).group(1))<153 or int(re.search(regex, i['full_command']).group(1))>500):
        pass
    else:
        final_clean.append(i)
len(final_clean)

clean_train = [x for x in final_clean if x['split'] == 'train']
clean_test = [x for x in final_clean if x['split'] == 'test']   

clean_ds_train = Dataset.from_list(clean_train)
clean_ds_test = Dataset.from_list(clean_test)

# clean_ds_train.push_to_hub("mbary/hue_commands_synth_5k_v7", private=True, split="train")
# clean_ds_test.push_to_hub("mbary/hue_commands_synth_5k_v7", split="test")