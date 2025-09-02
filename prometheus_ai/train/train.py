# import peft  # type: ignore # noqa: F401
# import unsloth  # type: ignore # noqa: F401

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Union, Literal
import logfire
from rich import print
import art
from art.local import LocalBackend
from art.utils import iterate_dataset
from art.utils.litellm import convert_litellm_choice_to_openai
from pydantic import BaseModel, Field
# import litellm
# litellm._turn_on_debug()
from litellm import acompletion
import weave
print(weave.__version__)

weave.init(project_name="prometheus_training")

sys.path.append(str(Path(__file__).parent.parent))

from utils.utils_types import  load_scenarios, score_action
from prometheus_ai import Agent
from utils.project_types import Scenario
from utils.agent_tools import turn_off, turn_on, set_scene, set_brightness, set_temperature, Action


class AGENTACTIONS(BaseModel): 
    action: Union[turn_on, 
                     turn_off, 
                     set_scene, 
                     set_brightness, 
                     set_temperature, 
                    #  Dim
                     ]


ROLLOUTS_PER_GROUP = 4

SYS_PROMPT="""You are an assistant parsing a command to be executed by a light controlling system.
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
        office, lounge, lounge floor lights, bedroom, all, tv
        #The available devices in each zone are: * office:
    - desk
    - ceiling
    - floor
* lounge:
    - standing
    - flartsy
    - tv1
    - tv2
* bedroom:
    - ceiling
    - ceiling
* tv:
    - sub
* lounge floor lamps:
    - standing
    - flartsy
"""


"""
For Now, I have chosen to train the Qwen2.5-1.5B-Instruct model.
It had a very decent success rate (91% after finetuning a few parameters) but
there was definintely place for improments with parameter-specific selection.
For example, 
"""

class TrajectoryTrain(art.Trajectory):
    final_answer: Union[str,None]=None
    scenario: Scenario = None
    action: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    total_score: Union[int,float,None] = None
    correct_tool: Optional[Union[int,None]] = None
    correct_zone: Optional[Union[int,None]] = None
    correct_scene: Optional[Union[int,None]] = None
    correct_light: Optional[Union[int,None]] = None
    correct_temperature: Optional[Union[int,None]] = None
    correct_brightness: Optional[Union[int,None]] = None
    correct_brightness_relative: Optional[Union[int,None]] = None
    correct_brightness_up_down: Optional[Union[int,None]] = None

@weave.op()
async def run_agent(
        model: Union[art.Model, str],
        scenario: Scenario,
        semaphore:asyncio.Semaphore) -> TrajectoryTrain:
    traj = TrajectoryTrain(
        reward=0.0,
        messages_and_choices=[]
    )

    if isinstance(model, art.Model):
        print("USING TRAINABLE MODEL")
        model_name=f"hosted_vllm/{model.name}"
        base_url=model.inference_base_url
        api_key=model.inference_api_key
        temperature=1

    else:
        model_name=f"hosted_vllm/{model}"
        base_url='http://localhost:8000/v1'
        api_key="None"
        temperature=0.2
        # temperature=1
    print(model)
    print(model_name)
    print(base_url)
    print(api_key)
    traj.messages_and_choices = [
        {'role':'system', 'content':SYS_PROMPT},
        {'role':'user', 'content':scenario.full_command}
    ]

    async with semaphore:

        try:
            response = await acompletion(
                model=model_name,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                # presence_penalty=1.5,
                # top_p=0.8,
                # extra_body={
                #     'repetition_penalty': 1.05,
                # },
                response_format=AGENTACTIONS,
                messages=traj.messages(),
                caching=False,
            )
            # print(response)
            # print(response.choices[0]['message'].content)
            # print(type(response.choices[0]['message'].content))
            act = json.loads(response.choices[0]['message'].content)
            resp_message = json.dumps(act)
            action = Action(**act['action'])
            traj.messages_and_choices.append(convert_litellm_choice_to_openai(response.choices[0]))
            traj.action=action

            return traj
        except Exception as e:
            traj.action=None
            traj.error=str(e)
            traj.messages_and_choices.append(convert_litellm_choice_to_openai(response.choices[0]))
            traj.error_type=type(e).__name__
            # if 'pydantic' in str(e).lower():
            #     print(str(e))
            return traj

@weave.op()
async def run_and_score_agent_train(model: Union[art.Model, str],
        scenario: Scenario,) -> TrajectoryTrain:
    semaphore = asyncio.Semaphore(16)
    traj = await run_agent(model, scenario, semaphore)

    if traj.action is None:
        traj.reward = 0.0
        return traj
    
    score = score_action(traj.action,scenario)
    normalized_total_score = sum(score.values()) / len(score) if len(score) > 0 else 0
    traj.reward=normalized_total_score

    traj.total_score=normalized_total_score
    traj.correct_tool=score['correct_tool']
    traj.correct_zone=score['correct_zone']
    traj.correct_scene=score['correct_scene']
    traj.correct_light=score['correct_light']
    traj.correct_temperature=score['correct_temperature']
    traj.correct_brightness=score['correct_brightness']
    traj.correct_brightness_relative=score['correct_brightness_relative']
    traj.correct_brightness_up_down=score['correct_brightness_up_down']

    

    return traj

# import os
# from dotenv import load_dotenv
# import art
# import peft
# load_dotenv()
# from rich import print
# print(os.environ)


async def train():

    train_dataset = load_scenarios(split="train", limit=1000)

    test_dataset = load_scenarios(split="test", limit=100)

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    model = art.TrainableModel(
        base_model=model_name,
        project="prometheus_training",
        name='qwen2.5-1.5B-first-train',)
    
    print("MODEL IN FUCKINF TRAIN")
    print(model)
    with LocalBackend() as backend:
        # await backend.register(model)
        await model.register(backend)

        training_iterator = iterate_dataset(train_dataset, groups_per_step=12, num_epochs=3)
        # agent = Agent(benchmarking=True, training=True, model=model)
        # print(training_iterator)
        # print(type(training_iterator))
        # for i in training_iterator:
        #     print("CHUJS")
        # for batch, epoch, global_step, epoch_step in training_iterator:
        for batch in training_iterator:
            # print(item)
            # print(type(item))
            # print(dir(item))
            groups=[]
            for scenario in batch.items:
                groups.append(
                    art.TrajectoryGroup(
                        trajectories=[run_and_score_agent_train(model,scenario)
                                      for _ in range(ROLLOUTS_PER_GROUP)]
                    )
                )
                finished_groups = await art.gather_trajectory_groups(groups)

                await model.train(finished_groups)

if __name__ == '__main__':
    asyncio.run(train())