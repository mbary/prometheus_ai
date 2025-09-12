import unsloth
import json 
import torch
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel
from pydantic import ValidationError
import weave
import wandb
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.utils_types import load_scenarios, score_action, Scenario
from utils.agent_tools import turn_off, turn_on, set_brightness, set_scene, set_temperature
from utils.project_types import Command, Brightness

AGENTACTIONS = Union[turn_on, turn_off, set_scene, set_brightness, set_temperature]


project_name="mbaryp2-mbary/grpo_training4"
entity,project = project_name.split("/")


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
}

Available zones: office, lounge, lounge floor lights, bedroom, all, tv

Available devices per zone:
* office: desk, ceiling, floor
* lounge: standing, flartsy, tv1, tv2  
* bedroom: ceiling, ceiling
* tv: sub
* lounge floor lights: standing, flartsy

Available scenes: natural light, relax, bloodbath, rest, disturbia, energize, concentrate, read, warm embrace, galaxy, phthalocyanine green love, starlight, tri colour, shrexy, nightlight, vapor wavey, dimmed, valley dawn, soho

Rules:
- Only include relevant parameters for each action
- Set irrelevant parameters to null
- Temperature range: 153-500
- Brightness range: 1-100"""

class GRPOHue:
    def __init__(self, model_name: str = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit", 
                 max_lora_rank: int = 32,
                 gpu_memory_utilization: float = 0.6,
                 fast_inference: bool = True,
                 max_seq_length: int = 2048,
                 ) -> None:

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=None,
            fast_inference=fast_inference,
            max_lora_rank=max_lora_rank,
            gpu_memory_utilization=gpu_memory_utilization,
        )

        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=max_lora_rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=max_lora_rank,
            lora_dropout=0,
            # bias="None",
            use_gradient_checkpointing="unsloth",
            random_state=3407
        )

    @weave.op()
    def parse_completion_to_action(self, completion: str) -> Union[AGENTACTIONS, None]:
        try:
            completion = completion.strip()
            if completion.startswith("```json"):
                completion = completion.replace("```json", "").replace("```", "").strip()
            
            data = json.loads(completion)
            action_type=data["action_type"]

            brightness_data = data["command"].get("brightness", {})
            brightness=None

            if brightness_data:
                brightness = Brightness(
                    brightness=brightness_data.get("brightness"),
                    relative=brightness_data.get("relative"),
                    up_down=brightness_data.get("up_down")
                )
            
            command = Command(
                zone=data["command"]["zone"],
                light=data["command"]["light"],
                scene=data["command"]["scene"],
                temperature=data["command"]["temperature"],
                brightness=brightness
            )

            action_map = {
                "turn_on": turn_on,
                "turn_off": turn_off,
                "set_scene": set_scene,
                "set_brightness": set_brightness,
                "set_temperature": set_temperature
            }
            
            if action_type in action_map:
                return action_map[action_type](
                    think=data["think"], 
                    command=command
                )
            else:
                return None
            
        except (json.JSONDecodeError, KeyError, ValidationError, TypeError, AttributeError) as e:
            return None
    
    @weave.op()
    def reward_function(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards=[]
        detailed_scores=[]

        scenarios = kwargs.get("scenarios", [])
        scenarios = [Scenario(**json.loads(s)) for s in scenarios]

        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            try:
                if i >=len(scenarios):
                    rewards.append(0.0)
                    continue

                scenario=scenarios[i]

                action = self.parse_completion_to_action(completion)

                if action is None:
                    rewards.append(float(0))
                    detailed_scores.append({
                        "parsing_success": False,
                        "total_reward": 0,
                        "scenario_id": scenario.id if hasattr(scenario, 'id') else i
                    })
                    continue

                scores = score_action(action, scenario)
                component_rewards = {
                    "correct_tool": scores["correct_tool"] * 3.0,
                    "correct_zone": scores["correct_zone"] * 2.0,
                    "correct_scene": scores["correct_scene"] * 1.5,
                    "correct_light": scores["correct_light"] * 1.0,
                    "correct_temperature": scores["correct_temperature"] * 1.0,
                    "correct_brightness": scores["correct_brightness"] * 1.0,
                    "correct_brightness_relative": scores["correct_brightness_relative"] * 0.5,
                    "correct_brightness_up_down": scores["correct_brightness_up_down"] * 0.5,
                    "parsing_bonus": 0.5
                }

                total_reward = sum(component_rewards.values())
                rewards.append(total_reward)

                detailed_score = {
                    "parsing_success": True,
                    "total_reward": total_reward,
                    "component_rewards": component_rewards,
                    "raw_scores": scores,
                    "scenario_id": scenario.id if hasattr(scenario, 'id') else i,
                    "action_type": action.action_type,
                    "predicted_zone": action.command.zone,
                    "expected_zone": scenario.zone,
                    "expected_action": scenario.action_type
                }
                detailed_scores.append(detailed_score)
    
            except Exception as e:
                rewards.append(0.0)

        return rewards
    
    @weave.op()
    def prepare_dataset(self, scenarios: List[Scenario]) -> Dataset:
        
        data=[]
        for scenario in scenarios:
            messages = [
                {"role":"system","content":SYS_PROMPT},
                {"role":"user", "content":scenario.full_command}
            ]
            formatted_prompt=self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            data.append({
                "prompt":formatted_prompt,
                "scenarios":scenario.model_dump_json()
            })
        dataset=Dataset.from_list(data)
        
        return dataset
    
    @weave.op()
    def setup_training_config(self, output_dir: str, max_steps: int) -> GRPOConfig:
        """Create training configuration - this will be traced"""
        training_config = {
            "output_dir": output_dir,
            "learning_rate": 5e-6,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "weight_decay": 0.1,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine",
            "optim": "paged_adamw_8bit",


            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            
            
            "num_generations": 4,
            "max_prompt_length": 1024,
            "max_completion_length": 1024,


            "num_train_epochs": 1,
            "max_steps": max_steps,
            "logging_steps": 5,
            "save_steps": 250,


            "use_vllm": True,
            "vllm_mode": "colocate",
            "vllm_gpu_memory_utilization": 0.7,

            # "loss_type": "dapo",
            # "loss_type": "grpo",
            # "loss_type": "dnpo",
            "loss_type": "dr_grpo",

            "beta": 0.01,
            "epsilon": 0.2,
            "epsilon_high": 0.28,
            "mask_truncated_completions": True,
            "scale_rewards": False,

            "remove_unused_columns": False,
            "dataloader_num_workers": 0,
            "gradient_checkpointing": True,
            "report_to": "wandb",
            "run_name": "grpo_training",
            "max_grad_norm": 0.5
        }
        return GRPOConfig(**training_config)
    
    def train(self,
              dataset_name: str = "mbary/hue_commands_synth_5k_v3",
              split: str = "train",
              limit: Optional[int] = 1000,
              output_dir: str = "grpo_hue_agent",
              max_steps: int = 500,
              ) -> GRPOTrainer:

        scenarios = load_scenarios(
            dataset_name=dataset_name,
            split=split,
            limit=limit,
            seed=42
        )

        dataset = self.prepare_dataset(scenarios)
        training_args = self.setup_training_config(output_dir, max_steps) 

        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=dataset,
            reward_funcs=self.reward_function,  
            args=training_args)

        trainer.train()
        trainer.save_model(output_dir)

        return trainer
    
    def save_model_prod(self, output_dir: str) -> None:
        """Save model for production - this will be traced"""
        self.model.save_pretrained_merged(
            f"{output_dir}_merged", 
            self.tokenizer, 
            save_method="merged_16bit"
        )        

@weave.op()
def create_agent(model_name: str) -> GRPOHue:
    """Create and return agent - this will be traced"""
    return GRPOHue(
        model_name=model_name,
        max_seq_length=2048,
        max_lora_rank=16
    )

def run_training_pipeline(agent: GRPOHue) -> GRPOTrainer:
    """Run the complete training pipeline - this will be traced"""
    trainer = agent.train(
        dataset_name="mbary/hue_commands_synth_5k_v3",
        split="train",
        limit=1000,
        output_dir="grpo_hue_agent",
        max_steps=500,
    )
    
    agent.save_model_prod("grpo_hue_agent_v1")
    return trainer

def main():
    """Main function that will be called within wandb context"""
    model_name = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit"
    

    agent = create_agent(model_name)  
    trainer = run_training_pipeline(agent)  
    
    return agent, trainer

if __name__ == "__main__":
    weave.init(project_name)

    with wandb.init(entity=entity, project=project, name="grpo_hue_agent5") as run:
        agent, trainer = main()  