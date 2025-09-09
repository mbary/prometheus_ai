import unsloth
import os, json, sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import weave
import wandb
from datasets import Dataset
from pydantic import ValidationError
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

sys.path.append(str(Path(__file__).parent.parent))
from utils.utils_types import load_scenarios, Scenario
from utils.agent_tools import turn_off, turn_on, set_brightness, set_scene, set_temperature
from utils.project_types import Command, Brightness
from reward_funcs import create_grpo_reward_functions

AGENTACTIONS = Union[turn_on, turn_off, set_scene, set_brightness, set_temperature]

PROJECT_FULL = "mbaryp2-mbary/grpo_training7"
entity, project = PROJECT_FULL.split("/")

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"grpo_split_rewards_{RUN_ID}"
RUN_ROOT = Path("runs") / "grpo_hue_agent" / RUN_ID
CKPT_DIR   = RUN_ROOT / "ckpt"
LORA_DIR   = RUN_ROOT / "lora"
MERGED_DIR = RUN_ROOT / "merged"
LOG_DIR    = RUN_ROOT / "logs"
CONF_DIR   = RUN_ROOT / "configs"
for p in (CKPT_DIR, LORA_DIR, MERGED_DIR, LOG_DIR, CONF_DIR):
    p.mkdir(parents=True, exist_ok=True)
os.environ["WANDB_DIR"] = str(RUN_ROOT)

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
                 max_seq_length: int = 2048) -> None:

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
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
            lora_alpha=max_lora_rank,
            lora_dropout=0,
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

    @weave.op()
    def parse_completion_to_action(self, completion: str) -> Union[AGENTACTIONS, None]:
        try:
            completion = completion.strip()
            if completion.startswith("```json"):
                completion = completion.replace("```json","").replace("```","").strip()
            data = json.loads(completion)
            action_type = data["action_type"]

            brightness_data = data["command"].get("brightness", {})
            brightness = None
            if brightness_data:
                brightness = Brightness(
                    brightness=brightness_data.get("brightness"),
                    relative=brightness_data.get("relative"),
                    up_down=brightness_data.get("up_down"),
                )
            command = Command(
                zone=data["command"]["zone"],
                light=data["command"]["light"],
                scene=data["command"]["scene"],
                temperature=data["command"]["temperature"],
                brightness=brightness,
            )
            action_map = {
                "turn_on": turn_on, "turn_off": turn_off, "set_scene": set_scene,
                "set_brightness": set_brightness, "set_temperature": set_temperature,
            }
            return action_map[action_type](think=data["think"], command=command) if action_type in action_map else None
        except (json.JSONDecodeError, KeyError, ValidationError, TypeError, AttributeError):
            return None

    @weave.op()
    def prepare_dataset(self, scenarios: List[Scenario]) -> Dataset:
        data = []
        for sc in scenarios:
            messages = [{"role":"system","content":SYS_PROMPT},
                        {"role":"user","content":sc.full_command}]
            formatted = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            data.append({"prompt": formatted, "scenarios": sc.model_dump_json()})
        return Dataset.from_list(data)

    @weave.op()
    def setup_training_config(self, max_steps: int) -> GRPOConfig:
        training_config = {
            "output_dir": str(CKPT_DIR),  
            "learning_rate": 5e-6,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "weight_decay": 0.1,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "cosine",
            "optim": "paged_adamw_8bit",

            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,

            "num_generations": 2,
            "max_prompt_length": 1024,
            "max_completion_length": 158,

            "num_train_epochs": 1,
            "max_steps": max_steps,
            "logging_steps": 5,
            "save_steps": 250,

            "use_vllm": True,
            "vllm_mode": "colocate",
            "vllm_gpu_memory_utilization": 0.7,

            "loss_type": "dr_grpo",
            "beta": 0.01,
            "epsilon": 0.2,
            "epsilon_high": 0.28,
            "delta": None,
            "mask_truncated_completions": True,
            "scale_rewards": False,

            "remove_unused_columns": False,
            "dataloader_num_workers": 0,
            "gradient_checkpointing": True,
            "report_to": "wandb",
            "max_grad_norm": 0.5,
        }
        (CONF_DIR / "train_args.json").write_text(json.dumps(training_config, indent=2))
        (CONF_DIR / "sys_prompt.txt").write_text(SYS_PROMPT)
        return GRPOConfig(**training_config)

    @weave.op()
    def train(self,
              dataset_name: str = "mbary/hue_commands_synth_5k_v3",
              split: str = "train",
              limit: Optional[int] = 3000,
              output_dir: str = str(CKPT_DIR),
              max_steps: int = 300,
              test_rewards_first: bool = True) -> GRPOTrainer:

        scenarios = load_scenarios(dataset_name=dataset_name, split=split, limit=limit, seed=42)

        reward_functions = create_grpo_reward_functions(self)
        if test_rewards_first:
            print("\n" + "="*60)
            print("Testing reward functions before training...")
            print("="*60)
            if not self.test_reward_functions(scenarios[:50], reward_functions):
                print("\nReward functions failed testing. Fix them before training!")
                raise ValueError("Reward functions not suitable for training")
            print("\n✅ Reward functions passed testing. Proceeding with training...")

        dataset = self.prepare_dataset(scenarios)
        training_args = self.setup_training_config(max_steps)

        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            train_dataset=dataset,
            reward_funcs=reward_functions,
            args=training_args,
            gen_kwargs={
                "temperature": 1.2,
                "top_p": 0.95,
                "do_sample": True,
                "max_new_tokens": training_args.max_completion_length,
            },
        )

        trainer.train()
        trainer.save_model(output_dir)

        self.model.save_pretrained(str(LORA_DIR))
        self.tokenizer.save_pretrained(str(LORA_DIR))

        self.model.save_pretrained_merged(str(MERGED_DIR), self.tokenizer, save_method="merged_16bit")
        
        return trainer

    def test_reward_functions(self, scenarios: List[Scenario], reward_functions: List) -> bool:
        test_completions = []
        for _ in range(10): test_completions.append("not valid json at all")
        for _ in range(10): test_completions.append(json.dumps({"wrong": "structure"}))
        for _ in range(15):
            test_completions.append(json.dumps({
                "think":"Processing","action_type":"wrong_action",
                "command":{"zone":"wrong_zone","light":None,"scene":None,"temperature":None,"brightness":None}
            }))
        for i in range(15):
            s = scenarios[i % len(scenarios)]
            test_completions.append(json.dumps({
                "think":"Executing command correctly","action_type":s.action_type,
                "command":{"zone":s.zone,"light":s.light,"scene":s.scene,
                           "temperature":s.temperature,
                           "brightness": s.brightness.dict() if s.brightness else None}
            }))
        prompts = ["test"] * len(test_completions)
        test_scenarios = [s.model_dump_json() for s in scenarios[:len(test_completions)]]
        all_rewards = []
        for i, rf in enumerate(reward_functions):
            rewards = rf(prompts, test_completions, scenarios=test_scenarios)
            all_rewards.append(rewards)
            arr = np.array(rewards)
            print(f"\n{getattr(rf,'__name__',f'Reward_{i}')}:")
            print(f"  Mean: {arr.mean():.3f}, Std: {arr.std():.3f}")
            print(f"  Range: [{arr.min():.3f}, {arr.max():.3f}]")
        combined = np.sum(all_rewards, axis=0)
        print(f"\nCombined rewards - Mean: {combined.mean():.3f}, Std: {combined.std():.3f}")
        return combined.std() > 0.5 and np.any(combined < 0) and np.any(combined > 0)

@weave.op()
def create_agent(model_name: str) -> GRPOHue:
    return GRPOHue(model_name=model_name, max_seq_length=2048, max_lora_rank=32)

@weave.op()
def run_training(model_name: str = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
                 dataset_name: str ="mbary/hue_commands_synth_5k_v3",
                 limit: int = 3000,
                 max_steps: int = 300):
    agent = create_agent(model_name)
    agent.train(dataset_name=dataset_name, split="train", limit=limit,
                output_dir=str(CKPT_DIR), max_steps=max_steps)
    
    (RUN_ROOT / "manifest.json").write_text(json.dumps({
            "run_id": RUN_ID, "run_name": RUN_NAME, "project": PROJECT_FULL,
            "base_model": model_name, "dataset": dataset_name, "limit": limit,
            "max_steps": max_steps, "use_vllm": True,
            "dirs": {"root": str(RUN_ROOT), "ckpt": str(CKPT_DIR),
                     "lora": str(LORA_DIR), "merged": str(MERGED_DIR),
                     "logs": str(LOG_DIR), "configs": str(CONF_DIR)},
        }, indent=2))

    return "done"

if __name__ == "__main__":
    weave.init(PROJECT_FULL)
    with wandb.init(entity=entity, project=project, name=RUN_NAME, dir=str(RUN_ROOT)):
        run_training()
    try: 
        weave.finish()
    except Exception: 
        pass
