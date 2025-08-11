# Prometheus AI

**Prometheus AI** is an intelligent voice-controlled (to be implemented) smart home assistant specifically designed to control Philips Hue lighting ecosystems. Built on top of the [Prometheus](https://github.com/mbary/Prometheus) framework, this agent uses language models to parse natural language commands and execute precise lighting control operations.

If Prometheus stole the flame from the Gods, Prometheus AI is the chariot carrying the light into your home.<br>
With the capability to understand and process complex commands, you too can experience the divine illumination of your living space.


The whole point of this project is to identify a model small enough that I can host locally and is good enough at parsing commands.<br>
I am looking for an excuse to do some reinforcement learning with it improving model performance. Unfortunately, it seems like even the smaller, 0.6B models are quite good at these sort of tasks...<br>
So I might RL it so that it understands commands in Polish language.

## Features

### Core Functionality
- **Natural Language Processing**: Understands complex lighting commands in natural language
- **Zone-Based Control**: Organize lights into zones for granular control
- **Scene Management**: Set predefined lighting scenes across zones
- **Voice Command Ready**: Designed for integration with wake word detection ("Hey Bridgette")

### Smart Actions
- **Turn On/Off**: Control individual lights, zones, or all lights
- **Set Brightness**: Absolute (50%) or relative ("increase by 20%") brightness control
- **Set Temperature**: Adjust color temperature (153K-500K) for ambiance
- **Scene Control**: Apply predefined scenes to zones
- **Individual Light Control**: Target specific lights within zones

## Planned Features

**Voice Transcription**:<br>
I am planning on implementing real-time voice transcription with [Faster Whisper](https://github.com/speaches-ai/speaches) by speaches-ai.<br>
VLLM unfortunately does not support neither streaming transcription nor timestamps (else I could just stich the results together using some overlapping window), hence why I must rely on third-party implementations.<br>
Implementing this will give a real assistant-like experience with (hopefully) minimal latency.


## Architecture

### Core Components

#### Agent (`prometheus_ai.py`)
The main agent class:
- Processes natural language commands
- Selects appropriate actions using structured LLM outputs
- Manages state and dependencies
- Executes lighting commands via the Prometheus framework

#### State Management
- **StateManager**: Tracks current state of all Hue resources
- **DependenciesManager**: Manages LLM clients, bridge connections, and configurations

#### Action Tools (`utils/agent_tools.py`)
Specialized tools for lighting control:
- `turn_on` / `turn_off`: Basic on/off control
- `set_scene`: Apply lighting scenes
- `set_brightness`: Brightness adjustment with relative/absolute modes
- `set_temperature`: Color temperature control

#### Benchmarking System
Comprehensive evaluation framework:
- Synthetic dataset with 5,000+ commands
- Multi-model performance comparison
- Detailed scoring metrics
- Configurable test scenarios

## Benchmarking

Prometheus AI includes a comprehensive benchmarking system to evaluate model performance.<br>
More details on the benchmarking itself can be found in the [benchmarking directory](./prometheus_ai/benchmarking).

### Evaluation Metrics
- **Tool Selection Accuracy**: Correct action chosen
- **Zone Recognition**: Proper zone identification
- **Parameter Extraction**: Accurate brightness, temperature, scene values
- **Error Rate**: Handling of malformed or impossible commands

## Dataset

The project uses a synthetic dataset of almost 5,000 natural language lighting commands:
- **Source**: [mbary/hue_commands_synth_5k_v3](https://huggingface.co/datasets/mbary/hue_commands_synth_5k_v3)
- **Structure**: Wake word phrase + action + parameters
- **Actions**: All supported lighting operations
- **Validation**: Human-reviewed for accuracy

### Example Commands
```
"Hey Bridgette, turn on the office lights"
"Bridgette, set the lounge to relax scene and dim to 40%"
"Hey Bridgette, increase bedroom brightness by 15%"
"Bridgette, set temperature to 250K in all zones"
```

The dataset is split into train/test (4000/981) and will be used to finetune a model for better performance on the task.<br>
The finetuning will be done with the help of either the [Verifiers](https://github.com/willccbb/verifiers/) or [ART](https://github.com/OpenPipe/ART) frameworks for a comprehensive GRPO training pipeline.<br>
It is likely that to do so, I might have to re-write the package ever so slightly, to make it compatible with the frameworks.<br>
I am yet to decide which model it is going to be, but it will liklely be one of the smaller Qwen models, allowing me to potrntialyl run the pipeline on my PC.<br>

### Code Structure
```
prometheus_ai/
├── prometheus_ai.py          # Main agent implementation
├── benchmarking/             # Performance evaluation
│   ├── benchmarking.py       # Core benchmarking logic
│   ├── multi_benchmark.py    # Multi-model comparison
│   └── configs/              # Test configurations
├── utils/                    # Utilities and types
│   ├── agent_tools.py        # Action implementations
│   └── project_types.py      # Data models
└── data/                     # Dataset generation
    └── generate_data.py      # Synthetic data creation
```