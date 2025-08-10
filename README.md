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

Prometheus AI includes a comprehensive benchmarking system to evaluate model performance.

### Evaluation Metrics
- **Tool Selection Accuracy**: Correct action chosen
- **Zone Recognition**: Proper zone identification
- **Parameter Extraction**: Accurate brightness, temperature, scene values
- **Error Rate**: Handling of malformed or impossible commands

### Model Performance Results
Recent benchmarking shows:
- **Qwen3-1.7B**: 99.8% success rate (optimized)
- **Qwen3-0.6B**: 78.4% success rate

*Note: Only local Qwen models have been comprehensively benchmarked. Other providers mentioned in configuration examples have not been tested yet.*

## Dataset

The project uses a synthetic dataset of 5,000+ natural language lighting commands:
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

## Research & Development

### Model Optimization
- **Parameter Tuning**: Optimized temperature, top_p, presence_penalty
- **Prompt Engineering**: Structured system prompts for consistent outputs
- **Mode Selection**: JSON vs Tools mode based on provider capabilities

### Future Enhancements
- **Speech Recognition**: Whisper integration for voice commands
- **Wake Word Detection**: Continuous listening for "Hey Bridgette"
- **Multi-Language Support**: Commands in multiple languages
- **Advanced Scenes**: Dynamic scene creation and modification

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