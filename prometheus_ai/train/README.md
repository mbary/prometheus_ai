# Model Fine-Tuning


As shown in the [benchmarking results](../benchmarking/README.md) the smaller models were already quite capable when it cames to following instructions.<br>
However, they would still often struggle with correctly extracting command information from the user input.<br>
This is why I have decided to perform small fine-tuning experiments on the smaller models to see if I can improve their performance in argument parsing.

#### NOTE
The initial benchmarking was implemented using pydantic structured outputs and instructor as the client patcher.<br>
As a result, the implemented agent was utilising tool-calls under the hood, without my explicit knowledge.<br>
Due to some instructor implementation quirks, I was unable to fine-tune a model with the same setup.<br>
I now understand both the limitations and advantages of frameworks such as instructor.<br>

For that reason I have decided to alter the agent design and am no longer relying on instructor/pydantic for structured outputs and am instead generating JSON outputs directly from the model followed by command parsing and tool calls.<br>

As a result, to provide comparable results, I had to adapt the benchmarking setup (/train/benchmarking.py) and re-run the benchmarks for the base models followed by the fine-tuned models.<br>

I will now provide a brief overview of the fine-tuning process and the results I have obtained.


## Changes
As mentioned in the Note above, in order to perform fine-tuning I had to make some changes to my agent design.<br>
Following is a short summary of these changes:
1) Dropped the use of pydantic for structured outputs
2) Implemented direct JSON output generation from the model
3) Implemented custom parsing of the JSON output to extract command arguments
4) Implemented custom tool-calling logic based on the parsed arguments

## Initial Fine-Tuning

So initially I finetuned the models that would esentially output the following structure:

```json
{
    "think": "Your reasoning for this action",
    "action_type": "turn_on|turn_off|set_scene|set_brightness|set_temperature",
    "command": {
        "zone": "office|lounge|lounge floor lights|bedroom|all|tv",
        "light": "light_name or null",
        "scene": "scene_name or null", 
        "temperature": "number_or_null",
        "brightness": {
            "brightness": "number_or_null",
            "relative": bool,
            "up_down": "up|down or null"
        }
    }
}
```
The output would then be parsed and the relevant tool would be called with the extracted arguments.<br>

### Identified Issues
This was quite a learning experience. Throughout the process I identified several issues with my approach/undestanding of the matter:
1. The nested structure, with similar naming convention (brightness:{brightness...}) proved to be too confusing for the tiny models. 
2. The lack of standardised argument values added unnecessary confusion i.e. in some cases, for the brightness param, the argument would be value|null whereas in other cases it would be value|bool, which meant the model got confused which is which, resulting in invalid outputs.
3. The lack of examples for certain commands (i.e. set_brightness) meant the model struggled to understand how to use them.
4. 
However, the nested structure with similar naming convention (brightness:{brightness}) proved to be too confusing for the tiny models.


## Benchmarking
Each benchmarked model was served locally on my machine, with the same parameters:
```shell
vllm serve $UNSLOTH_QWEN25_1_5_INSTRUCT --served-model-name base-qwen15b-bnb4 --quantization bitsandbytes --load-format bitsandbytes --max-model-len 4096 --max-num-batched-tokens 12000 --max-num-seqs 16 --enable-chunked-prefill --gpu-memory-utilization 0.6
```
Enabling up to 16 concurrent sequences, with 4096 token context length (with the system prompt taking up ~720 tokens - depending on the tokenizer), benchmarking 800 samples took up 64 seconds! (varies slightly depending on the model)<br>
![Benchmarking speed](images/benchmark_speed.png)


### Models
#### Base Models
All benchmarked models are the 'instruct' variants of the models.<br>
To showcase the power of finetuning, let's first look at the 'base' models:
- Qwen2.5 1.5B Instruct 
- Qweb2.5 1.5B BNB4 by Unsloth
  
As you can see, both models perform are quite good when it comes to parsing the user command, however, they perform incredibly poorly in outputing the actual JSON structure.<br> 
Meaning that IF they manage to output the correct JSON structure, they are very likely to have the correct arguments.<br>
Neither of the models managed to get above 30% success rate!
##### Qwen2.5 1.5B Instruct
![Qwen2.5 1.5B Instruct](images/base_qwen15b.png)

##### Qwen2.5 1.5B BNB4 by Unsloth
![Qwen2.5 1.5B BNB4 by Unsloth](images/base_bnb4.png)
#### Fine-tuned Model
Due to physical limitations of my machine, I chose to fine-tune the 4bit quantized version by unsloth, the *unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit* model.<br>
Though I am likely to attempt fine-tuning some other models too!<br>

The fine-tuned model performed significantly better than the base models, achieving an amazing 90% success rate!
While it performed slightly worse in parsing the correct arguments, it managed to output the correct JSON on a way more consistent basis.<br> 
![Fine-tuned Qwen2.5 1.5B BNB4 by Unsloth](images/first_tuned_model.png)




### Prompt Engineering



##TODO 
Add info about the prompt engineering and how few-shot increasing teh correct parsing
add info about the dataset used for training (littl einfo then point to the benchmarking readme)
