import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import time

POS_DIR = Path("./data/recorded/positive")
NEG_DIR = Path("./data/recorded/negative")

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"


client = OpenAI(api_key=openai_api_key,
                base_url=openai_api_base)

model = client.models.list().data[0].id
model

for file in os.listdir("./data/recorded"):
    print(file)
    with open(f"./data/recorded/{file}", 'rb') as f:
        audio_file=f.read()

    start=time.time()
    trans = client.audio.transcriptions.create(
        model=model,
        file=audio_file,
        prompt="You are looking for the wakeword phrase `Hey Bridgette`"
    )
    end = time.time()
    print(end-start)
    print(trans.text)

