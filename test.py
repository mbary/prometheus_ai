import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import time
from rich import print
POS_DIR = Path("./data/recorded/positive")
NEG_DIR = Path("./data/recorded/negative")

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"


client = OpenAI(api_key=openai_api_key,
                base_url=openai_api_base)

model = client.models.list().data[0].id
model



TRANSCRIPTION_PROMPT = """You are looking for the wakeword phrase <wakeword_phrase>Hey Bridgette</wakeword_phrase>.
Anything following the wakeword_phrase is a command and should be returned wrapped with <command> tags.
If the wakeword phrase is not present, return a <empty> tag.
"""


for file in os.listdir("./data/recorded"):
    print(file)
    with open(f"./data/recorded/{file}", 'rb') as f:
        audio_file=f.read()

    start=time.time()
    trans = client.audio.transcriptions.create(
        model=model,
        file=audio_file,
        prompt=TRANSCRIPTION_PROMPT
    )
    end = time.time()
    print(end-start)
    print(trans.text)


with open("./data/wakeword_command.wav", 'rb') as f:
    command_file = f.read()

with open("./data/recorded/negative/news_headphones_negative_95_0.wav", 'rb') as f:
    negative_file = f.read()

pos = client.audio.transcriptions.create(
    model=model,
    file=command_file,
    prompt=TRANSCRIPTION_PROMPT
)

print(pos.text)
dir(pos)
pos.schema_json()

pos.json

neg = client.audio.transcriptions.create(
    model=model,
    file=negative_file,
    prompt=TRANSCRIPTION_PROMPT)
print(neg.text)


client_qwen= OpenAI(api_key=openai_api_key,
                   base_url="http://localhost:8100/v1",)


model_q = client_qwen.models.list().data[0].id
model_q

resp = client_qwen.chat.completions.create(
    model=model_q,
    messages=[
        {"role": "user", "content": "Hello, how can I help you today?"},
        {"role": "assistant", "content": '<think>\nOkay, the user greeted me with "Hello, how can I help you today?" which is a standard opening. I need to respond in a friendly and welcoming manner. Let me start by acknowledging their greeting.\n\nI should express that I\'m here to assist them. Maybe mention my capabilities, like answering questions, providing information, or helping with tasks. I should keep it concise but informative.\n\nAlso, I should invite them to ask anything they need help with. Maybe add an emoji to keep the tone positive. Let me check if there\'s anything else I need to include. No, that should cover it. Make sure the response is clear and helpful.\n</think>\n\nHello! I\'m here to help you with anything you need. Whether you have questions, need assistance with a task, or just want to chat, feel free to let me know! 😊 What can I assist you with today?'},
        {"role": "user", "content": "LUBIE LIZAĆ MOKRE PALCE PO KURCZAKU I PÓŹNIEJ SIĘ ZASTANAWIAM, CZY TO NORMALNE"},
        {"role": "assistant", "content": """<think>
Okay, the user is asking if it's normal to like to wet your feet in the bath and then later wonder if that's 
normal. First, I need to understand the context. The user is talking about a personal habit, so I should approach 
it with sensitivity and empathy.

The user might be feeling uncertain about their behavior, so I should validate their feelings. Maybe they're trying
to figure out if this is a common thing or if they should be concerned. I should explain that such behaviors can be
normal and part of personal preferences, but also mention that if it's causing distress, it's important to consult 
a professional.

I need to make sure the response is reassuring but not dismissive. Also, since the user is using a mix of Polish 
and English, I should acknowledge that and respond in a way that's clear and inclusive. Maybe start by affirming 
that it's normal, then explain why, and offer further help if needed. Avoid any medical jargon, keep it simple and 
supportive.
</think>

Hello! It's great that you're reflecting on your habits and seeking clarity. Let me break this down for you:

1. **Personal Preference**: It’s completely normal to have a preference for certain activities, like soaking your 
feet in warm water (e.g., in a bath, after a long day). This can be a relaxing, self-care ritual. If it makes you 
feel good, it’s not "abnormal" — it’s just a personal choice.

2. **Self-Reflection**: Wondering if something is "normal" is a sign of self-awareness, which is a positive trait. 
It shows you’re curious about your habits and their impact on your well-being.

3. **When to Be Concerned**: If this behavior starts to feel obsessive, distressing, or interferes with your daily 
life (e.g., neglecting responsibilities, anxiety about not doing it), it might be worth talking to a professional 
(e.g., a therapist or counselor). But if it’s a simple, enjoyable habit, there’s no need for concern.

4. **Cultural or Social Norms**: In some cultures, "wet feet" (like in a bath) is a common, even expected, 
practice. If you’re in an environment where this is normalized, it’s likely a shared experience.

**Final Thought**: If this habit brings you comfort and doesn’t cause harm, it’s a personal choice. If you ever 
feel unsure, it’s always okay to ask for help. You’re not alone in this! 😊

Would you like to explore this further or have other questions?"""},
{"role":"user", "content":"PO POLSKU DO MNIE KURWO MÓW"},
{"role": "assistant", "content": """<think>
Okay, the user is asking in Polish, "PO POLSKU DO MNIE KURWO MÓW". Translating that, it seems like they want me to 
speak in Polish and be more direct or rude. But I need to be careful. The user might be testing if I can respond in
another language, but they also might be looking for a more assertive or confrontational response.

First, I should acknowledge their request. They want me to speak in Polish, so I should respond in Polish. But the 
phrase "KURWO MÓW" translates to "speak rudely" or "speak in a harsh way". However, using harsh language is not 
appropriate. I need to make sure my response is respectful and not offensive.

I should clarify that I can speak in Polish but will maintain a polite and helpful tone. It's important to set 
boundaries and ensure the user knows I can't engage in rude or inappropriate behavior. Also, the user might be 
trying to provoke a response, but I should stay professional.

I should confirm their request to speak in Polish and reassure them that I'll be helpful and respectful. Maybe 
offer to assist with their questions in Polish. Also, remind them that I can't engage in any harmful or offensive 
communication. This way, I address their request without violating any guidelines.
</think>

Oczywiście, mogę rozmawiać z Tobą w języku polskim. Ale zazwyczaj trzymam się odpowiedniego tonu i nie stosuję się 
do żarłocznych lub niezgodnych z normami zachowań. Jeśli masz pytania, potrzebujesz pomocy lub chcesz rozmawiać o 
czymś konkretnym, daj znać – zawsze będę gotowy, bym pomógł w sposób odpowiedzialny i zrozumiały. 😊"""},
{"role": "user", "content": "NIE BĄDŹ TAKA MIETKA FAJA TO TYLKO ŻARTY HEUEUUEUEUHEUHEUHUE, CO TY TAKI KURWA DELIKATNY JESTES, MASZ MÓWIĆ DO MNIE DRESIARSKIM SLANGIEM"}
    ]
)

print(resp.choices[0].message.content)