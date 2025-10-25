from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import torch

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ADAPTER_PATH = "./model"

print("Loading model...")

model = AutoPeftModelForCausalLM.from_pretrained(
    ADAPTER_PATH,
    dtype=torch.float32,
    device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
model.eval()

print("DEBUG_MODEL:")
print("MODEL_KEYS: ", list(model.state_dict().keys())[:5])
print("---DEBUG_MODEL_END---")

class Request(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(req: Request):
    print("API: GENERATING...")
    input_text = req.prompt
    inputs = tokenizer(input_text, return_tensors="pt")

    # If do_spample=False
    # No need to use temperature, top_p
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        repetition_penalty=1.2,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("DEBUG_RESPONSE:")
    print("RESPONSE_FULL: ")
    print(response)
    
    # Remove the eos - end of sequence
    if tokenizer.eos_token in response:
        response = response.split(tokenizer.eos_token)[0]

    # Remove user input if model repeats it
    if response.startswith(input_text):
        response = response[len(input_text):].strip()

    # Remove the redundant generation
    # Currently, the redundant starts after \n\n
    if "\n\n" in response:
        response = response.split("\n\n")[0]

    print("RESPONSE_FINAL: ")
    print(response)
    print("---RESPONSE_DEBUG_END---")

    print("API: COMPLETED!")
    return {"response": response}

# Sanity check
@app.get("/status")
def root():
    return { "message": "FineTunedModelAPI is running!" }

print("\n\n\nYour api is ready! Go to /status to get ready message.")
