import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==== Your model path (folder with config.json, tokenizer, pytorch_model.bin, etc) ====
model_path = "./distilgpt2_output"  # update if needed
onnx_output_path = "distilgpt2.onnx"

# ==== Load model and tokenizer ====
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.eval()

# ==== Prepare a dummy input matching your format ====
example_prompt = (
    "Scenario: It is raining heavily and I am going 60 miles per hour\n"
    "Question: How should I proceed?\n"
    "Answer:"
)
dummy = tokenizer(example_prompt, return_tensors="pt")
input_ids = dummy["input_ids"]

# ==== Export to ONNX (no cache/past_key_values, just prompt input) ====
torch.onnx.export(
    model,
    input_ids,
    onnx_output_path,
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence"},
        "logits": {0: "batch_size", 1: "sequence"},
    },
    opset_version=14,
    do_constant_folding=True,
)

print(f"Model exported to {onnx_output_path}")

