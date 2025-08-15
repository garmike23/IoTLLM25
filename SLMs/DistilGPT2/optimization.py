import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from onnxruntime.quantization import quantize_dynamic, QuantType

def expand(path):
    return os.path.expanduser(os.path.expandvars(path))

model_dir = expand("./distilgpt2_output")
onnx_fp32_path = expand("./distilgpt2_output.onnx")
onnx_int8_path = expand("./distilgpt2_output_int8.onnx")

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
model.eval()
model.config.use_cache = False  # no caching

# === Wrapper to avoid past_key_values issue ===
class GPT2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

wrapper = GPT2Wrapper(model)

enc = tokenizer(
    "Scenario: It is raining heavily\nQuestion: How should I proceed?\nAnswer:",
    return_tensors="pt"
)

# === Export ===
print("Exporting model to ONNX...")
torch.onnx.export(
    wrapper,  # use wrapper here
    (enc["input_ids"], enc["attention_mask"]),
    onnx_fp32_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size", 1: "sequence_length"}
    },
    opset_version=17
)

# === Quantize ===
print("Quantizing model to INT8...")
quantize_dynamic(
    model_input=onnx_fp32_path,
    model_output=onnx_int8_path,
    weight_type=QuantType.QInt8
)
print("Done!")
