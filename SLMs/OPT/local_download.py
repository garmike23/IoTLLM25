from transformers import AutoTokenizer, AutoModelForCausalLM

def expand(path):
    return os.path.expanduser(os.path.expandvars(path))

model_id = "facebook/opt-125m"
output_path = expand("./opt_local")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

tokenizer.save_pretrained(output_path)
model.save_pretrained(output_path)

