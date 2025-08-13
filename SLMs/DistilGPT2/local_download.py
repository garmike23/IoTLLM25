from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "distilgpt2"
output_path = "./distilgpt2_local"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

tokenizer.save_pretrained(output_path)
model.save_pretrained(output_path)

