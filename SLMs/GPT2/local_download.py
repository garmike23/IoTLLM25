from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "openai-community/gpt2"
output_path = "./gpt2_local"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

tokenizer.save_pretrained(output_path)
model.save_pretrained(output_path)

