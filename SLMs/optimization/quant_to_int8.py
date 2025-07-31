from onnxruntime.quantization import quantize_dynamic, QuantType

# Paths: change these as needed
input_model = "distilgpt2_carqa.onnx"
output_model = "distilgpt2_carqa_int8.onnx"

# Quantize dynamically (INT8 weights)
quantize_dynamic(
    model_input=input_model,
    model_output=output_model,
    weight_type=QuantType.QInt8  # Use QInt8 for INT8 quantization
)

print(f"Quantized INT8 model saved as {output_model}")

