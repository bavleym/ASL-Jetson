import onnx
model = onnx.load(r"best.onnx")
print("Model Inputs:")
for input in model.graph.input:
    print(input.name)
