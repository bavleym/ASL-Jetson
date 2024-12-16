import onnx
model = onnx.load(r"C:\Users\Gaiaf\OneDrive - csulb\Documents\CECS 490 B\best.onnx")
print("Model Inputs:")
for input in model.graph.input:
    print(input.name)
