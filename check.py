import onnxruntime as ort

# Check available providers
providers = ort.get_available_providers()
print("Available ONNX Runtime providers:")
for provider in providers:
    print(f"  - {provider}")

if 'DmlExecutionProvider' in providers:
    print("✓ ONNX Runtime with DirectML is available!")
    
    # Test with a simple session
    import numpy as np
    from onnxruntime import InferenceSession, SessionOptions
    
    # Create a simple model (identity function)
    import onnx
    from onnx import helper, TensorProto
    
    # Create a simple identity model
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1])
    node = helper.make_node('Identity', ['X'], ['Y'])
    graph = helper.make_graph([node], 'identity', [X], [Y])
    model = helper.make_model(graph)
    
    # Run with DirectML
    session = InferenceSession(model.SerializeToString(), providers=['DmlExecutionProvider'])
    result = session.run(['Y'], {'X': np.array([1.0], dtype=np.float32)})
    print(f"✓ ONNX DirectML test passed: {result[0]}")
    
else:
    print("✗ DirectML not available in ONNX Runtime")