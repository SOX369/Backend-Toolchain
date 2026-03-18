import onnx
from onnx import numpy_helper
import json

def onnx_dtype_to_str(dtype):
    return onnx.TensorProto.DataType.Name(dtype)

def extract_onnx_graph_to_json(onnx_path, json_path):
    model = onnx.load(onnx_path)
    graph = model.graph

    result = {
        "model_info": {
            "ir_version": model.ir_version,
            "producer_name": model.producer_name,
            "producer_version": model.producer_version,
            "opset_import": [
                {"domain": x.domain, "version": x.version}
                for x in model.opset_import
            ]
        },
        "inputs": [],
        "outputs": [],
        "initializers": [],
        "nodes": []
    }

    # -------- 输入张量 --------
    for inp in graph.input:
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                shape.append("dynamic")

        result["inputs"].append({
            "name": inp.name,
            "dtype": onnx_dtype_to_str(inp.type.tensor_type.elem_type),
            "shape": shape
        })

    # -------- 输出张量 --------
    for out in graph.output:
        shape = []
        for dim in out.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                shape.append("dynamic")

        result["outputs"].append({
            "name": out.name,
            "dtype": onnx_dtype_to_str(out.type.tensor_type.elem_type),
            "shape": shape
        })

    # -------- 权重 / 常量张量 --------
    for init in graph.initializer:
        tensor = numpy_helper.to_array(init)
        result["initializers"].append({
            "name": init.name,
            "dtype": onnx_dtype_to_str(init.data_type),
            "shape": list(tensor.shape)
        })

    # -------- 计算节点 --------
    for node in graph.node:
        attrs = {}
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.INT:
                attrs[attr.name] = attr.i
            elif attr.type == onnx.AttributeProto.FLOAT:
                attrs[attr.name] = attr.f
            elif attr.type == onnx.AttributeProto.STRING:
                attrs[attr.name] = attr.s.decode("utf-8")
            elif attr.type == onnx.AttributeProto.INTS:
                attrs[attr.name] = list(attr.ints)
            elif attr.type == onnx.AttributeProto.FLOATS:
                attrs[attr.name] = list(attr.floats)
            else:
                attrs[attr.name] = "UNSUPPORTED_ATTR_TYPE"

        result["nodes"].append({
            "name": node.name,
            "op_type": node.op_type,
            "inputs": list(node.input),
            "outputs": list(node.output),
            "attributes": attrs
        })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[OK] ONNX 模型结构已导出到: {json_path}")


if __name__ == "__main__":
    onnx_path = "Resnet640_cifar10_no_Normalize_int0810.onnx"   # 你的 ONNX 模型路径
    json_path = "model_struct.json"
    extract_onnx_graph_to_json(onnx_path, json_path)