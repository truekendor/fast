"""Convert the existing MobileNet and piece classifier weights to an embedded ONNX model."""

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _read_model(directory: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Read the TensorFlow.js manifest and decode its little-endian weight shards."""
    model = json.loads((directory / "model.json").read_bytes())
    weights = {}
    for group in model["weightsManifest"]:
        raw = b"".join((directory / name).read_bytes() for name in group["paths"])
        offset = 0
        for spec in group["weights"]:
            if "quantization" in spec:
                raise ValueError(f"unexpected quantized weight {spec['name']}")
            dtype = {"float32": "<f4", "int32": "<i4"}[spec["dtype"]]
            count = int(np.prod(spec["shape"]))
            array = np.frombuffer(raw, dtype=dtype, count=count, offset=offset)
            offset += array.nbytes
            weights[spec["name"]] = array.reshape(spec["shape"])
        if offset != len(raw):
            raise ValueError(f"unexpected trailing data in {directory}")
    return model, weights


def _text(value: dict) -> str:
    """Decode a TensorFlow graph string attribute."""
    return base64.b64decode(value["s"]).decode("ascii")


def _integers(value: dict) -> list[int]:
    """Decode an array of TensorFlow graph integers."""
    return [int(index) for index in value["list"]["i"]]


class _Export:
    """Build the equivalent ONNX graph with a channels-first convolution path."""

    def __init__(self) -> None:
        """Record nodes and initializers in graph order."""
        self.nodes = []
        self.initializers = {}

    def weight(self, name: str, value: np.ndarray) -> str:
        """Write a constant tensor once and return its graph name."""
        if name not in self.initializers:
            self.initializers[name] = numpy_helper.from_array(value, name)
        return name

    def op(self, kind: str, inputs: list[str], output: str, **attrs) -> str:
        """Append a node with a single output to the ONNX graph."""
        self.nodes.append(helper.make_node(kind, inputs, [output], **attrs))
        return output

    def backbone(self, model: dict, weights: dict[str, np.ndarray]) -> None:
        """Translate the exact MobileNet graph, splitting its fused convolutions."""
        for node in model["modelTopology"]["node"]:
            name, kind = node["name"], node["op"]
            inputs = [name.removesuffix(":0") for name in node.get("input", [])]
            attrs = node.get("attr", {})
            if kind == "Const":
                continue
            if kind == "Placeholder" and name == "inputs":
                self.op("Transpose", ["rgb"], name, perm=[0, 3, 1, 2])
            elif kind in ("_FusedConv2D", "FusedDepthwiseConv2dNative"):
                self.convolution(name, inputs, attrs, weights, depthwise=kind.startswith("Fused"))
            elif kind in ("AddV2", "Mul", "Sub"):
                for argument in inputs:
                    if argument in weights:
                        if weights[argument].ndim != 0:
                            raise ValueError(f"unsupported non-scalar broadcast: {argument}")
                        self.weight(argument, weights[argument])
                self.op("Add" if kind == "AddV2" else kind, inputs, name)
            elif kind == "Relu6":
                zero = self.weight("clip_zero", np.array(0, np.float32))
                six = self.weight("clip_six", np.array(6, np.float32))
                self.op("Clip", [*inputs, zero, six], name)
            elif kind == "Mean" and list(weights[inputs[1]]) == [1, 2]:
                self.op("ReduceMean", inputs[:1], name, axes=[2, 3], keepdims=int(attrs["keep_dims"]["b"]))
            elif kind == "Squeeze" and _integers(attrs["squeeze_dims"]) == [1, 2]:
                axes = self.weight("spatial_axes", np.array([2, 3], np.int64))
                self.op("Squeeze", [*inputs, axes], name)
            elif kind == "AvgPool" and _text(attrs["data_format"]) == "NHWC" and _text(attrs["padding"]) == "VALID":
                self.op("AveragePool", inputs, name, kernel_shape=_integers(attrs["ksize"])[1:3],
                        strides=_integers(attrs["strides"])[1:3])
            elif kind == "Identity" and name == "Identity":
                self.op("Identity", inputs, "features")
            else:
                raise ValueError(f"unsupported node {name}: {kind}")

    def convolution(self, name: str, inputs: list[str], attrs: dict,
                    weights: dict[str, np.ndarray], *, depthwise: bool) -> None:
        """Convert a TensorFlow weight layout and preserve the fused bias and activation."""
        operations = [base64.b64decode(value) for value in attrs["fused_ops"]["list"]["s"]]
        if operations not in ([b"BiasAdd"], [b"BiasAdd", b"Relu"]) or len(inputs) != 3:
            raise ValueError(f"unsupported fused operations: {operations}")
        if _text(attrs["data_format"]) != "NHWC" or _text(attrs["padding"]) != "SAME":
            raise ValueError(f"unsupported convolution layout or padding: {name}")
        filt = weights[inputs[1]]
        if depthwise:
            height, width, groups, multiplier = filt.shape
            filt = filt.transpose(2, 3, 0, 1).reshape(groups * multiplier, 1, height, width)
        else:
            groups = 1
            filt = filt.transpose(3, 2, 0, 1)
        weights_name = self.weight(inputs[1], filt)
        bias = self.weight(inputs[2], weights[inputs[2]])
        fused_relu = len(operations) == 2
        result = self.op("Conv", [inputs[0], weights_name, bias], name + "/conv" if fused_relu else name,
                         auto_pad="SAME_UPPER", strides=_integers(attrs["strides"])[1:3],
                         dilations=_integers(attrs["dilations"])[1:3], group=groups)
        if fused_relu:
            self.op("Relu", [result], name)

    def head(self, model: dict, weights: dict[str, np.ndarray]) -> None:
        """Append the existing three Dense layers using their original weights and biases."""
        incoming = "features"
        for layer in model["modelTopology"]["config"]["layers"]:
            config = layer["config"]
            name = config["name"]
            if layer["class_name"] != "Dense" or not config["use_bias"]:
                raise ValueError(f"unexpected head layer: {name}")
            kernel, bias = (self.weight(f"{name}/{key}", weights[f"{name}/{key}"])
                            for key in ("kernel", "bias"))
            product = self.op("MatMul", [incoming, kernel], f"{name}/product")
            offset = self.op("Add", [product, bias], f"{name}/offset")
            if config["activation"] == "relu":
                incoming = self.op("Relu", [offset], name)
            elif config["activation"] == "softmax":
                incoming = self.op("Softmax", [offset], "probabilities", axis=1)
            else:
                raise ValueError(f"unknown head activation: {config['activation']}")
        if incoming != "probabilities":
            raise ValueError("missing classifier output")

    def save(self, path: Path) -> None:
        """Validate and write a self-contained model with both diagnostic outputs."""
        values = helper.make_tensor_value_info
        graph = helper.make_graph(
            self.nodes,
            "Fast chess piece classifier",
            [values("rgb", TensorProto.FLOAT, ["batch", 224, 224, 3])],
            [values("features", TensorProto.FLOAT, ["batch", 1024]),
             values("probabilities", TensorProto.FLOAT, ["batch", 13])],
            list(self.initializers.values()),
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
        onnx.checker.check_model(model, full_check=True)
        onnx.save_model(model, path)


def main() -> None:
    """Convert the two existing web model directories without training or changing weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path, help="Directory containing mobilenet and classification_model")
    parser.add_argument("output", type=Path, help="ONNX model to write")
    args = parser.parse_args()
    exporter = _Export()
    exporter.backbone(*_read_model(args.models / "mobilenet"))
    exporter.head(*_read_model(args.models / "classification_model"))
    exporter.save(args.output)


if __name__ == "__main__":
    main()
