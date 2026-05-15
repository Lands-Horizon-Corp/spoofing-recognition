from __future__ import annotations

import glob
import os
from typing import cast

import numpy as np
import onnxruntime
import onnxruntime as ort
import spoofdet.config as config
import torch
from onnxruntime.quantization import CalibrationDataReader
from onnxruntime.quantization import CalibrationMethod
from onnxruntime.quantization import QuantFormat
from onnxruntime.quantization import quantize_static
from onnxruntime.quantization import QuantType
from onnxruntime.quantization import shape_inference
from spoofdet.data_module import SpoofDetDataModule
from spoofdet.models.sota_model import SpoofingDetection

import onnx


class DataLoaderReader(CalibrationDataReader):
    def __init__(self, dataloader, model_path, max_batches=50):
        self.dataloader = dataloader
        self.enum_data = iter(self.dataloader)
        self.max_batches = max_batches
        self.batch_counter = 0
        self.model_path = model_path

        # Get the input name from the ONNX model automatically
        session = onnxruntime.InferenceSession(
            self.model_path, providers=['CPUExecutionProvider'])
        self.input_name = session.get_inputs()[0].name

    def get_next(self):
        if self.batch_counter >= self.max_batches:
            print('Reached max batches, stopping.')
            return None
        try:
            images, _ = next(self.enum_data)
            print(
                f"Batch {self.batch_counter}: original tensor range [{images.min():.3f}, {images.max():.3f}], shape {images.shape}")  # noqa: E501
            input_data = images.cpu().numpy()
            print(
                f"After numpy: range [{input_data.min():.3f}, {input_data.max():.3f}]")
            self.batch_counter += 1
            return {self.input_name: input_data}
        except StopIteration:
            print('DataLoader exhausted.')
            return None


def find_best_checkpoint(checkpoint_dir: str) -> str:
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
    if not checkpoint_files:
        raise FileNotFoundError(
            f"No checkpoint files found in directory: {checkpoint_dir}")
    best_checkpoint = max(checkpoint_files, key=os.path.getctime)
    return best_checkpoint


def check_latest_onnx_file_version():
    onnx_files = glob.glob(f"onnx/{config.MODEL_NAME}_ver*_fp32.onnx")
    if not onnx_files:
        return 0
    versions = [int(os.path.basename(f).split('_ver')[1].split('_')[0])
                for f in onnx_files]
    return max(versions)


def main():
    datamodule = SpoofDetDataModule(
        json_train_path=str(config.TRAIN_JSON),
        json_test_path=config.TEST_JSON,
        root_dir=config.ROOT_DIR,
        bbox_lookup_path=config.BBOX_LOOKUP,
        target_size=config.TARGET_SIZE,
        bbox_original_size=config.BBOX_ORIGINAL_SIZE,
        train_img_count=config.TRAIN_IMG_COUNT,
        val_img_count=config.VAL_IMG_COUNT,
        test_img_count=config.TEST_IMG_COUNT,
        spoof_percent=config.SPOOF_PERCENT,
        batch_size=1,
        num_workers=config.NUM_WORKERS
    )
    datamodule.setup(stage='fit')
    val_dataloader = datamodule.val_dataloader()

    checkpoint_dir = 'checkpoints/'
    best_checkpoint_path = find_best_checkpoint(checkpoint_dir)
    model = SpoofingDetection.load_from_checkpoint(best_checkpoint_path)
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    dummy_input = torch.randn(1, 3, config.TARGET_SIZE,
                              config.TARGET_SIZE, device=device)

    latest_version = check_latest_onnx_file_version()
    fp32_path = f"onnx/{config.MODEL_NAME}_ver{latest_version + 1}_fp32.onnx"
    os.makedirs(os.path.dirname(fp32_path), exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_input,),
        fp32_path,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],

    )

    # input_path = 'onnx/mobile_net_v4_ver15_fp32.onnx'
    clean_path = 'onnx/mobile_net_v4_ver15_fp32_clean.onnx'

    model = onnx.load(fp32_path)
    output_name = model.graph.output[0].name
    print('Output tensor name:', output_name)

    # Find the node that produces this output
    for node in model.graph.node:
        if output_name in node.output:
            print('Output node name:', node.name, 'op_type:', node.op_type)
            _ = node.name
            break

    # 2. Clear existing shape information (value_info)
    # This removes the "256" hint causing the conflict
    del model.graph.value_info[:]

    # 3. Save the "clean" model
    onnx.save(model, clean_path)
    print(f"Sanitized model saved to {clean_path}")

    prepared_path = 'onnx/mobile_net_v4_prepared.onnx'
    shape_inference.quant_pre_process(clean_path, prepared_path)

    reader = DataLoaderReader(val_dataloader, prepared_path, max_batches=50)
    # --- Now run your quantization on the CLEAN model ---

    int8_model_path = 'onnx/mobile_net_v4_ver15_int8.onnx'

    # print("Starting quantization...")
    # quantize_dynamic(
    #     model_input=clean_path,  # Use the clean one!
    #     model_output=int8_model_path,
    #     weight_type=QuantType.QInt8
    # )
    # print(f"Success! Quantized model saved to: {int8_model_path}")

    print('Calibrating and Quantizing...')
    # quantize_static(
    #     model_input=clean_path,
    #     model_output=int8_model_path,
    #     calibration_data_reader=DataLoaderReader(small_val_loader, clean_path),
    #     # Quantize-DeQuantize format (best for x86/ARM)
    #     quant_format=QuantFormat.QDQ,
    #     weight_type=QuantType.QUInt8,  # Usually QInt8 is better for Static, but QUInt8 works too
    #     activation_type=QuantType.QUInt8
    # )

    # all_node_names = [node.name for node in model.graph.node]
    # nodes_to_quantize = [
    #     name for name in all_node_names if name != output_node_name]

    quantize_static(
        model_input=prepared_path,
        model_output=int8_model_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,           # Use signed for weights
        activation_type=QuantType.QUInt8,      # Unsigned for most activations
        # nodes_to_quantize=nodes_to_quantize,   # Exclude output node
        calibrate_method=CalibrationMethod.MinMax,  # Try entropy for better accuracy
        extra_options={'WeightSymmetric': True, 'ActivationSymmetric': False}
    )

    print('Static Quantization Complete!')

    session = ort.InferenceSession(int8_model_path, providers=[
                                   'CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name

    input_shape = session.get_inputs()[0].shape
    print(f"Input Name: {input_name}")
    print(f"Input Shape: {input_shape}")

    data = np.random.randn(1, 3, 224, 224).astype(np.float32)
    outputs = session.run(None, {input_name: data})

    prediction = cast(np.ndarray, outputs[0])
    print('Output shape:', prediction.shape)
    print('First 5 values:', prediction.flatten()[:5])


if __name__ == '__main__':
    main()
