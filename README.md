# Dehazing Model Optimization Framework

This framework provides tools for converting, compiling, quantizing, and deploying dehazing models to Qualcomm AI platforms.

## Table of Contents
- [Directory Structure](#directory-structure)
- [Setup Instructions](#setup-instructions)
- [Model Types and Formats](#model-types-and-formats)
- [Using the Framework](#using-the-framework)
- [Workflow Examples](#workflow-examples)
- [Troubleshooting](#troubleshooting)

## Directory Structure

```
DVA439 Intelligent Systems/
├── main.py                      # Main interface script
├── quantization.py              # Model quantization utilities
├── profiling.py                 # Model profiling utilities
├── full_run.py                  # End-to-end pipeline script
├── model_repos/                 # Repository for model architectures
│   ├── ChaIR/                   # ChaIR dehazing model
│   ├── CasDyF_Net/              # CasDyF-Net model
│   └── SFNet/                   # SFNet dehazing model
├── pretrained_models/           # Storage for model weights
│   ├── pkl/                     # Original PyTorch pickle models
│   ├── pt/                      # TorchScript models
│   └── onnx/                    # ONNX format models
├── datasets/                    # Training and test datasets
│   └── reside-outdoor/          # RESIDE outdoor dehazing dataset
├── outputs/                     # Output files from inference
│   ├── h5_files/                # HDF5 format output files
│   └── png_files/               # Converted PNG images
```

## Setup Instructions

1. **Environment Setup**:
    - Create a virtual environment and activate it
    ```bash
    python -m venv venv 
    ```
    ```bash
    .\venv\Scripts\activate
    ```
   ```bash
   pip install numpy requests torch torchvision pillow qai_hub
   ```

2. **Qualcomm AI Hub Setup**:
   - Register for an account at [Qualcomm AI Hub](https://qai-hub.qualcomm.com/)
   - Install the Qualcomm AI Hub Python package

    ```bash
    pip install qai-hub
    ```
    ```bash
    qai-hub configure --api_token <API_TOKEN>
    ```

   - Set up your API key as environment variable: `export QUALCOMM_AI_HUB_API_KEY=your_api_key`

## Model Types and Formats

This framework supports the following models and formats:

### Supported Models
- **ChaIR**: Dehazing model
- **IRNeXt**: Rethinking Convolutional Network Design for Image Restoration
- **FocalNet**: Focal Network for Image Restoration

### Supported File Formats
- **PKL (.pkl)**: Original PyTorch model weights
- **TorchScript (.pt)**: Traced/scripted PyTorch models
- **ONNX (.onnx)**: Open Neural Network Exchange format
- **TFLite (.tflite)**: TensorFlow Lite format

## Using the Framework

The main interface is through `main.py`, which provides an interactive menu for all operations.

### Starting the Framework

```bash
python main.py
```

### Available Operations

1. **Trace a model**: Convert PyTorch .pkl models to TorchScript (.pt)
   - Choose a model architecture
   - Provide path to .pkl file from the pretrained_models/pkl directory

2. **Compile a model**: Convert to ONNX/TFLite for deployment
   - Option to compile using QUI (Qualcomm AI Hub) or TorchScript
   - Results saved to pretrained_models/onnx directory

3. **Quantize a model**: Optimize model size and performance
   - Provide path to ONNX model or Job-ID for Qualcomm AI Hub model
   - Uses post-training quantization techniques

4. **Profile a model**: Analyze performance metrics
   - Measures inference time, memory usage, etc.

5. **Run Inference**: Execute the model on sample images
   - Requires a compiled model job code from Qualcomm AI Hub
   - Can use default test image or specify custom image

6. **Download a Model**: Retrieve compiled model from Qualcomm AI Hub
   - Results saved to pretrained_models/onnx directory

7. **Download Output**: Get inference results from Qualcomm AI Hub
   - Results saved to outputs directory

8. **Full Run**: Execute complete pipeline from model selection to inference

## Workflow Examples

### Example 1: Convert a ChaIR model from PKL to ONNX

1. Start the framework: `python main.py`
2. Choose Option 1 (Trace a model)
3. Select model 1 (ChaIR)
4. Enter PKL file path: `pretrained_models/pkl/ChaIR/model_ots_4073_9968.pkl`
5. Choose 'y' to save the TorchScript model
6. Choose Option 2 (Compile a model)
7. Enter the PT file path: `pretrained_models/pt/ChaIR_model_ots_4073_9968.pt`
8. Select 't' for TorchScript compilation

### Example 2: Run Inference on Qualcomm Device

1. Start the framework: `python main.py`
2. Choose Option 5 (Run Inference)
3. Enter the job code from Qualcomm AI Hub compilation
4. Enter sample image path or use default
5. Choose Option 7 (Download Output)
6. Enter the inference job code
7. Select 'y' to convert output to PNG

## Troubleshooting

- **File not found errors**: Ensure the correct file paths are provided relative to the project root
- **Compilation failures**: Check that input shapes match the model architecture (default: 1,3,256,256)
- **Qualcomm AI Hub issues**: Verify API key is set correctly and account has necessary permissions
- **Memory errors**: Large models may require additional system resources

For more information, refer to the documentation of individual model repositories in the model_repos directory.
