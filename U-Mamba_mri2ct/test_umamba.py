import torch
from torch import nn
from umamba.nnunetv2.nets.UMambaBot_3d import UMambaBot

# Assuming the classes you provided are in the current namespace or imported
# We define standard parameters for a 3D segmentation task
input_channels = 1
num_classes = 1
n_stages = 5
# Example feature map progression: 32, 64, 128, 256, 512
features_per_stage = [32, 64, 128, 256, 512]
kernel_sizes = [[3, 3, 3]] * n_stages
strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
n_conv_per_stage = 2
n_conv_per_stage_decoder = 2

# 1. Initialize the Model
print("Initializing UMambaBot...")
model = UMambaBot(
    input_channels=input_channels,
    n_stages=n_stages,
    features_per_stage=features_per_stage,
    conv_op=nn.Conv3d,
    kernel_sizes=kernel_sizes,
    strides=strides,
    n_conv_per_stage=n_conv_per_stage,
    num_classes=num_classes,
    n_conv_per_stage_decoder=n_conv_per_stage_decoder,
    conv_bias=True,
    norm_op=nn.InstanceNorm3d,
    norm_op_kwargs={'eps': 1e-5, 'affine': True},
    nonlin=nn.LeakyReLU,
    nonlin_kwargs={'inplace': True},
    deep_supervision=False  # Set to False for simple output check
)

# 2. Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
print(f"Model moved to {device}")

# 3. Create Dummy Input (Batch, Channels, Depth, Height, Width)
# Small dimensions to ensure it fits in memory during the first test
dummy_input = torch.randn(1, input_channels, 32, 32, 32).to(device)
print(f"Input shape: {dummy_input.shape}")

# 4. Perform Forward Pass
try:
    with torch.no_grad():
        output = model(dummy_input)
    
    print("--- Forward Pass Successful ---")
    if isinstance(output, (list, tuple)):
        print(f"Deep Supervision output count: {len(output)}")
        print(f"Primary output shape: {output[0].shape}")
    else:
        print(f"Output shape: {output.shape}")
        
except Exception as e:
    print(f"--- Forward Pass Failed ---")
    print(e)