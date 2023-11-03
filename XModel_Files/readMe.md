# Compiling XModel Files

The xmodel file generated from the quantization procedure contains a IR of the program that includes data transfers and layer types, but is not FPGA specific, and so cannot be instantiated.

Before you compile the IR into specific operations, make sure to open the graph IR and observe the program flow.
If parts of your neural network cannot be performed by the DPU, the IR will have multiple transfers back and forth to the CPU, where it expects
you to manually code the parts of the neural network it cannot implement.

You should instead alter the neural network architecture to be DPU compatible, and then retrain and requantize.

## Compilation Script

Compile.sh is the script from https://github.com/Xilinx/Vitis-AI-Tutorials/blob/1.4/Design_Tutorials/09-mnist_pyt/files/compile.sh, which is used pretty much as is from compiling quantized xmodel files into FPGA specific versions.