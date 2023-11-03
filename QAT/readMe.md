# Quantization Aware Training

Quantization Aware Training applies quantization on the forward pass of a neural network, meaning the loss function is calculated based on the actual output an FPGA would generate.
This is then backpropagated onto the non-quantized weights using normal differentials, e.g. using straight through estimation.

The script was ran with Vitis AI 1.4.1, as I was using the Alveo U50, and 1.4.1 was the last version to support this. This also means the version of pytorch is old, and you  may need to convert your .pth files using https://pytorch.org/docs/stable/generated/torch.save.html, with __use_new_zipfile_serialization=False.

I haven't included any of the dependencies for these scripts in this github, but QAT needs to be ran in the Vitis AI docker, in pytorch mode.

The dependencies for whatever neural network you are using will probably need to be installed by hand through trial and error: you may need to remove parts of the neural network scripts that aren't crucial if you cannot find a dependency that is installable into the Vitis AI docker.

## QAT Example

QAT example holds the original file I downloaded off the internet as the basis for the final script.

It performs QAT for a single CNN on MNIST.

## Original Train Script

Original Train Script holds train.py from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/train.py, which trains the pix2pix u-net on the GPU.

## Final Train
Final train is the actual QAT script I used to build the .xmodel files to use with a Vitis AI inference script, you can use the same .xmodel with my generic python inference script or the Nuke plugin inference script.

It is the result of splicing the two parents scripts together, with modifications to make them compatible.

Since I could never get Vitis AI 1.4.1 to run on the GPU, I would train the model normally on the GPU, convert the .pth file, and then do 5-20 QAT iterations on the CPU.
