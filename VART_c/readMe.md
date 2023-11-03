# VART C

This code is based off of https://github.com/Xilinx/Vitis-AI/blob/1.4.1/demo/VART/resnet50/src/main.cc, which is included as Vart C Example.

I originally wrote this thinking we would use it as a native Nuke plugin, but I never got the gcc versions to be compatible.
The ml server plugin became an easier choice at that point.

It was still useful though, as I couldn't get the profiler to run on the python script, but I could on the C program.

I don't know what the difference between the build scripts is anymore, so am including them both. build2.sh is very similiar to the one for the example file, if not the same, and build1.sh is one I reduced down a lot more.

