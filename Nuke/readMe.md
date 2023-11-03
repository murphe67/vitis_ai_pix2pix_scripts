# Nuke ML Server Plugin

The details of the plugin can be installed here:
https://github.com/TheFoundryVisionmongers/nuke-ML-server

Client installation works exactly as described.

The server should be cloned into the Vitis AI docker itself, and then its dependencies manually installed when you try to run it.
The default gaussian blur is a good way to experiment with the server, as it runs on the CPU.

You will probably have to remove parts of the server dependencies that can't be installed in the Vitis AI docker, I think it was an openexr package I removed.

I also for some reason copied the models folder into the server folder, and you need to place the .xmodel files relative to server.py

![server breadcrumb of where the folders are](https://github.com/murphe67/vitis_ai_pix2pix_scripts/blob/main/images/server_breadcrumb.png?raw=true)


Also don't use any of the GPU models on a computer without a GPU, or in the Vitis AI docker, which doesn't support GPUs, or everything will crash.

The final_server_model.py also uses the image unpacking and packing to process high resolution images.