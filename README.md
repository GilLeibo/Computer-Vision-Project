# Computer-Vision-Project
This repository contains files related to our Computer-Vision project at the Technion.
In this readme, we will explain how to use these files to run the desired networks on the Nvidia Jetson AGX Xavier GPU platform.

Our project is based upon the mmsegmentation open-source package for semantic segmentation. Link to mmsegmentation official website: https://mmsegmentation.readthedocs.io/en/latest/

Before using our files, one should install mmsegmentation according to the instructions on the official website and make sure the package is installed successfully according to the "verify the installation" section in the mmsegmentation installation guide.

After installation is completed successfully, one can download from this repository the RandomPicturesGenerator.py file (To code the file we used this reference: https://www.geeksforgeeks.org/convert-a-numpy-array-to-an-image/). Running this file will generate white-noise pictures (pixels values ranging from 0 to 255 randomly) with a height of 1080 pixels and width of 1440 pixels (one can change width and height in the code). The number of pictures to generate is according to the input from the user.
These pictures can be used to check the networks performances without relation to real pictures.<br>
Copy the generated pictures (the pictures themselves, not the folder) and paste them in the following route (if you don't have the folders in the route, create them manually):<br>
**mmsegmentation->data->cityscapes->leftimg8bit->val->lindau** <br>
make sure you also have the following route, create if not exist (val should be an empty folder):<br>
**mmsegmentation->data->cityscapes->gtFine->val**

Next, one can download the two other files in the repository - test_with_logs.py, run_networks.py.
Place the two files in the following route:
**mmsegmentation->tools**

run_networks.py is our main file and it runs the desired networks in test_with_logs.py with all of Jettson's power modes.
Before running the file, make sure your working directory is mmsegmentation.
Inside the file, paste the paths to the configs and checkpoints of the networks you want to run. Make sure they are placed in corresponding places in the lists. You should also check you downloaded the checkpoints and configs from mmsegmentation website in the "Model Zoo Statistics": https://mmsegmentation.readthedocs.io/en/latest/modelzoo_statistics.htmly and placed them in the mmsegmentation->checkpoints and mmsegmentation->configs folders.

Switching between 10W power mode to any other power mode, requires restart of the jetson, and that is why you cannot run 10W power mode with all other power modes.
When running the script and input is required, you should enter "0" to run all power modes except 10W and enter "1" to run 10W power mode only. Before each selection, make sure the current power mode is corresponding to your decision.
You will also be requested to enter your Jetson sudo password. This information is needed to switch between power modes.

The output of this script will be placed in a new folder named "tegrastats_recordings" which will be placed in mmsegmentation folder. The output will include the following files:

For each network:<br>
- tegrastats recording in txt file<br>
- Excel file of System time, RAM in use and GPU power consumption<br>
- Graph of RAM used VS time<br>
- Graph of GPU power consumption VS time<br>

Inferences data file:<br> 
- Excel file which organizes the average RAM consumption, average GPU power consumption for each of the networks and each power mode.

## Quantization
One of the project goals was to implement quantization techniques to improve inferences' times and memory consumption. We used the mmsegmentation's function which enables quantization from 32FP to 16FP. A description of that function can be found on mmsegmentation's official website: https://mmcv.readthedocs.io/en/latest/_modules/mmcv/runner/fp16_utils.html.

To run the desired networks and to analyze the performances, one can download these files: run_networks_quantize.py, test_origin.py, test_quantize_mmseg.py. Also, one should download Cityscapes dataset: https://www.cityscapes-dataset.com/.
The dataset should be placed in corresponding places at:
**mmsegmentation->data->cityscapes->leftimg8bit->val** and 
**mmsegmentation->data->cityscapes->gtFine->val**.

Our main script is run_networks_quantize.py which runs the two other scripts. Parameters that should be initizalied are similar to run_networks.py as we explained before.

The output of the script will be placed in **mmsegmentation->quantization_recordings**. Inside the directory will be two other directories:<br> 
- results-which contains the input pictures after segmentation of the network **without** qunatization
- quantization_results-which contains the input pictures after segmentation of the network **with** qunatization

Also, for each network, files will be generated the same as explained in run_networks.py. The overall results will be places in quantization_data Excel file that will conatin information similar as explained for run_networks.py with extra columns for quantization techqniue and mIoU result.
