# Computer-Vision-Project
This repository contains files related to our Computer-Vision project at the technion.
In this readme, we will explain how to use these files to run the desired networks on the Nvidia Jetson AGX Xavier GPU platform.

Out project is based upon the mmsegemantation open-source package for semenatic segmentation. Link to mmsegmentation official website: https://mmsegmentation.readthedocs.io/en/latest/

Before using our files, one should install mmsegmentation according to the instructions in the official website and make sure the package installed successfuly according to "verify the installation" section in the mmsegmentation installation guide.

After installation completed succeessfuly, one can download from this repository the RandomPicturesGenerator.py file. Running this file will generate white-noise pictures (pixles values ranging from 0 to 255 randomly) with height of 1080 pixles and width of 1440 pixles (one can change width and height in the code) and number of pictures according to input from user.
These pictures can be used to check networks performances without relation to real pictures.
Copy the generated pictures (the pictures themselves, not the folder) and paste them in the following route (if you dont have the folders in the route, create them manually):
mmsegmentation->data->cityscapes->leftimg8bit->val->lindau
make sure you also have the following route, create if not exist. (val should be empty folder):
mmsegmentation->data->cityscapes->gtFine->val

Next, one can download the two other files in the repository - test_with_logs.py, run_networks.py.
Place the two files in the following route: mmsegmentation->tools

run_networks.py is our main file and it runs the desierd networks in all Jettson's power modes.
Before running the file, make sure your working directory is mmsegmentation.
Inside the file, place the paths to the configs and checkpoints (make sure they placed in corresponding places in the lists) of the networks you want to run (you should also check you downloaded the checkpoints and configs from mmsegmenation website in the "Model Zoo Statistics": https://mmsegmentation.readthedocs.io/en/latest/modelzoo_statistics.html)

Switching between 10W power mode to any other power mode, requires restart of the jetson and that why you cannot run 10W power modes with all other power modes.
When running the script, you should enter "0" to run all power modes except 10W and enter "1" to run 10W power mode. Before each selection, make sure the current power mode is corresponding to your decision.
You will also be requested to enter yours Jetson sudo password. This information is needed to switch between power modes.

The output of this script will be placed in a new folder named "tegrastats_recordings" which will be places in mmsegmentaion folder. The output will include the following files:
For each network: 
	tegrastats recording in txt file
	Excel file of System time, RAM memory in use and GPU power consumption
	Graph of RAM memory used VS time
	Graph of GPU power consumption VS time
Inferences data - txt and Excel file which organizes the AVG RAM memory consumption and AVG GPU power consumption for each networks and each power mode
