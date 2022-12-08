import subprocess
from subprocess import Popen
import pandas as pd


def main():

    # delete all tegrastats_recordings directory content if exist
    cmd = 'rm tegrastats_recordings/*'
    subprocess.run(cmd, shell=True)

    # make tegrastats_recordings directory in mmsegmentation directory to store tegrastats recordings
    cmd = 'mkdir -p tegrastats_recordings'
    subprocess.run(cmd, shell=True)

    # create new file for inferences data
    cmd = 'touch ' + 'tegrastats_recordings/inferences_data.txt'
    subprocess.run(cmd, shell=True)
    cmd = 'printf "Network Power-Mode Time[sec] AVG-MEM-USED-SIZE[MB] AVG-GPU-POWER[mW]' + '\n"' +' >> ' + 'tegrastats_recordings/inferences_data.txt'
    subprocess.run(cmd, shell=True)

    # ask sudo password from user to be able to change power modes
    sudo_password = input("Enter sudo password: \n")

    # ask for which power modes to run the networks
    power_modes_entry = input("Enter 0 to run networks on all power modes except 10W (make sure your current power mode isn't 10W)\n"
                              "Enter 1 to run networks on 10W only (make sure your current power mode is 10W): \n")

    if power_modes_entry=='0':
        """""
            # run networks on all jetson power modes except 10W power mode, since it requires restart of the jetson.
            # legend (power_mode_num=power_mode_description):
            0=MAXN
            2=MODE_15W
            3=MODE_30W_ALL
            4=MODE_30W_6CORE
            5=MODE_30W_4CORE
            6=MODE_30W_2CORE
            """
        power_modes = [0, 2, 3, 4, 5, 6]

    elif power_modes_entry=='1':
        # run networks on 10W only power mode
        power_modes = [1]

    else:
        print("Input invalid\n")
        exit()

    # paths to networks config files (one should validate config files pre-downloaded to mmsegmentaion->configs folder)
    configs = [
        'configs/bisenetv1/bisenetv1_r18-d32_4x4_1024x1024_160k_cityscapes.py',
        'configs/bisenetv1/bisenetv1_r50-d32_in1k-pre_4x4_1024x1024_160k_cityscapes.py',
        'configs/bisenetv2/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes.py',
        'configs/bisenetv2/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes.py',
        'configs/segformer/segformer_mit-b0_8x1_1024x1024_160k_cityscapes.py',
        'configs/segformer/segformer_mit-b5_8x1_1024x1024_160k_cityscapes.py',
        'configs/deeplabv3plus/deeplabv3plus_r18-d8_769x769_80k_cityscapes.py',
        'configs/deeplabv3plus/deeplabv3plus_r101-d8_512x1024_80k_cityscapes.py',
    ]

    # paths to networks checkpoints files (one should validate checkpoints files pre-downloaded to mmsegmentaion->checkpoints folder)
    checkpoints = [
        'checkpoints/bisenetv1_r18-d32_4x4_1024x1024_160k_cityscapes_20210922_172239-c55e78e2.pth',
        'checkpoints/bisenetv1_r50-d32_in1k-pre_4x4_1024x1024_160k_cityscapes_20210917_234628-8b304447.pth',
        'checkpoints/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes_20210902_015551-bcf10f09.pth',
        'checkpoints/bisenetv2_fcn_4x8_1024x1024_160k_cityscapes_20210903_000032-e1a2eed6.pth',
        'checkpoints/segformer_mit-b0_8x1_1024x1024_160k_cityscapes_20211208_101857-e7f88502.pth',
        'checkpoints/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth',
        'checkpoints/deeplabv3plus_r18-d8_769x769_80k_cityscapes_20201226_083346-f326e06a.pth',
        'checkpoints/deeplabv3plus_r101-d8_512x1024_80k_cityscapes_20200606_114143-068fcfe9.pth',
    ]

    # run inference on all power modes and all networks
    for power in power_modes:

        # change power mode
        cmd = 'echo ' + sudo_password + ' | sudo -S nvpmodel -m ' + str(power)
        subprocess.run(cmd, shell=True)

        for config, checkpoint in zip(configs, checkpoints):

            cmd = 'python tools/test_with_logs.py '+config+' '+checkpoint+' '+'--show-dir results'
            subprocess.run(cmd, shell=True)

    # create Excel file of inferences_data.txt results
    tmp_file = pd.read_csv('tegrastats_recordings/inferences_data.txt', sep=' ', header=0)
    tmp_file.to_excel('tegrastats_recordings/inferences_data.xlsx', index=None)

if __name__ == '__main__':
    main()