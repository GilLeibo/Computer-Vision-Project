import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
    cmd = 'printf "Network Power-Mode Time[sec] MEDIAN-MEM-USED-SIZE[MB] AVG-MEM-USED-SIZE[MB] MEDIAN-GPU-POWER[mW] AVG-GPU-POWER[mW]' + '\n"' + ' >> ' + 'tegrastats_recordings/inferences_data.txt'
    subprocess.run(cmd, shell=True)

    # ask sudo password from user to be able to change power modes
    sudo_password = input("Enter sudo password: \n")

    # ask for which power modes to run the networks
    power_modes_entry = input("Enter 0 to run networks on all power modes except 10W (make sure your current power mode isn't 10W)\n"
                              "Enter 1 to run networks on 10W only (make sure your current power mode is 10W): \n")
    power_modes = None

    if power_modes_entry == '0':
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
        #power_modes = [0, 2, 3, 4, 5, 6]
        power_modes = [0]

    elif power_modes_entry == '1':
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

        # get current power mode string
        cmd = 'nvpmodel -q'
        cmd_out = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        power_mode_str = cmd_out.stdout.splitlines()[0].split()[-1]

        # create csv files for the current power_mode: MEM and Power
        cmd = 'touch tegrastats_recordings/' + power_mode_str + '_MEM.csv'
        subprocess.run(cmd, shell=True)
        cmd = 'printf "Empty\n" >> ' + 'tegrastats_recordings/' + power_mode_str + '_MEM.csv'
        subprocess.run(cmd, shell=True)
        cmd = 'touch tegrastats_recordings/' + power_mode_str + '_Power.csv'
        subprocess.run(cmd, shell=True)
        cmd = 'printf "Empty\n" >> ' + 'tegrastats_recordings/' + power_mode_str + '_Power.csv'
        subprocess.run(cmd, shell=True)

        for config, checkpoint in zip(configs, checkpoints):

            # run inference with corresponding power mode and network
            cmd = 'python tools/test_with_logs.py ' + config + ' ' + checkpoint + ' ' + '--show-dir results'
            subprocess.run(cmd, shell=True)

        # remove first empty column of csv files for the current power_mode MEM and Power
        power_mode_MEM_data = pd.read_csv('tegrastats_recordings/' + power_mode_str + '_MEM.csv', header=0)
        power_mode_MEM_data.drop(columns=power_mode_MEM_data.columns[0], axis=1, inplace=True)
        power_mode_Power_data = pd.read_csv('tegrastats_recordings/' + power_mode_str + '_Power.csv', header=0)
        power_mode_Power_data.drop(columns=power_mode_Power_data.columns[0], axis=1, inplace=True)

        # Plot and save graphs for network MEM and Power
        plotAndSaveGraph(power_mode_MEM_data, "MEM", power_mode_str, "[MB]")
        plotAndSaveGraph(power_mode_Power_data, "Power", power_mode_str, "[mW]")

        # delete csv files of MEM and Power of the current power_mode
        cmd = 'rm tegrastats_recordings/' + power_mode_str + '_MEM.csv'
        subprocess.run(cmd, shell=True)
        cmd = 'rm tegrastats_recordings/' + power_mode_str + '_Power.csv'
        subprocess.run(cmd, shell=True)

    # create Excel file of inferences_data.txt results and delete inferences_data.txt
    tmp_file = pd.read_csv('tegrastats_recordings/inferences_data.txt', sep=' ', header=0)
    tmp_file.to_excel('tegrastats_recordings/inferences_data.xlsx', index=None)
    cmd = 'rm tegrastats_recordings/inferences_data.txt'
    subprocess.run(cmd, shell=True)


def plotAndSaveGraph(df, value, power_mode_str, units):

    # clear figure state (so plots won't override)
    plt.clf()

    for index, colname in enumerate(df):
        x = np.arange(0, len(df.loc[:, colname]))
        y = df.loc[:, colname]

        # Plotting both the curves simultaneously
        plt.plot(x, y, label=colname)

    plt.xlabel("Time")
    plt.ylabel(value + units)
    plt.title(power_mode_str + '-' + value + units)
    plt.legend()

    # save figure
    plt.savefig('tegrastats_recordings/' + power_mode_str + '-' + value + '.png')

if __name__ == '__main__':
    main()