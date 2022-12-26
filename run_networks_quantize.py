import subprocess
from PIL import Image, ImageChops
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():

    # delete all quantization_recordings directory content if exist
    cmd = 'rm -r quantization_recordings'
    subprocess.run(cmd, shell=True)

    # make quantization_recordings directory in mmsegmentation directory to store quantization recordings
    cmd = 'mkdir -p quantization_recordings'
    subprocess.run(cmd, shell=True)

    # create new file for quantization data
    cmd = 'touch quantization_recordings/quantization_data.txt'
    subprocess.run(cmd, shell=True)
    cmd = 'printf "Network Power-Mode Quantization Time[sec] MEDIAN-MEM-USED-SIZE[MB] AVG-MEM-USED-SIZE[MB] MEDIAN-GPU-POWER[mW] AVG-GPU-POWER[mW] Model-Size[KB] mIoU' + '\n"' +' >> ' + 'quantization_recordings/quantization_data.txt'
    subprocess.run(cmd, shell=True)

    # ask sudo password from user to be able to activate tegrastats
    sudo_password = input("Enter sudo password: \n")
    cmd = 'echo ' + sudo_password + ' | sudo -S tegrastats --stop'
    subprocess.run(cmd, shell=True)

    # paths to networks config files (one should validate config files pre-downloaded to mmsegmentaion->configs folder)
    configs = [
        'configs/bisenetv1/bisenetv1_r18-d32_4x4_1024x1024_160k_cityscapes.py',
        'configs/bisenetv2/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes.py',
    ]

    # paths to networks checkpoints files (one should validate checkpoints files pre-downloaded to mmsegmentaion->checkpoints folder)
    checkpoints = [
        'checkpoints/bisenetv1_r18-d32_4x4_1024x1024_160k_cityscapes_20210922_172239-c55e78e2.pth',
        'checkpoints/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes_20210902_015551-bcf10f09.pth',
    ]

    for config, checkpoint in zip(configs, checkpoints):

        # create csv files for the network MEM and Power
        network_name = config.split('/')[2][:-3]
        cmd = 'touch quantization_recordings/' + network_name + '_MEM.csv'
        subprocess.run(cmd, shell=True)
        cmd = 'printf "Empty\n" >> ' + 'quantization_recordings/' + network_name + '_MEM.csv'
        subprocess.run(cmd, shell=True)
        cmd = 'touch quantization_recordings/' + network_name + '_Power.csv'
        subprocess.run(cmd, shell=True)
        cmd = 'printf "Empty\n" >> ' + 'quantization_recordings/' + network_name + '_Power.csv'
        subprocess.run(cmd, shell=True)

        # run regular model. Store results in quantization_recordings->origin_results
        cmd = 'python tools/test_origin.py '+config+' '+checkpoint+' '+'--show-dir quantization_recordings/origin_results --eval mIoU'
        subprocess.run(cmd, shell=True)

        # run mmseg_quantize model. Store results in quantization_recordings->mmseg_quantize_results
        cmd = 'python tools/test_quantize_mmseg.py ' + config + ' ' + checkpoint + ' ' + '--show-dir quantization_recordings/mmseg_quantize_results --eval mIoU'
        subprocess.run(cmd, shell=True)

        # run pytorch_dynamic_quantize model. Store results in quantization_recordings->pytorch_dynamic_quantize_results
        cmd = 'python tools/test_quantize_dynamic_pytorch.py ' + config + ' ' + checkpoint + ' ' + '--show-dir quantization_recordings/pytorch_dynamic_quantize_results --eval mIoU'
        subprocess.run(cmd, shell=True)

        # remove first empty column of csv files for the network MEM and Power
        network_MEM_data = pd.read_csv('quantization_recordings/' + network_name + '_MEM.csv', header=0)
        network_MEM_data.drop(columns=network_MEM_data.columns[0], axis=1, inplace=True)
        network_Power_data = pd.read_csv('quantization_recordings/' + network_name + '_Power.csv', header=0)
        network_Power_data.drop(columns=network_Power_data.columns[0], axis=1, inplace=True)

        # Plot and save graphs for network MEM and Power
        plotAndSaveGraph(network_MEM_data, "MEM", network_name, "[MB]")
        plotAndSaveGraph(network_Power_data, "Power", network_name, "[mW]")

        # delete csv files of MEM and Power of the network
        cmd = 'rm quantization_recordings/' + network_name + '_MEM.csv'
        subprocess.run(cmd, shell=True)
        cmd = 'rm quantization_recordings/' + network_name + '_Power.csv'
        subprocess.run(cmd, shell=True)


    # create Excel file from inferences_data.txt results and delete inferences_data.txt
    tmp_file = pd.read_csv('quantization_recordings/quantization_data.txt', sep=' ', header=0)
    tmp_file.to_excel('quantization_recordings/quantization_data.xlsx', index=None)
    cmd = 'rm quantization_recordings/quantization_data.txt'
    subprocess.run(cmd, shell=True)

    # generate diff pictures of quantize picture result of origin-mmseg and origin-pytorch and save result to quantization_recordings folder
    # assign images
    img_origin = Image.open("quantization_recordings/origin_results/lindau/lindau_000000_000019_leftImg8bit.png")
    img_mmseg_quantize = Image.open(
        "quantization_recordings/mmseg_quantize_results/lindau/lindau_000000_000019_leftImg8bit.png")
    img_pytorch_dynamic_quantize = Image.open(
        "quantization_recordings/pytorch_dynamic_quantize_results/lindau/lindau_000000_000019_leftImg8bit.png")

    # finding differences
    diff_origin_mmseg = ImageChops.difference(img_origin, img_mmseg_quantize)
    diff_origin_pytorch = ImageChops.difference(img_origin, img_pytorch_dynamic_quantize)

    # save images in quantization_recordings folder
    diff_origin_mmseg.save("quantization_recordings/diff_origin_mmseg.png")
    diff_origin_pytorch.save("quantization_recordings/diff_origin_pytorch.png")


def plotAndSaveGraph(df, value, network_name, units):

    # clear figure state (so plots won't override)
    plt.clf()

    for index, colname in enumerate(df):
        x = np.arange(0, len(df.loc[:, colname]))
        y = df.loc[:, colname]

        # Plotting both the curves simultaneously
        plt.plot(x, y, label=colname)

    plt.xlabel("Time")
    plt.ylabel(value + units)
    plt.title(network_name + '_' + value + units)
    plt.legend()

    # save figure
    plt.savefig('quantization_recordings/' + network_name + '_' + value + '.png')

if __name__ == '__main__':
    main()