import subprocess
from subprocess import Popen
import pandas as pd


def main():

    # delete all quantization_recordings directory content if exist
    cmd = 'rm -r quantization_recordings'
    subprocess.run(cmd, shell=True)

    # make quantization_recordings directory in mmsegmentation directory to store quantization recordings
    cmd = 'mkdir -p quantization_recordings'
    subprocess.run(cmd, shell=True)

    # create new file for quantization data
    cmd = 'touch ' + 'quantization_recordings/quantization_data.txt'
    subprocess.run(cmd, shell=True)
    cmd = 'printf "Network Power-Mode Quantization Time[sec] AVG-MEM-USED-SIZE[MB] AVG-GPU-POWER[mW] Model-Size[KB] mIoU' + '\n"' +' >> ' + 'quantization_recordings/quantization_data.txt'
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

        # run regular model. Store results in quantization_recordings->results
        cmd = 'python tools/test_origin.py '+config+' '+checkpoint+' '+'--show-dir quantization_recordings/results --eval mIoU'
        subprocess.run(cmd, shell=True)

        # run mmseg_quantize model. Store results in quantization_recordings->mmseg_quantize_results
        cmd = 'python tools/test_quantize_mmseg.py ' + config + ' ' + checkpoint + ' ' + '--show-dir quantization_recordings/mmseg_quantize_results --eval mIoU'
        subprocess.run(cmd, shell=True)

        # run pytorch_dynamic_quantize model. Store results in quantization_recordings->pytorch_dynamic_quantize_results
        cmd = 'python tools/test_quantize_dynamic_pytorch.py ' + config + ' ' + checkpoint + ' ' + '--show-dir quantization_recordings/pytorch_dynamic_quantize_results --eval mIoU'
        subprocess.run(cmd, shell=True)

    # create Excel file of inferences_data.txt results
    tmp_file = pd.read_csv('quantization_recordings/quantization_data.txt', sep=' ', header=0)
    tmp_file.to_excel('quantization_recordings/quantization_data..xlsx', index=None)

if __name__ == '__main__':
    main()