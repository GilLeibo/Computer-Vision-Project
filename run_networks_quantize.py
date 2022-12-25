import subprocess
from PIL import Image, ImageChops
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

        # run regular model. Store results in quantization_recordings->origin_results
        cmd = 'python tools/test_origin.py '+config+' '+checkpoint+' '+'--show-dir quantization_recordings/origin_results --eval mIoU'
        subprocess.run(cmd, shell=True)

        # run mmseg_quantize model. Store results in quantization_recordings->mmseg_quantize_results
        cmd = 'python tools/test_quantize_mmseg.py ' + config + ' ' + checkpoint + ' ' + '--show-dir quantization_recordings/mmseg_quantize_results --eval mIoU'
        subprocess.run(cmd, shell=True)

        # run pytorch_dynamic_quantize model. Store results in quantization_recordings->pytorch_dynamic_quantize_results
        cmd = 'python tools/test_quantize_dynamic_pytorch.py ' + config + ' ' + checkpoint + ' ' + '--show-dir quantization_recordings/pytorch_dynamic_quantize_results --eval mIoU'
        subprocess.run(cmd, shell=True)

    # create Excel file of inferences_data.txt results
    tmp_file = pd.read_csv('quantization_recordings/quantization_data.txt', sep=' ', header=0)
    tmp_file.to_excel('quantization_recordings/quantization_data.xlsx', index=None)

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

if __name__ == '__main__':
    main()