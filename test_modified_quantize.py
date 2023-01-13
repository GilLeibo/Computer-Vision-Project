# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import tempfile
import warnings

import mmcv
import numpy as np
import torch
from mmcv.engine import collect_results_cpu, collect_results_gpu
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from subprocess import Popen
import time
import subprocess
import pandas as pd
from statistics import mean
import math


def np2tmp(array, temp_file_name=None, tmpdir=None):
    """Save ndarray to local numpy file.

    Args:
        array (ndarray): Ndarray to save.
        temp_file_name (str): Numpy file name. If 'temp_file_name=None', this
            function will generate a file name with tempfile.NamedTemporaryFile
            to save ndarray. Default: None.
        tmpdir (str): Temporary directory to save Ndarray files. Default: None.
    Returns:
        str: The numpy file name.
    """

    if temp_file_name is None:
        temp_file_name = tempfile.NamedTemporaryFile(
            suffix='.npy', delete=False, dir=tmpdir).name
    np.save(temp_file_name, array)
    return temp_file_name


def single_gpu_test_modified_quantize(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    efficient_test=False,
                    opacity=0.5,
                    pre_eval=False,
                    format_only=False,
                    format_args={},
                    power_mode_str="",
                    network_name="",
                    tegra_filename="",
                    col_name=""):
    """Test with single GPU by progressive mode.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        show (bool): Whether show results during inference. Default: False.
        out_dir (str, optional): If specified, the results will be dumped into
            the directory to save output results.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.
    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx
    loader_indices = data_loader.batch_sampler

    pictures_MEM_AVG = []
    pictures_POWER_AVG = []
    pictures_TIMES = []

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():

            # start tegrastats recording
            cmd = 'sudo tegrastats --interval 10 --logfile quantization_recordings/' + tegra_filename + '.txt'
            Popen("exec " + cmd, shell=True)

            tic = time.perf_counter()  # start timer for inference

            # inference of current picture
            result = model(return_loss=False, **data)

            toc = time.perf_counter()  # stop timer for inference

            cmd = 'sudo tegrastats --stop'  # stop tegrastats recording
            subprocess.run(cmd, shell=True)

            # manipulate tegrastats file
            cmd = 'cat ' + 'quantization_recordings/' + tegra_filename + '.txt' + ' | tr -s "/" " " > quantization_recordings/tmp.txt'
            subprocess.run(cmd, shell=True)
            cmd = 'cat ' + 'quantization_recordings/tmp.txt' + ' | tr -s "mw" " " > quantization_recordings/tmp2.txt'
            subprocess.run(cmd, shell=True)

            tegrastats_MEM_data = pd.read_csv('quantization_recordings/tmp2.txt', sep=' ', usecols=[3], names=[col_name])
            tegrastats_Power_data = pd.read_csv('quantization_recordings/tmp2.txt', sep=' ', usecols=[32], names=[col_name])

            # remove temp  and tegrastats files
            cmd = 'rm quantization_recordings/tmp.txt'
            subprocess.run(cmd, shell=True)
            cmd = 'rm quantization_recordings/tmp2.txt'
            subprocess.run(cmd, shell=True)
            cmd = 'rm quantization_recordings/' + tegra_filename + '.txt'
            subprocess.run(cmd, shell=True)

            # calc MEM_AVG, POWER_AVG and Time
            elapsed_inference = float(f"{toc - tic:0.2f}")

            # potentially, dataframes can be empty if tegrastats wasn't fast enough to calc values.
            # In that case, use the last value from that list as the current value (we assume values should be similar)
            if(tegrastats_MEM_data.empty):
                MEM_avg = pictures_MEM_AVG[-1]
            else:
                MEM_avg = round(tegrastats_MEM_data.mean()[0])

            if(tegrastats_Power_data.empty):
                GPU_power_avg = pictures_POWER_AVG[-1]
            else:
                GPU_power_avg = round(tegrastats_Power_data.mean()[0])

            # add results to lists
            pictures_TIMES.append(elapsed_inference)
            pictures_MEM_AVG.append(MEM_avg)
            pictures_POWER_AVG.append(GPU_power_avg)


        if show or out_dir:
            img_tensor = data['img'][0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result,
                    palette=dataset.PALETTE,
                    show=show,
                    out_file=out_file,
                    opacity=opacity)

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.format_results(
                result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(result, indices=batch_indices)
            results.extend(result)
        else:
            results.extend(result)

        batch_size = len(result)
        for _ in range(batch_size):
            prog_bar.update()

    # write tegrastats TPP results to the network TPP csv file
    network_TPP_data = pd.read_csv('quantization_recordings/' + network_name + '_TPP.csv', header=0)
    tegrastats_TPP_data = pd.DataFrame(pictures_TIMES, columns=[col_name])
    TPP_result = pd.concat([network_TPP_data, tegrastats_TPP_data], axis=1)
    TPP_result.to_csv('quantization_recordings/' + network_name + '_TPP.csv', index=False)

    # write tegrastats MEM results to the network MEM csv file
    network_MEM_data = pd.read_csv('quantization_recordings/' + network_name + '_MEM.csv', header=0)
    tegrastats_MEM_data = pd.DataFrame(pictures_MEM_AVG, columns=[col_name])
    MEM_result = pd.concat([network_MEM_data, tegrastats_MEM_data], axis=1)
    MEM_result.to_csv('quantization_recordings/' + network_name + '_MEM.csv', index=False)

    # write tegrastats Power results to the network Power csv file
    network_Power_data = pd.read_csv('quantization_recordings/' + network_name + '_Power.csv', header=0)
    tegrastats_Power_data = pd.DataFrame(pictures_POWER_AVG, columns=[col_name])
    Power_result = pd.concat([network_Power_data, tegrastats_Power_data], axis=1)
    Power_result.to_csv('quantization_recordings/' + network_name + '_Power.csv', index=False)


    # calc MEM_AVG, POWER_AVG and Time_AVG to return to inferences_data file but without first and second pictures to avoid distortion
    pictures_TIMES.pop(0)
    pictures_TIMES.pop(0)
    pictures_MEM_AVG.pop(0)
    pictures_MEM_AVG.pop(0)
    pictures_POWER_AVG.pop(0)
    pictures_POWER_AVG.pop(0)

    pictures_TIMES_mean = str(round(mean(pictures_TIMES), 2))
    pictures_MEM_AVG_mean = str(round(mean(pictures_MEM_AVG)))
    pictures_POWER_AVG_mean = str(round(mean(pictures_POWER_AVG)))

    return results, pictures_TIMES_mean, pictures_MEM_AVG_mean, pictures_POWER_AVG_mean


def multi_gpu_test(model,
                   data_loader,
                   tmpdir=None,
                   gpu_collect=False,
                   efficient_test=False,
                   pre_eval=False,
                   format_only=False,
                   format_args={}):
    """Test model with multiple gpus by progressive mode.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (utils.data.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. The same path is used for efficient
            test. Default: None.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
            Default: False.
        efficient_test (bool): Whether save the results as local numpy files to
            save CPU memory during evaluation. Mutually exclusive with
            pre_eval and format_results. Default: False.
        pre_eval (bool): Use dataset.pre_eval() function to generate
            pre_results for metric evaluation. Mutually exclusive with
            efficient_test and format_results. Default: False.
        format_only (bool): Only format result for results commit.
            Mutually exclusive with pre_eval and efficient_test.
            Default: False.
        format_args (dict): The args for format_results. Default: {}.

    Returns:
        list: list of evaluation pre-results or list of save file names.
    """
    if efficient_test:
        warnings.warn(
            'DeprecationWarning: ``efficient_test`` will be deprecated, the '
            'evaluation is CPU memory friendly with pre_eval=True')
        mmcv.mkdir_or_exist('.efficient_test')
    # when none of them is set true, return segmentation results as
    # a list of np.array.
    assert [efficient_test, pre_eval, format_only].count(True) <= 1, \
        '``efficient_test``, ``pre_eval`` and ``format_only`` are mutually ' \
        'exclusive, only one of them could be true .'

    model.eval()
    results = []
    dataset = data_loader.dataset
    # The pipeline about how the data_loader retrieval samples from dataset:
    # sampler -> batch_sampler -> indices
    # The indices are passed to dataset_fetcher to get data from dataset.
    # data_fetcher -> collate_fn(dataset[index]) -> data_sample
    # we use batch_sampler to get correct data idx

    # batch_sampler based on DistributedSampler, the indices only point to data
    # samples of related machine.
    loader_indices = data_loader.batch_sampler

    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    for batch_indices, data in zip(loader_indices, data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        if efficient_test:
            result = [np2tmp(_, tmpdir='.efficient_test') for _ in result]

        if format_only:
            result = dataset.format_results(
                result, indices=batch_indices, **format_args)
        if pre_eval:
            # TODO: adapt samples_per_gpu > 1.
            # only samples_per_gpu=1 valid now
            result = dataset.pre_eval(result, indices=batch_indices)

        results.extend(result)

        if rank == 0:
            batch_size = len(result) * world_size
            for _ in range(batch_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results
