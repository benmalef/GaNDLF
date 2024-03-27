import os, datetime, subprocess, sys
from copy import deepcopy
from pprint import pprint
import random
import numpy as np
import torch
import SimpleITK as sitk
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull
from typing import Optional, Union


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def checkPatchDivisibility(patch_size: np.ndarray, number: Optional[int] = 16) -> bool:
    """
    This function checks the divisibility of a numpy array or integer for architectural integrity

    Args:
        patch_size (np.ndarray): The patch size for checking.
        number (Optional[int], optional): The number to check divisibility for. Defaults to 16.

    Returns:
        bool: If all elements of array are divisible or not, after taking 2D patches into account.
    """
    if isinstance(patch_size, int):
        patch_size_to_check = np.array(patch_size)
    else:
        patch_size_to_check = patch_size
    # for 2D, don't check divisibility of last dimension
    if patch_size_to_check[-1] == 1:
        patch_size_to_check = patch_size_to_check[:-1]
    # for 2D, don't check divisibility of first dimension
    elif patch_size_to_check[0] == 1:
        patch_size_to_check = patch_size_to_check[1:]
    if np.count_nonzero(np.remainder(patch_size_to_check, number)) > 0:
        return False

    # adding check to address https://github.com/mlcommons/GaNDLF/issues/53
    # there is quite possibly a better way to do this
    unique = np.unique(patch_size_to_check)
    if (unique.shape[0] == 1) and (unique[0] < number):
        return False
    return True


def get_date_time() -> str:
    """
    Get a well-parsed date string

    Returns:
        str: The date in format YYYY/MM/DD::HH:MM:SS
    """
    now = datetime.datetime.now().strftime("%Y/%m/%d::%H:%M:%S")
    return now


def get_unique_timestamp() -> str:
    """
    Get a well-parsed timestamp string to be used for unique filenames

    Returns:
        str: The date in format YYYYMMDD_HHMMSS
    """
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return now


def get_filename_extension_sanitized(filename: str) -> str:
    """
    This function returns the extension of the filename with leading and trailing characters removed.
    Args:
        filename (str): The filename to be processed.
    Returns:
        str: The filename with extension removed.
    """
    _, ext = os.path.splitext(filename)
    # if .gz or .nii file is detected, always return .nii.gz
    if (ext == ".gz") or (ext == ".nii"):
        ext = ".nii.gz"
    return ext


def parse_version(version_string: str) -> int:
    """
    Parses version string, discards last identifier (NR/alpha/beta) and returns an integer for comparison.

    Args:
        version_string (str): The string to be parsed.

    Returns:
        int: The version number.
    """
    version_string_split = version_string.replace("-dev", "")
    version_string_split = version_string_split.split(".")
    if len(version_string_split) > 3:
        del version_string_split[-1]
    return int("".join(version_string_split))


def version_check(version_from_config: str, version_to_check: str) -> bool:
    """
    This function checks if the version of the config file is compatible with the version of the code.

    Args:
        version_from_config (str): The version of the config file.
        version_to_check (str): The version of the code or model to check.

    Returns:
        bool: If the version of the config file is compatible with the version of the code.
    """
    version_to_check_int = parse_version(version_to_check)
    min_ver = parse_version(version_from_config["minimum"])
    max_ver = parse_version(version_from_config["maximum"])
    if (min_ver > version_to_check_int) or (max_ver < version_to_check_int):
        sys.exit(
            "Incompatible version of GaNDLF detected ("
            + str(version_to_check_int)
            + ")"
        )

    return True


def checkPatchDimensions(patch_size: np.ndarray, numlay: int) -> int:
    """
    This function checks the divisibility of a numpy array or integer for architectural integrity

    Args:
        patch_size (np.ndarray): The patch size for checking.
        number (int, optional): The number to check divisibility for. Defaults to 16.

    Returns:
        int: Largest multiple of 2 (less than or equal to numlay) that each element of patch size is divisible by to yield int >= 2
    """
    if isinstance(patch_size, int):
        patch_size_to_check = np.array(patch_size)
    else:
        patch_size_to_check = patch_size
    # for 2D, don't check divisibility of last dimension
    if patch_size_to_check[-1] == 1:
        patch_size_to_check = patch_size_to_check[:-1]

    if all(
        [x >= 2 ** (numlay + 1) and x % 2**numlay == 0 for x in patch_size_to_check]
    ):
        return numlay
    else:
        # base2 = np.floor(np.log2(patch_size_to_check))
        base2 = np.array([getBase2(x) for x in patch_size_to_check])
        remain = patch_size_to_check / 2**base2  # check that at least 1

        layers = np.where(remain == 1, base2 - 1, base2)
        return int(np.min(layers))


def getBase2(num: int) -> int:
    """
    Compute the base 2 logarithm of a number.

    Args:
        num (int): the number

    Returns:
        int: the base 2 logarithm of the number
    """
    # helper for checkPatchDimensions (returns the largest multiple of 2 that num is evenly divisible by)
    base = 0
    while num % 2 == 0:
        num = num / 2
        base = base + 1
    return base


def get_array_from_image_or_tensor(
    input_tensor_or_image: Union[torch.Tensor, sitk.Image]
) -> np.ndarray:
    """
    This function returns the numpy array from a torch.Tensor or sitk.Image.

    Args:
        input_tensor_or_image (Union[torch.Tensor, sitk.Image]): The input tensor or image.

    Returns:
        np.ndarray: The numpy array.
    """
    assert isinstance(
        input_tensor_or_image, (torch.Tensor, sitk.Image, np.ndarray)
    ), "Input must be a torch.Tensor or sitk.Image or np.ndarray, but got " + str(
        type(input_tensor_or_image)
    )
    if isinstance(input_tensor_or_image, torch.Tensor):
        return input_tensor_or_image.detach().cpu().numpy()
    elif isinstance(input_tensor_or_image, sitk.Image):
        return sitk.GetArrayFromImage(input_tensor_or_image)
    elif isinstance(input_tensor_or_image, np.ndarray):
        return input_tensor_or_image


def set_determinism(seed: Optional[int] = 42) -> None:
    """
    This function sets the determinism for the random number generators.

    Args:
        seed (Optional[int], optional): The seed for the random number generators. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    if torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def print_and_format_metrics(
    cohort_level_metrics: dict,
    sample_level_metrics: dict,
    metrics_dict_from_parameters: dict,
    mode: str,
    length_of_dataloader: int,
) -> dict:
    """
    This function prints and formats the metrics.

    Args:
        cohort_level_metrics (dict): The cohort level metrics calculated from the GANDLF.metrics.overall_stats function.
        sample_level_metrics (dict): The sample level metrics calculated from separate samples from the dataloader(s).
        metrics_dict_from_parameters (dict): The metrics dictionary to populate.
        mode (str): The mode of the metrics (train, val, test).
        length_of_dataloader (int): The length of the dataloader.

    Returns:
        dict: The metrics dictionary populated with the metrics.
    """

    def __update_metric_from_list_to_single_string(input_metrics_dict: dict) -> dict:
        """
        Helper function to update the metric from list to single string.

        Args:
            input_metrics_dict (dict): The input metrics dictionary.

        Returns:
            dict: The output metrics dictionary.
        """
        output_metrics_dict = deepcopy(input_metrics_dict)
        for metric in input_metrics_dict.keys():
            if isinstance(input_metrics_dict[metric], list):
                output_metrics_dict[metric] = ("_").join(
                    str(input_metrics_dict[metric])
                    .replace("[", "")
                    .replace("]", "")
                    .replace(" ", "")
                    .split(",")
                )

        pprint(output_metrics_dict)
        return output_metrics_dict

    output_metrics_dict = deepcopy(cohort_level_metrics)
    for metric in metrics_dict_from_parameters:
        if isinstance(sample_level_metrics[metric], np.ndarray):
            to_print = (sample_level_metrics[metric] / length_of_dataloader).tolist()
        else:
            to_print = sample_level_metrics[metric] / length_of_dataloader
        output_metrics_dict[metric] = to_print
    for metric in output_metrics_dict.keys():
        print(
            "     Epoch Final   " + mode + " " + metric + " : ",
            output_metrics_dict[metric],
        )
    output_metrics_dict = __update_metric_from_list_to_single_string(
        output_metrics_dict
    )

    return output_metrics_dict


def define_average_type_key(params: dict, metric_name: str) -> str:
    """
    Determine the average type key from the metric config.

    Args:
        params (dict): The parameter dictionary containing training and data information.
        metric_name (str): The name of the metric.

    Returns:
        str: The average type key.
    """
    average_type_key = params["metrics"][metric_name].get("average", "macro")
    return average_type_key


def define_multidim_average_type_key(params: dict, metric_name: str) -> str:
    """
    Determine the multidimensional average type key from the metric config.

    Args:
        params (dict): The parameter dictionary containing training and data information.
        metric_name (str): The name of the metric.

    Returns:
        str: The multidimensional average type key.
    """
    average_type_key = params["metrics"][metric_name].get("multidim_average", "global")
    return average_type_key


def determine_classification_task_type(params: dict) -> str:
    """
    This function determines the classification task type from the parameters.

    Args:
        params (dict): The parameter dictionary containing training and data information.

    Returns:
        str: The classification task type (binary or multiclass).
    """
    task = "binary" if params["model"]["num_classes"] == 2 else "multiclass"
    return task


def get_git_hash() -> str:
    """
    Get the git hash of the current commit.

    Returns:
        str: The git hash of the current commit.
    """
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.getcwd())
            .decode("ascii")
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_hash = "None"
    return git_hash
