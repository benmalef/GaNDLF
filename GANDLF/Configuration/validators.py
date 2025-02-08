import traceback

from GANDLF.Configuration.scheduler_parameters import Scheduler
from GANDLF.Configuration.utils import initialize_key
from GANDLF.metrics import surface_distance_ids


def validate_loss_function(value) -> dict:
    if isinstance(value, dict):  # if this is a dict
        if len(value) > 0:  # only proceed if something is defined
            for key in value:  # iterate through all keys
                if key == "mse":
                    if (value[key] is None) or not ("reduction" in value[key]):
                        value[key] = {}
                        value[key]["reduction"] = "mean"
                else:
                    # use simple string for other functions - can be extended with parameters, if needed
                    value = key
    else:
        if value == "focal":
            value = {"focal": {}}
            value["focal"]["gamma"] = 2.0
            value["focal"]["size_average"] = True
        elif value == "mse":
            value = {"mse": {}}
            value["mse"]["reduction"] = "mean"

    return value


def validate_metrics(value) -> dict:
    if not isinstance(value, dict):
        temp_dict = {}
    else:
        temp_dict = value

    # initialize metrics dict
    for metric in value:
        # assigning a new variable because some metrics can be dicts, and we want to get the first key
        comparison_string = metric
        if isinstance(metric, dict):
            comparison_string = list(metric.keys())[0]
        # these metrics always need to be dicts
        if comparison_string in [
            "accuracy",
            "f1",
            "precision",
            "recall",
            "specificity",
            "iou",
        ]:
            if not isinstance(metric, dict):
                temp_dict[metric] = {}
            else:
                temp_dict[comparison_string] = metric
        elif not isinstance(metric, dict):
            temp_dict[metric] = None

        # special case for accuracy, precision, recall, and specificity; which could be dicts
        ## need to find a better way to do this
        if any(
            _ in comparison_string
            for _ in ["precision", "recall", "specificity", "accuracy", "f1"]
        ):
            if comparison_string != "classification_accuracy":
                temp_dict[comparison_string] = initialize_key(
                    temp_dict[comparison_string], "average", "weighted"
                )
                temp_dict[comparison_string] = initialize_key(
                    temp_dict[comparison_string], "multi_class", True
                )
                temp_dict[comparison_string] = initialize_key(
                    temp_dict[comparison_string], "mdmc_average", "samplewise"
                )
                temp_dict[comparison_string] = initialize_key(
                    temp_dict[comparison_string], "threshold", 0.5
                )
                if comparison_string == "accuracy":
                    temp_dict[comparison_string] = initialize_key(
                        temp_dict[comparison_string], "subset_accuracy", False
                    )
        elif "iou" in comparison_string:
            temp_dict["iou"] = initialize_key(
                temp_dict["iou"], "reduction", "elementwise_mean"
            )
            temp_dict["iou"] = initialize_key(temp_dict["iou"], "threshold", 0.5)
        elif comparison_string in surface_distance_ids:
            temp_dict[comparison_string] = initialize_key(
                temp_dict[comparison_string], "connectivity", 1
            )
            temp_dict[comparison_string] = initialize_key(
                temp_dict[comparison_string], "threshold", None
            )

    value = temp_dict
    return value


def validate_class_list(value):
    if isinstance(value, str):
        if ("||" in value) or ("&&" in value):
            # special case for multi-class computation - this needs to be handled during one-hot encoding mask construction
            print(
                "WARNING: This is a special case for multi-class computation, where different labels are processed together, `reverse_one_hot` will need mapping information to work correctly"
            )
            temp_class_list = value
            # we don't need the brackets
            temp_class_list = temp_class_list.replace("[", "")
            temp_class_list = temp_class_list.replace("]", "")
            value = temp_class_list.split(",")
        else:
            try:
                value = eval(value)
                return value
            except Exception as e:
                ## todo: ensure logging captures assertion errors
                assert (
                    False
                ), f"Could not evaluate the `class_list` in `model`, Exception: {str(e)}, {traceback.format_exc()}"
                # logging.error(
                #     f"Could not evaluate the `class_list` in `model`, Exception: {str(e)}, {traceback.format_exc()}"
                # )
    return value


def validate_patch_size(patch_size, dimension) -> list:
    if isinstance(patch_size, int) or isinstance(patch_size, float):
        patch_size = [patch_size]
    if len(patch_size) == 1 and dimension is not None:
        actual_patch_size = []
        for _ in range(dimension):
            actual_patch_size.append(patch_size[0])
        patch_size = actual_patch_size
    if len(patch_size) == 2:  # 2d check
        # ensuring same size during torchio processing
        patch_size.append(1)
        if dimension is None:
            dimension = 2
    elif len(patch_size) == 3:  # 2d check
        if dimension is None:
            dimension = 3
    return [patch_size, dimension]


def validate_norm_type(norm_type, architecture):
    if norm_type is None or norm_type.lower() == "none":
        if "vgg" in architecture:
            raise ValueError(
                "Normalization type cannot be 'None' for non-VGG architectures"
            )
    else:
        print("WARNING: Initializing 'norm_type' as 'batch'", flush=True)
        norm_type = "batch"
    return norm_type


def validate_parallel_compute_command(value):
    parallel_compute_command = value
    parallel_compute_command = parallel_compute_command.replace(
        "'", ""
    )  # TODO: Check it again,should change from ' to `
    parallel_compute_command = parallel_compute_command.replace('"', "")
    value = parallel_compute_command
    return value

def validate_schedular(value, learning_rate):
    if isinstance(value, str):
        value = Scheduler(type=value)
        if value.step_size is None:
            value.step_size = learning_rate / 5.0
    return value