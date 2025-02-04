
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


def validate_an_example(value, patch) -> dict:
    return value


def validate_patch(self):
    if isinstance(self.patch_size, int) or isinstance(self.patch_size, float):
        self.patch_size = [self.patch_size]
    if len(self.patch_size) == 1 and self.model.dimension is not None:
        actual_patch_size = []
        for _ in range(self.model.dimension):
            actual_patch_size.append(self.patch_size[0])
        self.patch_size = actual_patch_size
    if len(self.patch_size) == 2:  # 2d check
        # ensuring same size during torchio processing
        self.patch_size.append(1)
        if self.model.dimension is None:
            self.model.dimension = 2
    elif len(self.patch_size) == 3:  # 2d check
        if self.model.dimension is None:
            self.model.dimension = 3

    return self
