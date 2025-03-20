import traceback
from typing import Union
import yaml
from pydantic import ValidationError

from GANDLF.Configuration.parameters_config import Parameters
from GANDLF.Configuration.exclude_parameters import exclude_parameters
from GANDLF.Configuration.utils import handle_configuration_errors


def _parseConfig(
    config_file_path: Union[str, dict], version_check_flag: bool = True
) -> None:
    """
    This function parses the configuration file and returns a dictionary of parameters.

    Args:
        config_file_path (Union[str, dict]): The filename of the configuration file.
        version_check_flag (bool, optional): Whether to check the version in configuration file. Defaults to True.

    Returns:
        dict: The parameter dictionary.
    """
    params = config_file_path
    if not isinstance(config_file_path, dict):
        params = yaml.safe_load(open(config_file_path, "r"))

    return params


def ConfigManager(
    config_file_path: Union[str, dict], version_check_flag: bool = True
) -> dict:
    """
    This function parses the configuration file and returns a dictionary of parameters.

    Args:
        config_file_path (Union[str, dict]): The filename of the configuration file.
        version_check_flag (bool, optional): Whether to check the version in configuration file. Defaults to True.

    Returns:
        dict: The parameter dictionary.
    """
    try:
        parameters_config = Parameters(
            **_parseConfig(config_file_path, version_check_flag)
        )
        parameters = parameters_config.model_dump(
            exclude={
                field
                for field in exclude_parameters
                if getattr(parameters_config, field) is None
            }
        )
        return parameters
    except ValidationError as e:
        handle_configuration_errors(e)

    except Exception as e:
        ## todo: ensure logging captures assertion errors
        assert (
            False
        ), f"Config parsing failed: {config_file_path=}, {version_check_flag=}, Exception: {str(e)}, {traceback.format_exc()}"
        # logging.error(
        #     f"gandlf config parsing failed: {config_file_path=}, {version_check_flag=}, Exception: {str(e)}, {traceback.format_exc()}"
        # )
        # raise
