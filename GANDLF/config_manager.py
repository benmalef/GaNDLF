# import logging
from typing import Optional, Union
from pydantic import ValidationError
import yaml


from GANDLF.Configuration.Parameters.parameters import Parameters


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
        parameters = Parameters(
            **_parseConfig(config_file_path, version_check_flag)
        ).model_dump()
        return parameters
    # except Exception as e:
    #     ## todo: ensure logging captures assertion errors
    #     assert (
    #         False
    #     ), f"Config parsing failed: {config_file_path=}, {version_check_flag=}, Exception: {str(e)}, {traceback.format_exc()}"
    #     # logging.error(
    #     #     f"gandlf config parsing failed: {config_file_path=}, {version_check_flag=}, Exception: {str(e)}, {traceback.format_exc()}"
    #     # )
    #     # raise
    except ValidationError as exc:
        print(exc.errors())
