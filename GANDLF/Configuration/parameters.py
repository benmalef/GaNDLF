from pydantic import BaseModel, ConfigDict
from GANDLF.Configuration.default_parameters import DefaultParameters
from GANDLF.Configuration.user_defined_parameters import UserDefinedParameters


class ParametersConfiguration(BaseModel):
    model_config = ConfigDict()


class Parameters(ParametersConfiguration, DefaultParameters, UserDefinedParameters):
    pass
