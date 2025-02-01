from pydantic import BaseModel,field_validator
from GANDLF.config_manager import  version_check
from importlib.metadata import version

#TODO: Define the paramaters from the config_manager

#TODO: Some parameters maybe can define as a seperated Model

class Version(BaseModel):
    minimum: str
    maximum: str

class UserDefinedParameters(BaseModel):
    version: Version

    @classmethod
    @field_validator('version', mode='after')
    def validate_version(cls, values: Version) -> Version:
        if version_check(values.model_dump(), version_to_check=version("GANDLF")):
            return values
