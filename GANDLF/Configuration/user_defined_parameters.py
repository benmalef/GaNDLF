from typing import Union
from pydantic import BaseModel, model_validator, Field
from GANDLF.config_manager import version_check
from importlib.metadata import version
from typing_extensions import Self


class Version(BaseModel):
    minimum: str
    maximum: str

    @model_validator(mode="after")
    def validate_version(self) -> Self:
        if version_check(self.model_dump(), version_to_check=version("GANDLF")):
            return self


class PatchSize(BaseModel):
    patch_size: list[Union[int, float]]

    @model_validator(mode="after")
    def validate_patch_size(self) -> Self:
        if isinstance(self.patch_size, int) or isinstance(self.patch_size, float):
            self.patch_size = [self.patch_size]
            return self


class UserDefinedParameters(BaseModel):
    version: Version = Field(
        default=Version(minimum=version("GANDLF"), maximum=version("GANDLF")),
        description="Whether weighted loss is to be used or not.",
    )
    patch_size: PatchSize = Field(..., description="Patch size.")
