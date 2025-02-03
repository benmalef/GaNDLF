from typing import Union
from pydantic import BaseModel, model_validator, Field, field_validator
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


class Model(BaseModel):
    dimension: Union[int, None] = Field(description="Dimension.", default=None)


class UserDefinedParameters(BaseModel):
    version: Version = Field(
        default=Version(minimum=version("GANDLF"), maximum=version("GANDLF")),
        description="Whether weighted loss is to be used or not.",
    )
    patch_size: Union[list[Union[int, float]], int, float] = Field(
        ..., description="Patch size."
    )
    model: Model = Field(..., description="Model.")

    # Validators
    @model_validator(mode="after")
    def validate_patch_size(self) -> Self:
        # Validation for patch_size
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
