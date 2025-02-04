from typing import Union
from pydantic import (
    BaseModel,
    model_validator,
    Field,
    field_validator,
    AfterValidator,
    BeforeValidator,
)
from GANDLF.config_manager import version_check
from importlib.metadata import version
from typing_extensions import Self, Literal, Annotated, Optional
from GANDLF.Configuration.validators import *
from GANDLF.Configuration.model_parameters import Model


class Version(BaseModel):
    minimum: str
    maximum: str

    @model_validator(mode="after")
    def validate_version(self) -> Self:
        if version_check(self.model_dump(), version_to_check=version("GANDLF")):
            return self



class UserDefinedParameters(BaseModel):
    version: Version = Field(
        default=Version(minimum=version("GANDLF"), maximum=version("GANDLF")),
        description="Whether weighted loss is to be used or not.",
    )
    patch_size: Union[list[Union[int, float]], int, float] = Field(
        description="Patch size."
    )
    model: Annotated[Model,Field(description="Model.")]
    modality: Literal["rad", "histo", "path"] = Field(description="Modality.")
    loss_function: Annotated[
        Union[dict, str],
        Field(description="Loss function."),
        AfterValidator(validate_loss_function),
    ]
    metrics: Annotated[
        Union[dict, list[str]],
        Field(description="Metrics."),
        AfterValidator(validate_metrics),
    ]

    # Validators
    @model_validator(mode="after")
    def validate(self) -> Self:
        return validate_patch(self) # check if it is the right approach
