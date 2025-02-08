from typing import Union
from pydantic import BaseModel, model_validator, Field, AfterValidator, ConfigDict

from GANDLF.Configuration.default_parameters import DefaultParameters
from GANDLF.Configuration.scheduler_parameters import Scheduler
from GANDLF.config_manager import version_check
from importlib.metadata import version
from typing_extensions import Self, Literal, Annotated, Optional
from GANDLF.Configuration.validators import *
from GANDLF.Configuration.model_parameters import Model


class Version(BaseModel):  # TODO: Maybe should be to another folder
    minimum: str
    maximum: str

    @model_validator(mode="after")
    def validate_version(self) -> Self:
        if version_check(self.model_dump(), version_to_check=version("GANDLF")):
            return self


class NestedTraining(BaseModel):  # TODO: Maybe should be in another folder
    stratified: bool = Field(
        default=False,
        description="this will perform stratified k-fold cross-validation but only with offline data splitting",
    )
    testing: int = Field(
        default=-5,
        description="this controls the number of testing data folds for final model evaluation; [NOT recommended] to disable this, use '1'",
    )
    validation: int = Field(
        default=-5,
        description="this controls the number of validation data folds to be used for model *selection* during training (not used for back-propagation)",
    )
    proportional: bool = Field(default=None)

    @model_validator(mode="after")
    def validate_nested_training(self) -> Self:
        if self.proportional is not None:
            self.stratified = self.proportional
        return self





class UserDefinedParameters(DefaultParameters):
    version: Version = Field(
        default=Version(minimum=version("GANDLF"), maximum=version("GANDLF")),
        description="Whether weighted loss is to be used or not.",
    )
    patch_size: Union[list[Union[int, float]], int, float] = Field(
        description="Patch size."
    )
    model: Model = Field(..., description="The model to use. ")
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
    nested_training: NestedTraining = Field(description="Nested training.")
    parallel_compute_command: str = Field(
        default="", description="Parallel compute command."
    )
    scheduler: Union[str, Scheduler] = Field(description="Scheduler.")

    # Validators
    @model_validator(mode="after")
    def validate(self) -> Self:
        # valiadate the patch_size
        self.patch_size, self.model.dimension = validate_patch_size(
            self.patch_size, self.model.dimension
        )
        # validate the parallel_compute_command
        self.parallel_compute_command = validate_parallel_compute_command(
            self.parallel_compute_command
        )
        #validate scheduler
        self.scheduler = validate_schedular(self.scheduler, self.learning_rate)


        return self
