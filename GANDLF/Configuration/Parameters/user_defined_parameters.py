from typing import Union
from pydantic import BaseModel, model_validator, Field, AfterValidator
from GANDLF.Configuration.Parameters.default_parameters import DefaultParameters
from GANDLF.Configuration.Parameters.nested_training_parameters import NestedTraining
from GANDLF.Configuration.Parameters.optimizer_parameters import OptimizerConfig
from GANDLF.Configuration.Parameters.patch_sampler import PatchSamplerConfig
from GANDLF.Configuration.Parameters.scheduler_parameters import SchedulerConfig
from GANDLF.utils import version_check
from importlib.metadata import version
from typing_extensions import Self, Literal, Annotated
from GANDLF.Configuration.Parameters.validators import (
    validate_schedular,
    validate_optimizer,
    validate_loss_function,
    validate_metrics,
    validate_data_preprocessing,
    validate_patch_size,
    validate_parallel_compute_command,
    validate_patch_sampler,
    validate_data_augmentation,
    validate_data_postprocessing_after_reverse_one_hot_encoding,
    validate_differential_privacy,
)
from GANDLF.Configuration.Parameters.model_parameters import ModelConfig


class Version(BaseModel):  # TODO: Maybe should be to another folder
    minimum: str
    maximum: str

    @model_validator(mode="after")
    def validate_version(self) -> Self:
        if version_check(self.model_dump(), version_to_check=version("GANDLF")):
            return self


class InferenceMechanismConfig(BaseModel):
    grid_aggregator_overlap: Literal["crop", "average"] = Field(default="crop")
    patch_overlap: int = Field(default=0)


class UserDefinedParameters(DefaultParameters):
    version: Version = Field(
        default=Version(minimum=version("GANDLF"), maximum=version("GANDLF")),
        description="GANDLF version",
    )
    patch_size: Union[list[Union[int, float]], int, float] = Field(
        description="Patch size."
    )
    model: ModelConfig = Field(..., description="The model to use. ")
    modality: Literal["rad", "histo", "path"] = Field(description="Modality.")
    loss_function: Annotated[
        Union[dict, str],
        Field(description="Loss function."),
        AfterValidator(validate_loss_function),
    ]
    metrics: Annotated[
        Union[dict, list[Union[str, dict, set]]],
        Field(description="Metrics."),
        AfterValidator(validate_metrics),
    ]
    nested_training: NestedTraining = Field(description="Nested training.")
    parallel_compute_command: str = Field(
        default="", description="Parallel compute command."
    )
    scheduler: Union[str, SchedulerConfig] = Field(
        description="Scheduler.", default=SchedulerConfig(type="triangle_modified")
    )
    optimizer: Union[str, OptimizerConfig] = Field(
        description="Optimizer.", default=OptimizerConfig(type="adam")
    )  # TODO: Check it again for (opt)
    patch_sampler: Union[str, PatchSamplerConfig] = Field(
        description="Patch sampler.", default=PatchSamplerConfig()
    )
    inference_mechanism: InferenceMechanismConfig = Field(
        description="Inference mechanism.", default=InferenceMechanismConfig()
    )
    data_postprocessing_after_reverse_one_hot_encoding: dict = Field(
        description="data_postprocessing_after_reverse_one_hot_encoding.", default={}
    )
    differential_privacy: Union[bool, dict] = Field(
        description="Differential privacy.", default=None
    )
    # TODO: It should be defined with a better way (using a BaseModel class)
    data_preprocessing: Annotated[
        dict,
        Field(description="Data preprocessing."),
        AfterValidator(validate_data_preprocessing),
    ] = {}
    # TODO: It should be defined with a better way (using a BaseModel class)
    data_augmentation: Annotated[dict, Field(description="Data augmentation.")] = {}

    # Validators
    @model_validator(mode="after")
    def validate(self) -> Self:
        # validate the patch_size
        self.patch_size, self.model.dimension = validate_patch_size(
            self.patch_size, self.model.dimension
        )
        # validate the parallel_compute_command
        self.parallel_compute_command = validate_parallel_compute_command(
            self.parallel_compute_command
        )
        # validate scheduler
        self.scheduler = validate_schedular(self.scheduler, self.learning_rate)
        # validate optimizer
        self.optimizer = validate_optimizer(self.optimizer)
        # validate patch_sampler
        self.patch_sampler = validate_patch_sampler(self.patch_sampler)
        # validate_data_augmentation
        self.data_augmentation = validate_data_augmentation(
            self.data_augmentation, self.patch_size
        )
        # validate data_postprocessing_after_reverse_one_hot_encoding
        (
            self.data_postprocessing_after_reverse_one_hot_encoding,
            self.data_postprocessing,
        ) = validate_data_postprocessing_after_reverse_one_hot_encoding(
            self.data_postprocessing_after_reverse_one_hot_encoding,
            self.data_postprocessing,
        )
        # validate differential_privacy
        self.differential_privacy = validate_differential_privacy(
            self.differential_privacy, self.batch_size
        )

        return self
