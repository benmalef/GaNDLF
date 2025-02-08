from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Literal


class Scheduler(BaseModel):
    model_config = ConfigDict(
        extra="allow"
    )
    type: Literal[
        "triangle",
        "triangle_modified",
        "exp",
        "step",
        "reduce-on-plateau",
        "cosineannealing",
        "triangular",
        "triangular2",
        "exp_range",
    ] = Field(description="triangle/triangle_modified use LambdaLR but triangular/triangular2/exp_range uses CyclicLR",)
    # min_lr: 0.00001, #TODO: this should be defined ??
    # max_lr: 1, #TODO: this should be defined ??
    step_size: float = Field(description="step_size",default=None)
