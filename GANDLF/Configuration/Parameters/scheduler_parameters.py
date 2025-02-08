from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Literal


TYPE_OPTIONS = Literal[
    "triangle",
    "triangle_modified",
    "exp",
    "step",
    "reduce-on-plateau",
    "cosineannealing",
    "triangular",
    "triangular2",
    "exp_range",
]

# It allows extra parameters
class Scheduler(BaseModel):
    model_config = ConfigDict(extra= "allow")
    type: TYPE_OPTIONS = Field(
        description="triangle/triangle_modified use LambdaLR but triangular/triangular2/exp_range uses CyclicLR"
    )
    # min_lr: 0.00001, #TODO: this should be defined ??
    # max_lr: 1, #TODO: this should be defined ??
    step_size: float = Field(description="step_size", default=None)
