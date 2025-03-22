from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Literal

from GANDLF.schedulers import global_schedulers_dict

TYPE_OPTIONS = Literal[tuple(global_schedulers_dict.keys())]

class base_triangle_config(BaseModel):
    min_lr: float = Field(default= (10 ** -3))
    max_lr: float = Field(default=1)



# It allows extra parameters
class SchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: TYPE_OPTIONS = Field(
        description="triangle/triangle_modified use LambdaLR but triangular/triangular2/exp_range uses CyclicLR",
        default = "triangle"
    )
    step_size: float = Field(description="step_size", default=None)