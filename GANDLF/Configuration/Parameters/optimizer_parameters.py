from pydantic import BaseModel, Field
from typing_extensions import Literal

OPTIMIZER_OPTIONS = Literal[
    "sgd",
    "asgd",
    "adam",
    "adamw",
    "adamax",
    "sparseadam",
    "rprop",
    "adadelta",
    "adagrad",
    "rmsprop",
]


class Optimizer(BaseModel):
    type: OPTIMIZER_OPTIONS = Field(description="Type of optimizer to use")
