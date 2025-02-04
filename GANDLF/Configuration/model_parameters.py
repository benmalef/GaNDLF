from pydantic import (
    BaseModel,
    model_validator,
    Field,
    field_validator,
    AfterValidator,
    BeforeValidator,
)

from typing_extensions import Self, Literal, Annotated, Optional
from typing import Union

class Model(BaseModel):
    dimension: Union[int, None] = Field(description="Dimension.", default=None)
    architecture: Union[str,dict] = Field(description="Architecture.")
    final_layer: str = Field(description="Final layer.")
    norm_type: Optional[str] = Field(description="Normalization type.")
    base_filters: Optional[int] = Field(description="Base filters.")
    class_list: Optional[list] = Field(default=[],description="Class list.")
    num_channels: Optional[int] = Field(description="Number of channels.")
    type: Optional[str] = Field(description="Type of model.")
    data_type: Optional[str] = Field(description="Data type.")
    save_at_every_epoch: bool = Field(default=False,description="Save at every epoch.")
    amp: bool = Field(default = False,description="Amplifier.")
    ignore_label_validation: None = Field(default=None ,description="Ignore label validation.") # To check it
    print_summary:bool = Field(default=True ,description="Print summary.")

    @model_validator(mode="after")
    def validate_model(self):
        if self.amp is False:
            print("NOT using Mixed Precision Training")


