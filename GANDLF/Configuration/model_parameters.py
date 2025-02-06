from pydantic import (
    BaseModel,
    model_validator,
    Field, AliasChoices, field_validator, AfterValidator, ConfigDict
)

from typing_extensions import Self, Literal, Annotated, Optional
from typing import Union
from GANDLF.Configuration.validators import validate_class_list

class Model(BaseModel):
    model_config = ConfigDict(extra='allow')
    dimension: Optional[int] = Field(description="Dimension.")
    architecture: Union[str,dict] = Field(description="Architecture.")
    final_layer: str = Field(description="Final layer.")
    norm_type: str = Field(description="Normalization type.",default= None) # TODO: check it
    base_filters: Optional[int] = Field(description="Base filters.", default= None, validate_default= True) # default is 32
    class_list: Union[list, str] = Field(default=[],description="Class list." ) # TODO: check it for  class_list: '[0,1||2||3,1||4,4]'
    num_channels: Optional[int] = Field(description="Number of channels.", validation_alias=AliasChoices('num_channels', "n_channels","channels","model_channels" )) # TODO: check it
    type: Optional[str] = Field(description="Type of model.",default= "torch")
    data_type: str = Field(description="Data type.",default= "FP32")
    save_at_every_epoch: bool = Field(default=False,description="Save at every epoch.")
    amp: bool = Field(default = False,description="Amplifier.")
    ignore_label_validation: int = Field(default=None ,description="Ignore label validation.") #TODO:  To check it
    print_summary:bool = Field(default=True ,description="Print summary.")
    batch_norm: int = Field(default=None,deprecated="batch_norm is no longer supported, please use 'norm_type' in 'model' instead.")


    @model_validator(mode="after")
    def model_validate(self) -> Self:
        self.class_list = validate_class_list(self.class_list)
        if self.amp is False:
            print("NOT using Mixed Precision Training")

        if self.save_at_every_epoch:
            print("WARNING: 'save_at_every_epoch' will result in TREMENDOUS storage usage; use at your own risk.") # TODO: It is better to use logging.warning

        if self.base_filters is None:
            self.base_filters = 32
            print("Using default 'base_filters' in 'model': ", self.base_filters)
        return self