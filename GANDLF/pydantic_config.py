from pydantic import BaseModel
from enum import StrEnum
from GANDLF.models.modelBase import ModelBase


class Version(ModeBase):
    minimum:str
    maximum:str

class Batch_size(ModeBase):
    batch_size:int|flaot

class ModalityEnums(StrEnum):
    rad="rad"
    histo="histo"
    path="path"




class Parameters(ModeBase):
    version:Version
    batch_size:Batch_size
    modality: ModalityEnums

