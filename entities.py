from pydantic import BaseModel
from typing import Union

class Input(BaseModel):
    humanMessage : str
    resetFlag: bool