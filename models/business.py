from pydantic import BaseModel

class Business(BaseModel):
    name: str
    address: str
    phone: str
