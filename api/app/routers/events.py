from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class EventIn(BaseModel):
    id: str
    schema: str
    time: str

class EventOut(BaseModel):
    id: str
    received: bool = True

@router.post("", response_model=EventOut, summary="Ingest an event (stub)")
def create_event(evt: EventIn):
    # TODO: wire M04_event_builder output -> persistence -> enqueue framing
    return EventOut(id=evt.id)
