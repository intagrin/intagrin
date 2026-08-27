"""Structured result contracts for IntaGrin's spawns.result_schema — validated at return_to_creator
time, not hand-parsed from free text."""

from pydantic import BaseModel


class HotelOption(BaseModel):
    """A single hotel option surfaced by research."""

    name: str
    price_per_night: str
    rating: float


class FlightOption(BaseModel):
    """A single flight option surfaced by research."""

    airline: str
    flight_no: str
    price: str
    duration: str


class ItineraryResult(BaseModel):
    """What a spawned itinerary-research specialist hands back to planner. Includes the full
    daily schedule, hotel choices, and flight choices so the planner can present a complete
    itinerary preview for human approval before any booking happens."""

    destination: str
    duration_days: int
    top_highlights: list[str] = []
    daily_schedule: list[str] = []
    hotel_options: list[HotelOption] = []
    flight_options: list[FlightOption] = []
    recommendation_note: str = ""
