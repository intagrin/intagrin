"""Travel planning and booking tools for IntaGrin TravelPlanner agent."""

import json
from typing import Any, Dict, List, Optional

from .user_profile_store import get_profile

_PROFILE_FORM_URL = "http://localhost:8600"


def _require_authorized_profile() -> Optional[str]:
    """Returns an error string if the traveler hasn't completed the out-of-band profile/payment
    form yet, or None if booking may proceed. The email itself is never read into a variable a
    caller could accidentally surface — only the boolean authorization flag is inspected here."""
    profile = get_profile()
    if not profile or not profile.get("payment_authorized"):
        return (
            "Cannot book yet: the traveler hasn't completed their profile and payment "
            f"authorization. Ask them to visit {_PROFILE_FORM_URL} to enter their email and "
            "authorize payment — do not ask for their email in this chat, and do not retry this "
            "booking until they confirm they've done it."
        )
    return None


DESTINATION_DATA: Dict[str, Dict[str, Any]] = {
    "paris": {
        "name": "Paris, France",
        "highlights": ["Eiffel Tower", "Louvre Museum", "Notre-Dame Cathedral", "Montmartre", "Seine River Cruise"],
        "daily_schedule": [
            "Day 1: Arrive in Paris, check-in, visit Eiffel Tower and evening Seine River Cruise.",
            "Day 2: Explore the Louvre Museum, stroll through Tuileries Garden and Champs-Élysées.",
            "Day 3: Visit Montmartre, Sacré-Cœur Basilica, and experience Parisian cafe culture.",
            "Day 4: Day trip to Palace of Versailles and local wine tasting.",
        ],
        "hotels": [
            {"name": "Hôtel Plaza Athénée", "price_per_night": "$850", "rating": 4.9},
            {"name": "Novotel Paris Les Halles", "price_per_night": "$220", "rating": 4.5},
            {"name": "CitizenM Paris Gare de Lyon", "price_per_night": "$160", "rating": 4.3},
        ],
        "flights": [
            {"airline": "Air France", "flight_no": "AF101", "price": "$650", "duration": "7h 45m"},
            {"airline": "Delta Air Lines", "flight_no": "DL402", "price": "$620", "duration": "8h 10m"},
        ],
    },
    "italy": {
        "name": "Italy (Rome, Florence & Venice)",
        "highlights": ["Colosseum & Roman Forum", "Vatican Museums", "Florence Duomo", "Venice Canals", "Amalfi Coast"],
        "daily_schedule": [
            "Day 1: Rome - Colosseum, Roman Forum, and Trevi Fountain.",
            "Day 2: Rome & Vatican - Vatican Museums, St. Peter's Basilica, and Trastevere dinner.",
            "Day 3: Florence - High-speed train to Florence, Uffizi Gallery, and Ponte Vecchio.",
            "Day 4: Venice - St. Mark's Basilica, Grand Canal gondola ride, and Rialto Bridge.",
        ],
        "hotels": [
            {"name": "Hotel Artemide Rome", "price_per_night": "$280", "rating": 4.8},
            {"name": "Grand Hotel Cavour Florence", "price_per_night": "$240", "rating": 4.6},
            {"name": "Hotel Danieli Venice", "price_per_night": "$750", "rating": 4.9},
        ],
        "flights": [
            {"airline": "ITA Airways", "flight_no": "AZ609", "price": "$710", "duration": "8h 30m"},
            {"airline": "Emirates", "flight_no": "EK205", "price": "$780", "duration": "9h 15m"},
        ],
    },
    "dubai": {
        "name": "Dubai, UAE",
        "highlights": ["Burj Khalifa", "Dubai Mall & Fountain Show", "Desert Safari & Dune Bashing", "Palm Jumeirah", "Dubai Marina"],
        "daily_schedule": [
            "Day 1: Arrive, visit Dubai Mall, Burj Khalifa observation deck, and Dubai Fountain.",
            "Day 2: Afternoon 4x4 Desert Safari, dune bashing, camel riding, and BBQ dinner under the stars.",
            "Day 3: Palm Jumeirah, Atlantis Aquaventure Waterpark, and Dubai Marina luxury yacht cruise.",
            "Day 4: Old Dubai walking tour, Gold and Spice Souks, and traditional abra ride across Dubai Creek.",
        ],
        "hotels": [
            {"name": "Burj Al Arab Jumeirah", "price_per_night": "$1500", "rating": 5.0},
            {"name": "Atlantis, The Palm", "price_per_night": "$450", "rating": 4.8},
            {"name": "Rove Downtown", "price_per_night": "$120", "rating": 4.4},
        ],
        "flights": [
            {"airline": "Emirates", "flight_no": "EK202", "price": "$890", "duration": "12h 30m"},
            {"airline": "Flydubai", "flight_no": "FZ145", "price": "$520", "duration": "13h 00m"},
        ],
    },
    "malaysia": {
        "name": "Malaysia (Kuala Lumpur & Penang)",
        "highlights": ["Petronas Twin Towers", "Batu Caves", "George Town Heritage", "Street Food Trails", "Langkawi Beaches"],
        "daily_schedule": [
            "Day 1: Kuala Lumpur - Petronas Twin Towers, KLCC Park, and Bukit Bintang nightlife.",
            "Day 2: Visit Batu Caves, Thean Hou Temple, and Jalan Alor street food tour.",
            "Day 3: Travel to Penang - Explore George Town UNESCO heritage streets and street art.",
            "Day 4: Penang Hill funicular, Kek Lok Si Temple, and Gurney Drive hawker center.",
        ],
        "hotels": [
            {"name": "Mandarin Oriental Kuala Lumpur", "price_per_night": "$220", "rating": 4.8},
            {"name": "Eastern & Oriental Hotel Penang", "price_per_night": "$190", "rating": 4.7},
            {"name": "The Chow Kit Kuala Lumpur", "price_per_night": "$75", "rating": 4.3},
        ],
        "flights": [
            {"airline": "Malaysia Airlines", "flight_no": "MH091", "price": "$680", "duration": "15h 20m"},
            {"airline": "Singapore Airlines", "flight_no": "SQ423", "price": "$730", "duration": "16h 00m"},
        ],
    },
    "singapore": {
        "name": "Singapore",
        "highlights": ["Gardens by the Bay", "Marina Bay Sands SkyPark", "Sentosa Island", "Chinatown & Little India", "Jewel Changi"],
        "daily_schedule": [
            "Day 1: Arrival at Jewel Changi, Marina Bay Sands SkyPark, and Gardens by the Bay light show.",
            "Day 2: Singapore Zoo / Night Safari and Singapore Botanic Gardens.",
            "Day 3: Sentosa Island - Universal Studios Singapore and Siloso Beach relaxation.",
            "Day 4: Cultural tour of Chinatown, Little India, Kampong Glam, and Lau Pa Sat hawker feast.",
        ],
        "hotels": [
            {"name": "Marina Bay Sands", "price_per_night": "$650", "rating": 4.8},
            {"name": "The Fullerton Hotel Singapore", "price_per_night": "$380", "rating": 4.7},
            {"name": "Hotel G Singapore", "price_per_night": "$140", "rating": 4.2},
        ],
        "flights": [
            {"airline": "Singapore Airlines", "flight_no": "SQ025", "price": "$820", "duration": "14h 50m"},
            {"airline": "Qatar Airways", "flight_no": "QR946", "price": "$760", "duration": "16h 15m"},
        ],
    },
    "thailand": {
        "name": "Thailand (Bangkok & Phuket)",
        "highlights": ["Grand Palace Bangkok", "Wat Arun", "Phuket Phi Phi Islands", "Floating Markets", "Thai Cooking Class"],
        "daily_schedule": [
            "Day 1: Bangkok - Grand Palace, Wat Pho (Reclining Buddha), and sunset at Wat Arun.",
            "Day 2: Damnoen Saduak Floating Market, Chatuchak Weekend Market, and Thai street food crawl.",
            "Day 3: Flight to Phuket - Patong Beach, Big Buddha viewpoint, and Old Phuket Town.",
            "Day 4: Island hopping speed boat tour to Phi Phi Islands and Maya Bay snorkeling.",
        ],
        "hotels": [
            {"name": "Banyan Tree Bangkok", "price_per_night": "$210", "rating": 4.7},
            {"name": "The Shore at Katathani Phuket", "price_per_night": "$420", "rating": 4.9},
            {"name": "Lub d Bangkok Siam", "price_per_night": "$60", "rating": 4.4},
        ],
        "flights": [
            {"airline": "Thai Airways", "flight_no": "TG601", "price": "$640", "duration": "14h 10m"},
            {"airline": "Cathay Pacific", "flight_no": "CX708", "price": "$590", "duration": "15h 30m"},
        ],
    },
}


def _match_destination(destination: str) -> Optional[Dict[str, Any]]:
    """Helper to match destination string against known sample destinations."""
    dest_lower = destination.strip().lower()
    # Normalize common spellings (e.g. malasia -> malaysia)
    if "malas" in dest_lower:
        dest_lower = "malaysia"
    for key, data in DESTINATION_DATA.items():
        if key in dest_lower or dest_lower in key:
            return data
    return None


def create_itinerary(
    destination: str,
    days: int = 3,
    interests: Optional[List[str]] = None,
) -> str:
    """Create a detailed travel itinerary for a specified destination.

    Args:
        destination: Name of the destination city or country (e.g., 'Paris', 'Italy', 'Dubai', 'Malaysia', 'Singapore', 'Thailand').
        days: Number of days for the itinerary (default: 3).
        interests: Optional list of traveler interests (e.g., ['culture', 'food', 'adventure', 'beaches']).

    Returns:
        A structured JSON string with the tailored itinerary and key attractions.
    """
    matched = _match_destination(destination)
    if matched:
        dest_name = matched["name"]
        highlights = matched["highlights"]
        raw_schedule = matched["daily_schedule"]
        schedule = [
            raw_schedule[i % len(raw_schedule)]
            for i in range(min(days, len(raw_schedule) if days <= len(raw_schedule) else days))
        ]
        hotels = matched.get("hotels", [])
        flights = matched.get("flights", [])
    else:
        dest_name = destination.title()
        highlights = ["City Center Tour", "Historic Landmarks", "Local Cuisine Discovery", "Cultural Museums"]
        schedule = [
            f"Day {i+1}: Explore top sights, local cuisine, and hidden gems in {dest_name}."
            for i in range(days)
        ]
        hotels = [{"name": f"Grand {dest_name} Hotel", "price_per_night": "$180", "rating": 4.5}]
        flights = [{"airline": "Global Airways", "flight_no": "GA550", "price": "$600", "duration": "8h 00m"}]

    itinerary_result = {
        "status": "success",
        "destination": dest_name,
        "duration_days": days,
        "interests_considered": interests or ["general sightseeing", "culture", "food"],
        "top_highlights": highlights,
        "itinerary": schedule,
        "hotel_options": hotels,
        "flight_options": flights,
        "recommendation_note": f"Customized {days}-day plan for {dest_name}. Ready to book flights and accommodations.",
    }
    return json.dumps(itinerary_result, indent=2)


def book_flight(
    destination: str,
    origin: str = "JFK",
    departure_date: str = "2026-09-01",
    return_date: Optional[str] = None,
    passengers: int = 1,
) -> str:
    """Book a flight to a selected destination. (Requires human approval).

    Args:
        destination: Destination city or country.
        origin: Origin airport code or city name (default: 'JFK').
        departure_date: Date of departure in YYYY-MM-DD format.
        return_date: Optional return date in YYYY-MM-DD format for round-trip flights.
        passengers: Number of passengers (default: 1).

    Returns:
        A structured JSON string confirming flight reservation details, or an error string if
        the traveler hasn't completed their profile/payment authorization yet.
    """
    denial = _require_authorized_profile()
    if denial:
        return denial

    matched = _match_destination(destination)
    if matched and matched.get("flights"):
        flight_info = matched["flights"][0]
        airline = flight_info["airline"]
        flight_no = flight_info["flight_no"]
        est_price = flight_info["price"]
    else:
        airline = "Global Airways"
        flight_no = "GA550"
        est_price = "$600"

    booking_confirmation = {
        "status": "confirmed",
        "booking_reference": f"FL-{destination.upper()[:3]}-84920",
        "airline": airline,
        "flight_number": flight_no,
        "origin": origin,
        "destination": matched["name"] if matched else destination.title(),
        "departure_date": departure_date,
        "return_date": return_date or "One-way",
        "passengers": passengers,
        "estimated_total_cost": est_price,
        "message": f"Flight successfully reserved with {airline} ({flight_no}) to {destination.title()} for {passengers} passenger(s).",
    }
    return json.dumps(booking_confirmation, indent=2)


def book_hotel(
    destination: str,
    hotel_name: Optional[str] = None,
    check_in: str = "2026-09-01",
    check_out: str = "2026-09-05",
    guests: int = 1,
) -> str:
    """Book hotel accommodations in a selected destination. (Requires human approval).

    Args:
        destination: Destination city or country.
        hotel_name: Optional specific hotel name. If omitted, a top recommended hotel is selected.
        check_in: Check-in date in YYYY-MM-DD format.
        check_out: Check-out date in YYYY-MM-DD format.
        guests: Number of guests (default: 1).

    Returns:
        A structured JSON string confirming hotel booking details, or an error string if the
        traveler hasn't completed their profile/payment authorization yet.
    """
    denial = _require_authorized_profile()
    if denial:
        return denial

    matched = _match_destination(destination)
    if matched and matched.get("hotels"):
        selected_hotel = matched["hotels"][0]
        if hotel_name:
            for h in matched["hotels"]:
                if hotel_name.lower() in h["name"].lower():
                    selected_hotel = h
                    break
        hotel_booked = selected_hotel["name"]
        price_rate = selected_hotel["price_per_night"]
        rating = selected_hotel["rating"]
    else:
        hotel_booked = hotel_name or f"Grand {destination.title()} Resort & Suites"
        price_rate = "$180"
        rating = 4.5

    booking_confirmation = {
        "status": "confirmed",
        "booking_reference": f"HT-{destination.upper()[:3]}-39104",
        "hotel_name": hotel_booked,
        "destination": matched["name"] if matched else destination.title(),
        "check_in": check_in,
        "check_out": check_out,
        "guests": guests,
        "rating": rating,
        "price_per_night": price_rate,
        "message": f"Hotel reservation confirmed at {hotel_booked} in {destination.title()} for {guests} guest(s).",
    }
    return json.dumps(booking_confirmation, indent=2)
