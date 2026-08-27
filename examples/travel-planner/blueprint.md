# Vision
Create a travel planner that can help users to create itineraries, book hotels and flights based on user inputs.
If asked to plan multi city trips then it should spin multiple travel planners.
After each itinerary research, the full itinerary must be presented to the human for review and
approval BEFORE any booking happens. Multi-city trips must properly plan all dates including
travel/transit time between cities.

# Constraints
Booking a new flight and hotel MUST require human approval (requires_approval gate — hard pause).
The planner MUST present each itinerary to the user and wait for their go-ahead before booking.
Multi-city trips MUST account for travel time between destinations when computing dates.

# Tech stack
Use Sqlite
Use api_key authentication
No rag required
Use gemini 3.5 flash model as primary model
Use name default agent name as planner agent
