import random
import string
from collections import defaultdict
from data_structs import Aircraft, Airport, Demand, Flight
from data_structs import Vertex, FlightEdge    

def load_data(state):
    # Generate airport codes (e.g., 'A', 'B', ..., 'L')
    airport_codes = list(string.ascii_uppercase[:state.num_airports])
    # Create vertices for each airport
    state.airports = {code: Vertex(code) for code in airport_codes}

    # Generate aircraft
    state.aircraft = [Aircraft(aircraft_id=i+1, capacity=200)
                      for i in range(state.num_aircraft)]

    # Generate random demands
    state.demands = []
    for i in range(1, state.num_demands + 1):
        origin = random.choice(airport_codes)
        destination = random.choice(airport_codes)
        while destination == origin:
            destination = random.choice(airport_codes)
        total_weight = random.randint(10, 100)
        demand_ready_time = random.choice(state.demand_ready_times)
        demand = Demand(
            demand_id=i,
            origin=origin,
            destination=destination,
            total_weight=total_weight,
            demand_ready_time=demand_ready_time
        )
        state.demands.append(demand)
        
def assign_aircraft_bases(state):
    # Calculate total demand weight per airport
    demand_weights = defaultdict(float)
    for demand in state.demands:
        demand_weights[demand.origin] += demand.total_weight

    # Sort airports by total demand weight in descending order
    airports_by_demand = sorted(state.airports.keys(),
                                key=lambda x: -demand_weights[x])

    # Assign aircraft to airports based on demand
    aircraft_per_airport = len(state.aircraft) // len(state.airports)
    extra_aircraft = len(state.aircraft) % len(state.airports)
    aircraft_index = 0

    for airport_code in airports_by_demand:
        num_aircraft = aircraft_per_airport + (1 if extra_aircraft > 0 else 0)
        if extra_aircraft > 0:
            extra_aircraft -= 1

        for _ in range(num_aircraft):
            if aircraft_index < len(state.aircraft):
                state.aircraft[aircraft_index].home_base = airport_code
                state.aircraft[aircraft_index].current_location = airport_code
                aircraft_index += 1
            else:
                break
    
def generate_possible_flights(state):
    # Generate all possible routes between different airports (excluding A->A)
    airport_codes = list(state.airports.keys())
    all_possible_routes = [(a, b) for a in airport_codes for b in airport_codes if a != b]

    # Limit the possible flights to a subset (e.g., 50%) if desired
    random.shuffle(all_possible_routes)
    subset_size = int(len(all_possible_routes) * 0.5)
    subset_possible_routes = all_possible_routes[:subset_size]

    # Store possible flights without scheduling them
    state.possible_flights = []
    for origin_code, destination_code in subset_possible_routes:
        origin_vertex = state.airports[origin_code]
        destination_vertex = state.airports[destination_code]
        flight_edge = FlightEdge(origin_vertex, destination_vertex)
        state.possible_flights.append(flight_edge)
