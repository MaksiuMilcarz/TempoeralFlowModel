import heapq
from collections import defaultdict
import random
import time
import itertools

from data_structs import Aircraft, Airport, Demand, Flight, PrioritizedItem

class Scheduler:
    def __init__(self):
        self.demands = []
        self.aircraft = []
        self.airports = {}
        self.flights = []
        self.flight_schedule = {}
        self.demand_paths = defaultdict(list)  # Records the flight IDs and weights assigned to each demand

        self.SCHEDULE_DURATION = 2 * 24 * 60  # One week in minutes
        self.RESERVED_RETURN_TIME = 180  # Reserved time in minutes for aircraft to return to base
        self.max_path_length = 5  # Maximum number of flights in a path for a demand
        
        self.num_airports = 12
        self.num_aircraft = 8
        self.num_demands = 35
        self.demand_ready_times = [0, 720, 1440]

    def load_data(self):
        import string

        # Generate airport codes (e.g., 'A', 'B', ..., 'Z')
        airport_codes = list(string.ascii_uppercase[:self.num_airports])
        self.airports = {code: Airport(code) for code in airport_codes}

        # Generate aircraft
        self.aircraft = [Aircraft(aircraft_id=i+1, capacity=200) for i in range(self.num_aircraft)]

        # Generate random demands
        self.demands = []
        for i in range(1, self.num_demands + 1):
            origin = random.choice(airport_codes)
            destination = random.choice(airport_codes)
            while destination == origin:
                destination = random.choice(airport_codes)
            total_weight = random.randint(10, 100)  # Random weight between 10 and 100
            demand_ready_time = random.choice(self.demand_ready_times)
            demand = Demand(
                demand_id=i,
                origin=origin,
                destination=destination,
                total_weight=total_weight,
                demand_ready_time=demand_ready_time
            )
            self.demands.append(demand)

    def assign_aircraft_bases(self):
        # Calculate total demand weight per airport
        demand_weights = defaultdict(float)
        for demand in self.demands:
            demand_weights[demand.origin] += demand.total_weight

        # Sort airports by total demand weight in descending order
        airports_by_demand = sorted(self.airports.keys(), key=lambda x: -demand_weights[x])

        # Assign aircraft to airports based on demand
        aircraft_per_airport = len(self.aircraft) // len(self.airports)
        extra_aircraft = len(self.aircraft) % len(self.airports)
        aircraft_index = 0

        for airport_code in airports_by_demand:
            num_aircraft = aircraft_per_airport + (1 if extra_aircraft > 0 else 0)
            extra_aircraft -= 1 if extra_aircraft > 0 else 0

            for _ in range(num_aircraft):
                if aircraft_index < len(self.aircraft):
                    self.aircraft[aircraft_index].home_base = airport_code
                    self.aircraft[aircraft_index].current_location = airport_code
                    aircraft_index += 1
                else:
                    break

    def generate_possible_flights(self):
        airport_codes = list(self.airports.keys())
        all_possible_flights = [(a, b) for a in airport_codes for b in airport_codes if a != b]

        # Randomly sample flights (e.g., select 70% of all possible flights)
        sample_fraction = 0.6
        num_flights_to_select = int(len(all_possible_flights) * sample_fraction)
        selected_flights = random.sample(all_possible_flights, num_flights_to_select)

        # Define flight durations
        self.flight_durations = {}
        for (origin, destination) in selected_flights:
            duration = abs(ord(origin) - ord(destination)) * 30 + 60  # Example duration calculation
            self.flight_durations[(origin, destination)] = duration

        self.possible_flights = set(selected_flights)

    def generate_aircraft_routes(self):
        flight_id_counter = 1
        for aircraft in self.aircraft:
            current_time = 0  # Start time at 0
            current_location = aircraft.home_base
            flights_for_aircraft = []
            aircraft_route_complete = False

            while current_time < self.SCHEDULE_DURATION - self.RESERVED_RETURN_TIME and not aircraft_route_complete:
                # Identify possible next flights from current location
                possible_flights = [(dest, self.flight_durations[(current_location, dest)])
                                    for (orig, dest) in self.possible_flights
                                    if orig == current_location]

                if not possible_flights:
                    # No flights available from current location
                    break

                # Prioritize flights that can transport demands
                def flight_priority(flight):
                    dest, _ = flight
                    # Earliest demand ready time that can be picked up at current location
                    demands_at_current = [
                        demand for demand in self.demands
                        if demand.origin == current_location and demand.untransported_weight > 0 and demand.demand_ready_time <= current_time
                    ]
                    if demands_at_current:
                        return 0  # Highest priority
                    else:
                        return 1  # Lower priority

                possible_flights.sort(key=flight_priority)

                scheduled_flight = False
                for destination, flight_time in possible_flights:
                    departure_time = current_time + aircraft.turnaround_time
                    arrival_time = departure_time + flight_time

                    # Check if there's enough time to perform this flight and return home
                    return_flight_duration = self.flight_durations.get((destination, aircraft.home_base), float('inf'))
                    total_time = arrival_time + aircraft.turnaround_time + return_flight_duration
                    if total_time > self.SCHEDULE_DURATION:
                        continue  # Not enough time; try next flight

                    # Schedule the flight
                    flight = Flight(
                        flight_id=flight_id_counter,
                        origin=current_location,
                        destination=destination,
                        departure_time=departure_time,
                        arrival_time=arrival_time,
                        aircraft_id=aircraft.aircraft_id,
                        capacity=aircraft.capacity
                    )
                    flights_for_aircraft.append(flight)
                    self.flights.append(flight)
                    flight_id_counter += 1

                    # Update current location and time
                    current_location = destination
                    current_time = arrival_time
                    scheduled_flight = True

                    # Attempt to assign demands to this flight
                    self.assign_demands_to_flight(flight)

                    break  # Proceed to next iteration after scheduling a flight

                if not scheduled_flight:
                    # No suitable flights found; schedule return to base if needed
                    if current_location != aircraft.home_base:
                        flight_id_counter = self.schedule_return_flight(aircraft, current_location, current_time, flights_for_aircraft, flight_id_counter)
                    aircraft_route_complete = True
                    break

                # Prevent infinite loops by limiting the number of flights per aircraft
                if len(flights_for_aircraft) >= 50:  # Adjust as necessary
                    aircraft_route_complete = True
                    break

            # Schedule return to home base if not already there
            if current_location != aircraft.home_base:
                flight_id_counter = self.schedule_return_flight(aircraft, current_location, current_time, flights_for_aircraft, flight_id_counter)
            aircraft.route = flights_for_aircraft

    def assign_demands_to_flight(self, flight):
        available_capacity = flight.capacity

        # Find demands that can be picked up at the flight's origin
        demands_at_origin = [
            demand for demand in self.demands
            if demand.origin == flight.origin and demand.untransported_weight > 0 and demand.demand_ready_time <= flight.departure_time
        ]

        # Assign as many demands as possible
        for demand in demands_at_origin:
            weight_to_assign = min(available_capacity, demand.untransported_weight)
            if weight_to_assign > 0:
                flight.demands_assigned.append((demand, weight_to_assign))
                demand.untransported_weight -= weight_to_assign
                available_capacity -= weight_to_assign
                # Record the demand's path
                self.demand_paths[demand.demand_id].append(([flight], weight_to_assign, flight.arrival_time))
            if available_capacity <= 0:
                break  # Aircraft is full

    def schedule_return_flight(self, aircraft, current_location, current_time, flights_for_aircraft, flight_id_counter):
        return_flight_duration = self.flight_durations.get((current_location, aircraft.home_base))
        if return_flight_duration:
            departure_time = current_time + aircraft.turnaround_time
            arrival_time = departure_time + return_flight_duration
            if arrival_time <= self.SCHEDULE_DURATION:
                flight = Flight(
                    flight_id=flight_id_counter,
                    origin=current_location,
                    destination=aircraft.home_base,
                    departure_time=departure_time,
                    arrival_time=arrival_time,
                    aircraft_id=aircraft.aircraft_id,
                    capacity=aircraft.capacity
                )
                flights_for_aircraft.append(flight)
                self.flights.append(flight)
                flight_id_counter += 1
        return flight_id_counter

     
     
     
     
     
     
     
     
     
     
     
     
     
    def run(self):
        start_time = time.time()
        function_times = {}
        
        t0 = time.time()
        self.load_data()
        function_times['load_data'] = time.time() - t0
        
        t0 = time.time()
        self.assign_aircraft_bases()
        function_times['assign_aircraft_base'] = time.time() - t0
        
        t0 = time.time()
        self.generate_possible_flights()
        function_times['generate_possible_flights'] = time.time() - t0
        
        t0 = time.time()
        self.generate_aircraft_routes()
        function_times['generate_aircraft_routes'] = time.time() - t0

        self.display_results()
        
        total_time = time.time() - start_time
        function_times['total_time'] = total_time
        print("\nFunction Times:")
        for func_name, duration in function_times.items():
            print(f"{func_name} took {duration:.6f} seconds")
    
    def display_results(self):
        print("\nAircraft Routes:")
        for aircraft in self.aircraft:
            route_str = ' -> '.join(f"{flight.origin}->{flight.destination}({flight.flight_id})" for flight in aircraft.route)
            print(f"Aircraft {aircraft.aircraft_id} (Base {aircraft.home_base}): {route_str}")

        print("\nDemand Assignments:")
        total_demands = len(self.demands)
        total_weight = sum(d.total_weight for d in self.demands)
        transported_weight = 0
        delivery_times = []
        for demand in self.demands:
            paths = self.demand_paths.get(demand.demand_id, [])
            if paths:
                print(f"Demand {demand.demand_id} from {demand.origin} to {demand.destination}:")
                for path_flights, weight, arrival_time in paths:
                    flight_ids = [flight.flight_id for flight in path_flights]
                    print(f"  Path: {flight_ids}, Weight: {weight}, Arrival Time: {arrival_time}")
                    transported_weight += weight
                    delivery_times.append(arrival_time - demand.demand_ready_time)
            else:
                print(f"Demand {demand.demand_id} from {demand.origin} to {demand.destination}: Not transported")

        unmet_weight = sum(d.untransported_weight for d in self.demands)

        print("\nUntransported Demands:")
        for demand in self.demands:
            if demand.untransported_weight > 0:
                print(f"Demand {demand.demand_id} untransported weight: {demand.untransported_weight}")

        # Calculate statistics
        unmet_demand_percentage = (unmet_weight / total_weight) * 100 if total_weight > 0 else 0
        average_delivery_time = sum(delivery_times) / len(delivery_times) if delivery_times else 0
        unique_flights = len(set(flight.flight_id for flight in self.flights))
        total_capacity = sum(f.capacity for f in self.flights)
        cap_demand_ratio = total_capacity / total_weight if total_weight > 0 else 0

        print("\nStatistics:")
        print(f"Total Capacity: {total_capacity}")
        print(f"Total Demand Weight: {total_weight}")
        print(f"Total Transported Weight: {transported_weight}")
        print(f"Capacity-to-demand ratio: {cap_demand_ratio:.2f}")
        print(f"Unmet Demand Percentage: {unmet_demand_percentage:.2f}%")
        print(f"Average Delivery Time: {average_delivery_time:.2f} minutes")
        print(f"Number of Unique Flights: {unique_flights}")
