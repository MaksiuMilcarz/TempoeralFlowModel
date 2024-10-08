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
        self.num_aircraft = 10
        self.num_demands = 40
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
        sample_fraction = 0.7
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
            current_time = aircraft.available_time
            current_location = aircraft.current_location
            flights_for_aircraft = []
            aircraft_route_complete = False

            while current_time < self.SCHEDULE_DURATION - self.RESERVED_RETURN_TIME and not aircraft_route_complete:
                # Find possible flights from the current location
                possible_flights = [(dest, self.flight_durations[(current_location, dest)])
                                    for (orig, dest) in self.possible_flights
                                    if orig == current_location]

                if not possible_flights:
                    break  # No further flights possible

                # Prioritize flights that can transport untransported demand
                demand_destinations = [
                    dest for dest in [dest for (dest, _) in possible_flights]
                    if any(demand.origin == current_location and demand.destination == dest and demand.untransported_weight > 0
                        for demand in self.demands)
                ]
                if demand_destinations:
                    # Filter possible_flights to those destinations with unmet demands
                    possible_flights = [(dest, dur) for (dest, dur) in possible_flights if dest in demand_destinations]

                # Sort possible flights by earliest demand ready time
                sorted_flights = sorted(possible_flights, key=lambda x: min(
                    (demand.demand_ready_time for demand in self.demands
                    if demand.origin == current_location and demand.destination == x[0] and demand.untransported_weight > 0),
                    default=self.SCHEDULE_DURATION
                ))

                if not sorted_flights:
                    # No flights can transport demands; schedule return to base
                    if current_location != aircraft.home_base:
                        return_flight_duration = self.flight_durations.get((current_location, aircraft.home_base))
                        if return_flight_duration:
                            departure_time = current_time + aircraft.turnaround_time
                            arrival_time = departure_time + return_flight_duration
                            if arrival_time <= self.SCHEDULE_DURATION:
                                return_flight = Flight(
                                    flight_id=flight_id_counter,
                                    origin=current_location,
                                    destination=aircraft.home_base,
                                    departure_time=departure_time,
                                    arrival_time=arrival_time,
                                    aircraft_id=aircraft.aircraft_id,
                                    capacity=aircraft.capacity
                                )
                                flights_for_aircraft.append(return_flight)
                                self.flights.append(return_flight)
                                flight_id_counter += 1
                    break  # Exit the while loop for this aircraft

                # Select the next flight based on sorted_flights
                destination, flight_time = sorted_flights[0]

                # Get the earliest demand_ready_time for this destination
                matching_demands = [
                    demand.demand_ready_time for demand in self.demands
                    if demand.origin == current_location and demand.destination == destination and demand.untransported_weight > 0
                ]

                if matching_demands:
                    earliest_ready_time = min(matching_demands)
                    departure_time = max(current_time + aircraft.turnaround_time, earliest_ready_time)
                else:
                    # No matching demands; assign departure_time as current_time + turnaround_time
                    departure_time = current_time + aircraft.turnaround_time

                arrival_time = departure_time + flight_time

                if arrival_time > self.SCHEDULE_DURATION - self.RESERVED_RETURN_TIME:
                    # Not enough time to perform this flight; schedule return to base
                    if current_location != aircraft.home_base:
                        return_flight_duration = self.flight_durations.get((current_location, aircraft.home_base))
                        if return_flight_duration:
                            departure_time = current_time + aircraft.turnaround_time
                            arrival_time = departure_time + return_flight_duration
                            if arrival_time <= self.SCHEDULE_DURATION:
                                return_flight = Flight(
                                    flight_id=flight_id_counter,
                                    origin=current_location,
                                    destination=aircraft.home_base,
                                    departure_time=departure_time,
                                    arrival_time=arrival_time,
                                    aircraft_id=aircraft.aircraft_id,
                                    capacity=aircraft.capacity
                                )
                                flights_for_aircraft.append(return_flight)
                                self.flights.append(return_flight)
                                flight_id_counter += 1
                    break  # Exit the while loop for this aircraft

                # Schedule this flight
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

                # Prevent infinite loops by limiting the number of flights per aircraft
                if len(flights_for_aircraft) >= 50:  # Adjust as necessary
                    break

            # Schedule return to home base if not already there
            if current_location != aircraft.home_base:
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

            # Assign the route to the aircraft
            aircraft.route = flights_for_aircraft

    def assign_demands_to_flights(self):
        # Build a flight graph
        flight_graph = defaultdict(list)
        for flight in self.flights:
            flight_graph[flight.origin].append(flight)

        # Sort demands by ready time
        demands_sorted = sorted(self.demands, key=lambda d: d.demand_ready_time)

        for demand in demands_sorted:
            while demand.untransported_weight > 0:
                path = self.find_earliest_path(demand, flight_graph)
                if path:
                    # Assign demand to the flights in the path
                    min_available_capacity = min(
                        flight.capacity - sum(w for (d, w) in flight.demands_assigned) for flight in path
                    )
                    weight_to_assign = min(demand.untransported_weight, min_available_capacity)

                    if weight_to_assign > 0:
                        for flight in path:
                            flight.demands_assigned.append((demand, weight_to_assign))
                        demand.untransported_weight -= weight_to_assign
                        # Record the demand's path
                        arrival_time = path[-1].arrival_time
                        self.demand_paths[demand.demand_id].append((path, weight_to_assign, arrival_time))
                    else:
                        break  # No capacity available
                else:
                    break  # No path found

    def find_earliest_path(self, demand, flight_graph):
        import heapq
        max_path_length = self.max_path_length

        heap = []
        counter = itertools.count()

        # Start with flights departing from the demand's origin after the demand's ready time
        for flight in flight_graph.get(demand.origin, []):
            if flight.departure_time >= demand.demand_ready_time and \
               flight.capacity - sum(w for (d, w) in flight.demands_assigned) > 0:
                heapq.heappush(heap, PrioritizedItem(flight.arrival_time, next(counter), [flight]))

        visited = set()

        while heap:
            item = heapq.heappop(heap)
            path = item.item
            last_flight = path[-1]

            if last_flight.destination == demand.destination:
                return path  # Found a path

            if len(path) >= max_path_length:
                continue

            visited_key = (last_flight.destination, last_flight.arrival_time)
            if visited_key in visited:
                continue
            visited.add(visited_key)

            for next_flight in flight_graph.get(last_flight.destination, []):
                if next_flight.departure_time >= last_flight.arrival_time + last_flight.aircraft_turnaround_time and \
                   next_flight.capacity - sum(w for (d, w) in next_flight.demands_assigned) > 0:
                    new_path = path + [next_flight]
                    heapq.heappush(heap, PrioritizedItem(next_flight.arrival_time, next(counter), new_path))

        return None  # No path found
     
     
     
     
     
     
     
     
     
     
     
     
     
     
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
        
        t0 = time.time()
        self.assign_demands_to_flights()
        function_times['assign_demands_to_flights'] = time.time() - t0

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
        for demand_id, paths in self.demand_paths.items():
            demand = next(d for d in self.demands if d.demand_id == demand_id)
            print(f"Demand {demand_id} from {demand.origin} to {demand.destination}:")
            for path, weight, arrival_time in paths:
                flight_ids = [flight.flight_id for flight in path]
                print(f"  Path: {flight_ids}, Weight: {weight}, Arrival Time: {arrival_time}")
                transported_weight += weight
                delivery_times.append(arrival_time - demand.demand_ready_time)
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
        print(f"capacity-to-demand ratio: {cap_demand_ratio:.2f}")
        print(f"Unmet Demand Percentage: {unmet_demand_percentage:.2f}%")
        print(f"Average Delivery Time: {average_delivery_time:.2f} minutes")
        print(f"Number of Unique Flights: {unique_flights}")