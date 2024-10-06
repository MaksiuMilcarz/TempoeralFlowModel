import heapq
from collections import defaultdict
import random
import time

from data_structs import Aircraft, Airport, Demand, Flight, PrioritizedItem

class Scheduler:
    def __init__(self):
        self.demands = []
        self.aircraft = []
        self.airports = {}
        self.flights = []
        self.flight_schedule = {}
        self.demand_paths = defaultdict(list)  # Records the flight IDs and weights assigned to each demand

        self.SCHEDULE_DURATION = 7 * 24 * 60  # One week in minutes
        self.RESERVED_RETURN_TIME = 180  # Reserved time in minutes for aircraft to return to base

    def load_data(self):
        # Define airports
        airport_codes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.airports = {code: Airport(code) for code in airport_codes}

        # Define demands with demand_ready_time
        self.demands = [
            Demand(1, 'A', 'D', 500, demand_ready_time=0),
            Demand(2, 'B', 'C', 700, demand_ready_time=120),
            Demand(3, 'C', 'E', 600, demand_ready_time=240),
            Demand(4, 'D', 'A', 400, demand_ready_time=360),
            Demand(5, 'E', 'B', 800, demand_ready_time=480),
            Demand(6, 'F', 'G', 550, demand_ready_time=600),
            Demand(7, 'G', 'H', 650, demand_ready_time=720),
            Demand(8, 'H', 'F', 750, demand_ready_time=840),
            Demand(9, 'A', 'E', 900, demand_ready_time=960),
            Demand(10,'B', 'D', 850, demand_ready_time=1080),
            Demand(11,'C', 'F', 950, demand_ready_time=1200),
            Demand(12,'D', 'G', 1000, demand_ready_time=1320),
            Demand(13,'E', 'H', 600, demand_ready_time=1440),
            Demand(14,'F', 'A', 500, demand_ready_time=1560),
            Demand(15,'G', 'B', 700, demand_ready_time=1680),
            Demand(16,'H', 'C', 800, demand_ready_time=1800),
            # Add more demands as needed
        ]

        # Define aircraft without home bases
        self.aircraft = [
            Aircraft(aircraft_id=i+1, capacity=100) for i in range(14)
        ]

        # Define fixed flight durations between airports (in minutes)
        self.flight_durations = {}
        for origin in airport_codes:
            for destination in airport_codes:
                if origin != destination:
                    # For simplicity, assign a fixed duration based on alphabetical distance
                    duration = abs(ord(origin) - ord(destination)) * 30 + 60  # Base time + distance factor
                    self.flight_durations[(origin, destination)] = duration

    def assign_aircraft_bases(self):
        # Assign aircraft to airports based on demands
        demand_counts = defaultdict(float)
        for demand in self.demands:
            demand_counts[demand.origin] += demand.total_weight

        # Sort airports by total demand weight
        airports_sorted = sorted(self.airports.keys(), key=lambda x: -demand_counts[x])

        # Assign aircraft to airports with highest demand
        aircraft_count = len(self.aircraft)
        airports_count = len(self.airports)
        aircraft_per_airport = aircraft_count // airports_count

        index = 0
        for airport_code in airports_sorted:
            for _ in range(aircraft_per_airport):
                if index < aircraft_count:
                    self.aircraft[index].home_base = airport_code
                    index +=1
                else:
                    break

        # Assign remaining aircraft to airports with highest remaining demand
        while index < aircraft_count:
            for airport_code in airports_sorted:
                if index < aircraft_count:
                    self.aircraft[index].home_base = airport_code
                    index +=1
                else:
                    break

        # If any aircraft have no home base, assign them randomly
        for aircraft in self.aircraft:
            if not aircraft.home_base:
                aircraft.home_base = random.choice(list(self.airports.keys()))

    def prune_flights(self):
        # Implement a function to prune unlikely flights
        # For example, limit flights to only between airports with demands
        # Or only generate flights that could potentially carry demands

        # Collect airports involved in demands
        demand_airports = set()
        for demand in self.demands:
            demand_airports.add(demand.origin)
            demand_airports.add(demand.destination)

        # Generate a set of potential flights between demand airports
        all_possible_flights = list()
        for origin in demand_airports:
            for destination in demand_airports:
                if origin != destination:
                    all_possible_flights.append((origin, destination))

        # Determine the fraction of flights to allow (e.g., 50%)
        fraction = 0.5  # Adjust this value as needed
        n = int(len(all_possible_flights) * fraction)

        # Randomly sample 'n' flights to include
        allowed_flights = set(random.sample(all_possible_flights, n))

        return allowed_flights

    def generate_flight_schedule(self):
        potential_flights = self.prune_flights()

        flight_id_counter = 1
        for aircraft in self.aircraft:
            current_time = aircraft.available_time
            current_location = aircraft.home_base
            flights_for_aircraft = []
            aircraft_route_complete = False

            while current_time < self.SCHEDULE_DURATION - self.RESERVED_RETURN_TIME and not aircraft_route_complete:
                # Find potential flights from the current location
                possible_destinations = [dest for (orig, dest) in potential_flights if orig == current_location]
                
                if not possible_destinations:
                    # No further flights; schedule return to base if not already there
                    if current_location != aircraft.home_base:
                        # Schedule return flight to home base
                        return_flight_time = self.flight_durations.get((current_location, aircraft.home_base), None)
                        if return_flight_time:
                            departure_time = current_time + aircraft.turnaround_time
                            arrival_time = departure_time + return_flight_time
                            if arrival_time <= self.SCHEDULE_DURATION:
                                return_flight = Flight(
                                    flight_id=flight_id_counter,
                                    origin=current_location,
                                    destination=aircraft.home_base,
                                    departure_time=departure_time,
                                    arrival_time=arrival_time,
                                    aircraft_id=aircraft.aircraft_id,
                                    capacity=aircraft.capacity,
                                    aircraft_turnaround_time=aircraft.turnaround_time
                                )
                                flights_for_aircraft.append(return_flight)
                                self.flights.append(return_flight)
                                self.flight_schedule[flight_id_counter] = return_flight
                                flight_id_counter += 1
                        aircraft_route_complete = True
                    break  # No possible destinations

                # Select the next destination based on potential demands
                next_flight = None
                for destination in possible_destinations:
                    flight_time = self.flight_durations[(current_location, destination)]
                    departure_time = current_time + aircraft.turnaround_time
                    arrival_time = departure_time + flight_time

                    if arrival_time > self.SCHEDULE_DURATION - self.RESERVED_RETURN_TIME:
                        continue  # Skip flights that exceed schedule duration

                    # Check if there is any demand from current_location to destination
                    has_demand = any(
                        demand.origin == current_location and demand.destination == destination and demand.untransported_weight > 0
                        for demand in self.demands
                    )

                    if has_demand:
                        # Schedule this flight
                        flight = Flight(
                            flight_id=flight_id_counter,
                            origin=current_location,
                            destination=destination,
                            departure_time=departure_time,
                            arrival_time=arrival_time,
                            aircraft_id=aircraft.aircraft_id,
                            capacity=aircraft.capacity,
                            aircraft_turnaround_time=aircraft.turnaround_time
                        )
                        next_flight = flight
                        break  # Found a suitable flight

                if next_flight:
                    flights_for_aircraft.append(next_flight)
                    self.flights.append(next_flight)
                    self.flight_schedule[flight_id_counter] = next_flight
                    flight_id_counter += 1

                    # Update current location and time
                    current_location = next_flight.destination
                    current_time = next_flight.arrival_time
                else:
                    # No suitable flights; schedule return to base if not already there
                    if current_location != aircraft.home_base:
                        # Schedule return flight to home base
                        return_flight_time = self.flight_durations.get((current_location, aircraft.home_base), None)
                        if return_flight_time:
                            departure_time = current_time + aircraft.turnaround_time
                            arrival_time = departure_time + return_flight_time
                            if arrival_time <= self.SCHEDULE_DURATION:
                                return_flight = Flight(
                                    flight_id=flight_id_counter,
                                    origin=current_location,
                                    destination=aircraft.home_base,
                                    departure_time=departure_time,
                                    arrival_time=arrival_time,
                                    aircraft_id=aircraft.aircraft_id,
                                    capacity=aircraft.capacity,
                                    aircraft_turnaround_time=aircraft.turnaround_time
                                )
                                flights_for_aircraft.append(return_flight)
                                self.flights.append(return_flight)
                                self.flight_schedule[flight_id_counter] = return_flight
                                flight_id_counter += 1
                        aircraft_route_complete = True
                    else:
                        # Aircraft is already at home base; end route
                        aircraft_route_complete = True

            # Assign the route to the aircraft
            aircraft.route = flights_for_aircraft

    def generate_initial_solution(self):
        # Initialize total untransported weight
        total_untransported_weight = sum(demand.untransported_weight for demand in self.demands)
        total_demand_weight = total_untransported_weight

        # Sort demands based on demand_ready_time
        demands_sorted = sorted(self.demands, key=lambda x: x.demand_ready_time)

        for demand in demands_sorted:
            # While there is untransported weight for this demand
            while demand.untransported_weight > 0:
                # Find feasible flights starting after demand_ready_time
                feasible_flights = [flight for flight in self.flights
                                    if flight.origin == demand.origin and
                                    flight.departure_time >= demand.demand_ready_time]

                # Use a priority queue to select flights with earliest arrival times
                flight_queue = []
                for flight in feasible_flights:
                    heapq.heappush(flight_queue, PrioritizedItem(flight.arrival_time, ([flight], flight.aircraft_id)))

                path_found = False
                visited = set()
                while flight_queue and not path_found:
                    item = heapq.heappop(flight_queue)
                    arrival_time = item.priority
                    path, aircraft_id = item.item
                    last_flight = path[-1]
                    visited.add((last_flight.destination, aircraft_id))

                    # Calculate handling time
                    if len(path) == 1:
                        # First flight, include handling time at origin
                        handling_time = 0  # Assuming demand handling time occurs during aircraft turnaround
                    else:
                        # Include demand handling time if changing aircraft
                        if last_flight.aircraft_id != path[-2].aircraft_id:
                            handling_time = demand.handling_time
                        else:
                            handling_time = 0  # No handling time if same aircraft

                    # Update arrival time to include handling time
                    arrival_time += handling_time

                    if last_flight.destination == demand.destination:
                        # Calculate how much weight can be transported
                        min_capacity = min(flight.capacity - sum(wt for (d, wt) in flight.demands_assigned) for flight in path)
                        weight_to_transport = min(demand.untransported_weight, min_capacity)

                        if weight_to_transport > 0:
                            # Assign partial demand to flights in the path
                            for flight in path:
                                flight.demands_assigned.append((demand, weight_to_transport))
                            demand.untransported_weight -= weight_to_transport
                            self.demand_paths[demand.demand_id].append((path, weight_to_transport))
                            path_found = True
                        else:
                            continue  # No capacity, try next path
                    else:
                        # Explore connecting flights
                        connecting_flights = [flight for flight in self.flights
                                              if flight.origin == last_flight.destination and
                                              flight.departure_time >= last_flight.arrival_time + last_flight.aircraft_turnaround_time and
                                              (flight.destination, flight.aircraft_id) not in visited]

                        for conn_flight in connecting_flights:
                            new_path = path + [conn_flight]
                            total_arrival_time = conn_flight.arrival_time
                            heapq.heappush(flight_queue, PrioritizedItem(total_arrival_time, (new_path, conn_flight.aircraft_id)))

                if not path_found:
                    print(f"Could not fully transport Demand {demand.demand_id}. Untransported weight: {demand.untransported_weight}")
                    break  # Cannot transport more of this demand

        # Calculate total untransported weight
        total_untransported_weight = sum(demand.untransported_weight for demand in self.demands)
        transported_weight = total_demand_weight - total_untransported_weight

        print(f"Total demand weight: {total_demand_weight}")
        print(f"Total transported weight: {transported_weight}")
        print(f"Total untransported weight: {total_untransported_weight}")

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
        self.generate_flight_schedule()
        function_times['generate_flight_schedule'] = time.time() - t0
        
        t0 = time.time()
        self.generate_initial_solution()
        function_times['generate_initial_solution'] = time.time() - t0
        
        t0 = time.time()
        self.display_results()
        function_times['display_results'] = time.time() - t0
        
        total_time = time.time() - start_time
        function_times['total_time'] = time.time() - t0
        for func_name, duration in function_times.items():
            print(f"{func_name} took {duration:.6f} seconds")
            
    def get_possible_connections(self, destination):
        # Return possible intermediate destinations that can connect to the final destination
        # For simplicity, let's return all airports connected to the destination
        connected_airports = set()
        for (orig, dest) in self.flight_durations.keys():
            if dest == destination:
                connected_airports.add(orig)
        return connected_airports
    
    def find_shortest_path(self, demand):
        # Dijkstra's algorithm to find the shortest path based on arrival time
        flight_graph = defaultdict(list)
        for flight in self.flights:
            flight_graph[flight.origin].append(flight)

        visited = set()
        heap = []
        # Start with flights departing from the demand's origin after demand_ready_time
        for flight in flight_graph[demand.origin]:
            if flight.departure_time >= demand.demand_ready_time:
                heapq.heappush(heap, (flight.arrival_time, [flight]))

        while heap:
            arrival_time, path = heapq.heappop(heap)
            last_flight = path[-1]
            if last_flight.destination == demand.destination:
                return path  # Found a path to the destination
            if (last_flight.destination, last_flight.aircraft_id) in visited:
                continue
            visited.add((last_flight.destination, last_flight.aircraft_id))
            for next_flight in flight_graph.get(last_flight.destination, []):
                if next_flight.departure_time >= last_flight.arrival_time + last_flight.aircraft_turnaround_time:
                    new_path = path + [next_flight]
                    heapq.heappush(heap, (next_flight.arrival_time, new_path))
        return None  # No path found

    def display_results(self):
        print("\nAircraft Routes:")
        for aircraft in self.aircraft:
            route = ' -> '.join([f"{flight.origin}->{flight.destination}({flight.flight_id})" for flight in aircraft.route])
            print(f"Aircraft {aircraft.aircraft_id} (Base {aircraft.home_base}) Route: {route}")

        print("\nDemand Assignments:")
        for demand in self.demands:
            if demand.demand_id in self.demand_paths:
                print(f"Demand {demand.demand_id} assigned paths:")
                for path, weight in self.demand_paths[demand.demand_id]:
                    flight_ids = [f.flight_id for f in path]
                    print(f"  Flights: {flight_ids}, Weight: {weight}")
            else:
                print(f"Demand {demand.demand_id} could not be assigned any path.")

        print("\nUntransported Demands:")
        for demand in self.demands:
            if demand.untransported_weight > 0:
                print(f"Demand {demand.demand_id} untransported weight: {demand.untransported_weight}")