import time
import itertools
import heapq
from collections import defaultdict

from data_structs import Flight, PrioritizedItem
from data_structs import FlightEdge
from state import State
import setup

# =============================================================================

class InitialSolution:
    def __init__(self):
        self.state = State()  
    
# =============================================================================

    def generate_aircraft_routes(self):
        state = self.state
        flight_id_counter = 1

        for aircraft in state.aircraft:
            current_time = 0  # Start time at 0
            current_location = aircraft.home_base
            aircraft.route = []
            aircraft_route_complete = False

            while current_time < state.SCHEDULE_DURATION - state.RESERVED_RETURN_TIME and not aircraft_route_complete:
                # Identify possible next flights from current location
                possible_flights = [flight for flight in state.possible_flights if flight.origin.code == current_location]

                if not possible_flights:
                    # No flights available from current location
                    break

                # Prioritize flights based on demand or other criteria
                possible_flights.sort(key=lambda f: self.flight_priority(f))

                scheduled_flight = False
                for flight in possible_flights:
                    # Schedule the flight at the earliest possible time
                    departure_time = current_time + aircraft.turnaround_time
                    duration = abs(ord(flight.origin.code) - ord(flight.destination.code)) * 30 + 60
                    arrival_time = departure_time + duration

                    # Check if there's enough time to perform this flight and return home
                    time_to_return = self.estimate_return_time(flight.destination.code, aircraft.home_base, arrival_time)
                    if time_to_return is None or arrival_time + aircraft.turnaround_time + time_to_return > state.SCHEDULE_DURATION:
                        continue  # Not enough time; try next flight

                    # Assign flight details
                    flight.flight_id = flight_id_counter
                    flight.departure_time = departure_time
                    flight.arrival_time = arrival_time
                    flight.aircraft_id = aircraft.aircraft_id
                    flight.capacity = aircraft.capacity

                    # Add flight to aircraft route
                    aircraft.route.append(flight)
                    state.flights.append(flight)
                    flight_id_counter += 1

                    # Update current location and time
                    current_location = flight.destination.code
                    current_time = arrival_time
                    scheduled_flight = True

                    break  # Proceed to next iteration after scheduling a flight

                if not scheduled_flight:
                    # No suitable flights found; schedule return to base if needed
                    if current_location != aircraft.home_base:
                        current_location, current_time = self.schedule_return_flight(
                            aircraft, current_location, current_time)
                    aircraft_route_complete = True
                    break

                # Prevent infinite loops by limiting the number of flights per aircraft
                if len(aircraft.route) >= 50:  # Adjust as necessary
                    aircraft_route_complete = True
                    break

            # Schedule return to home base if not already there
            if current_location != aircraft.home_base:
                current_location, current_time = self.schedule_return_flight(
                    aircraft, current_location, current_time)
# =============================================================================

    def assign_demands_to_flights(self):
        state = self.state
        # Build a flight graph (now includes dynamically scheduled flights)
        flight_graph = defaultdict(list)
        for flight in state.flights:
            flight_graph[flight.origin.code].append(flight)

        # Map aircraft IDs to aircraft objects
        aircraft_dict = {ac.aircraft_id: ac for ac in state.aircraft}

        # Sort demands by ready time
        demands_sorted = sorted(state.demands, key=lambda d: d.demand_ready_time)

        for demand in demands_sorted:
            while demand.untransported_weight > 0:
                path = self.find_earliest_path(demand, flight_graph, aircraft_dict)
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
                        state.demand_paths[demand.demand_id].append((path, weight_to_assign, arrival_time))
                    else:
                        break  # No capacity available
                else:
                    break  # No path found
            if demand.untransported_weight > 0:
                # print(f"Demand {demand.demand_id} untransported weight: {demand.untransported_weight}")
                continue

# =============================================================================

    def find_earliest_path(self, demand, flight_graph, aircraft_dict):
        state = self.state
        max_path_length = state.max_path_length

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

            if last_flight.destination.code == demand.destination:
                return path  # Found a path

            if len(path) >= max_path_length:
                continue

            visited_key = (last_flight.destination.code, last_flight.arrival_time)
            if visited_key in visited:
                continue
            visited.add(visited_key)

            last_flight_aircraft = aircraft_dict[last_flight.aircraft_id]
            aircraft_turnaround_time = last_flight_aircraft.turnaround_time

            for next_flight in flight_graph.get(last_flight.destination.code, []):
                if next_flight.departure_time >= last_flight.arrival_time + aircraft_turnaround_time and \
                   next_flight.capacity - sum(w for (d, w) in next_flight.demands_assigned) > 0:
                    new_path = path + [next_flight]
                    heapq.heappush(heap, PrioritizedItem(next_flight.arrival_time, next(counter), new_path))

        return None  # No path found

# =============================================================================

    def schedule_return_flight(self, aircraft, current_location, current_time):
        state = self.state
        path = self.find_path_for_aircraft(current_location, aircraft.home_base, current_time, aircraft)
        if path:
            aircraft.route.extend(path)
            state.flights.extend(path)
            current_location = aircraft.home_base
            current_time = path[-1].arrival_time
        else:
            print(f"Aircraft {aircraft.aircraft_id} cannot return to base {aircraft.home_base} from {current_location}")
        return current_location, current_time
           
# =============================================================================

    def find_path_for_aircraft(self, origin_code, destination_code, current_time, aircraft):
        state = self.state
        max_path_length = state.max_path_length

        heap = []
        counter = itertools.count()

        # Start with possible flights from current location
        possible_flights = [flight for flight in state.possible_flights if flight.origin.code == origin_code]

        for flight_template in possible_flights:
            departure_time = current_time + aircraft.turnaround_time
            duration = abs(ord(flight_template.origin.code) - ord(flight_template.destination.code)) * 30 + 60
            arrival_time = departure_time + duration
            if arrival_time > state.SCHEDULE_DURATION:
                continue

            flight = FlightEdge(flight_template.origin, flight_template.destination)
            flight.flight_id = next(counter)
            flight.departure_time = departure_time
            flight.arrival_time = arrival_time
            flight.aircraft_id = aircraft.aircraft_id
            flight.capacity = aircraft.capacity

            heapq.heappush(heap, PrioritizedItem(arrival_time, flight.flight_id, [flight]))

        visited = set()

        while heap:
            item = heapq.heappop(heap)
            path = item.item
            last_flight = path[-1]

            if last_flight.destination.code == destination_code:
                return path  # Found a path

            if len(path) >= max_path_length:
                continue

            visited_key = (last_flight.destination.code, last_flight.arrival_time)
            if visited_key in visited:
                continue
            visited.add(visited_key)

            next_possible_flights = [flight for flight in state.possible_flights if flight.origin.code == last_flight.destination.code]

            for flight_template in next_possible_flights:
                departure_time = last_flight.arrival_time + aircraft.turnaround_time
                duration = abs(ord(flight_template.origin.code) - ord(flight_template.destination.code)) * 30 + 60
                arrival_time = departure_time + duration
                if arrival_time > state.SCHEDULE_DURATION:
                    continue

                flight = FlightEdge(flight_template.origin, flight_template.destination)
                flight.flight_id = next(counter)
                flight.departure_time = departure_time
                flight.arrival_time = arrival_time
                flight.aircraft_id = aircraft.aircraft_id
                flight.capacity = aircraft.capacity

                new_path = path + [flight]
                heapq.heappush(heap, PrioritizedItem(arrival_time, flight.flight_id, new_path))

        return None  # No path found

# =============================================================================

    def flight_priority(self, flight):
        # Implement your own logic to prioritize flights
        # For example, prioritize flights to high-demand destinations
        state = self.state
        demand_destinations = {d.destination for d in state.demands}
        if flight.destination.code in demand_destinations:
            return 0  # High priority
        else:
            return 1  # Lower priority
        
    def estimate_return_time(self, current_location, home_base, current_time):
        state = self.state
        # Use possible flights to estimate return time
        # For simplicity, assume direct flight if available
        for flight in state.possible_flights:
            if flight.origin.code == current_location and flight.destination.code == home_base:
                duration = abs(ord(current_location) - ord(home_base)) * 30 + 60
                return duration + state.aircraft[0].turnaround_time  # Assuming same turnaround time for all aircraft
        return None  # No direct flight available
     
     
     
     
     
     
     
     
     
     
     
     
     
    def run(self):
        state = self.state
        start_time = time.time()
        function_times = {}
        
        t0 = time.time()
        setup.load_data(state)
        function_times['load_data'] = time.time() - t0
        
        t0 = time.time()
        setup.assign_aircraft_bases(state)
        function_times['assign_aircraft_base'] = time.time() - t0
        
        t0 = time.time()
        setup.generate_possible_flights(state)
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

# =============================================================================

    def display_results(self):
        state = self.state
        print("\nAircraft Routes:")
        for aircraft in state.aircraft:
            route_str = ' -> '.join(f"{flight.origin.code}->{flight.destination.code}({flight.flight_id})"
                            for flight in aircraft.route)
            print(f"Aircraft {aircraft.aircraft_id} (Base {aircraft.home_base}): {route_str}")

        print("\nDemand Assignments:")
        total_demands = len(state.demands)
        total_weight = sum(d.total_weight for d in state.demands)
        transported_weight = 0
        delivery_times = []
        for demand in state.demands:
            paths = state.demand_paths.get(demand.demand_id, [])
            if paths:
                print(f"Demand {demand.demand_id} from {demand.origin} to {demand.destination}:")
                for path_flights, weight, arrival_time in paths:
                    for path_flights, weight, arrival_time in paths:
                        path_str = ' -> '.join(f"{flight.origin.code}->{flight.destination.code}({flight.flight_id})"
                                            for flight in path_flights)
                    print(f"  Path: {path_str}, Weight: {weight}, Arrival Time: {arrival_time}")
                    transported_weight += weight
                    delivery_times.append(arrival_time - demand.demand_ready_time)
            else:
                print(f"Demand {demand.demand_id} from {demand.origin} to {demand.destination}: Not transported")

        unmet_weight = sum(d.untransported_weight for d in state.demands)

        print("\nUntransported Demands:")
        for demand in state.demands:
            if demand.untransported_weight > 0:
                print(f"Demand {demand.demand_id} untransported weight: {demand.untransported_weight}")

        # Calculate statistics
        unmet_demand_percentage = (unmet_weight / total_weight) * 100 if total_weight > 0 else 0
        average_delivery_time = (sum(delivery_times) / len(delivery_times)
                                 if delivery_times else 0)
        unique_flights = len(set(flight.flight_id for flight in state.flights))
        total_capacity = sum(f.capacity for f in state.flights)
        cap_demand_ratio = total_capacity / total_weight if total_weight > 0 else 0

        print("\nStatistics:")
        print(f"Total Capacity: {total_capacity}")
        print(f"Total Demand Weight: {total_weight}")
        print(f"Total Transported Weight: {transported_weight}")
        print(f"Capacity-to-demand ratio: {cap_demand_ratio:.2f}")
        print(f"Unmet Demand Percentage: {unmet_demand_percentage:.2f}%")
        print(f"Average Delivery Time: {average_delivery_time:.2f} minutes")
        print(f"Number of Unique Flights: {unique_flights}")
