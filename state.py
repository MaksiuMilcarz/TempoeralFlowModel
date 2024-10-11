from collections import defaultdict

class State:
    def __init__(self):
        self.demands = []
        self.aircraft = []
        self.airports = {}
        self.flights = []
        self.flight_schedule = {}
        self.demand_paths = defaultdict(list)  # Records flight IDs and weights

        self.SCHEDULE_DURATION = 2 * 24 * 60  # Two days in minutes
        self.RESERVED_RETURN_TIME = 180  # Reserved time in minutes for return
        

        self.num_airports = 100
        self.num_aircraft = 70
        self.num_demands = 100
        self.max_path_length =  10
        self.demand_ready_times = [0, 720, 1440]

        self.flight_durations = {}
        self.possible_flights = set()
        
        # New attributes for the explicit graph
        self.graph = None  # Will be initialized in setup.py