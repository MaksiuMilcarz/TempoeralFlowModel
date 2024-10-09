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
        self.max_path_length = 5  # Maximum flights in a path for a demand

        self.num_airports = 12
        self.num_aircraft = 10
        self.num_demands = 35
        self.demand_ready_times = [0, 720, 1440]

        self.flight_durations = {}
        self.possible_flights = set()