from dataclasses import dataclass, field
import itertools

# Some Settings
transition_time_of_aircraft = 60
transition_time_of_demand = 40

@dataclass(order=True)
class PrioritizedItem:
    priority: int
    count: int
    item: object = field(compare=False)

@dataclass
class Demand:
    demand_id: int
    origin: str
    destination: str
    total_weight: float
    demand_ready_time: int  # Time when demand becomes available (in minutes)
    handling_time: int = transition_time_of_demand  # Time to load/unload demand when changing aircraft
    untransported_weight: float = field(init=False)  # Initialize in __post_init__

    def __post_init__(self):
        self.untransported_weight = self.total_weight

@dataclass
class Aircraft:
    aircraft_id: int
    capacity: float
    home_base: str = None  # Assigned during aircraft base assignment
    available_time: int = 0  # Time when the aircraft becomes available
    current_location: str = None
    route: list = field(default_factory=list)  # Sequence of flights assigned
    turnaround_time: int = transition_time_of_aircraft  # Aircraft turnaround time in minutes

@dataclass
class Flight:
    flight_id: int
    origin: str
    destination: str
    departure_time: int
    arrival_time: int
    aircraft_id: int = None  # Assigned during scheduling
    capacity: float = None   # Assigned based on aircraft
    demands_assigned: list = field(default_factory=list)  # Demands assigned
    aircraft_turnaround_time: int = transition_time_of_aircraft  # Time needed for aircraft between flights

@dataclass
class Airport:
    code: str
    
class Vertex:
    def __init__(self, code):
        self.code = code  # Airport code
        self.edges = []   # List of outgoing edges

class FlightEdge:
    def __init__(self, origin, destination):
        self.origin = origin           # Origin Vertex
        self.destination = destination # Destination Vertex
        self.flight_id = None          # Will be assigned when the flight is scheduled
        self.departure_time = None     # Will be assigned when the flight is scheduled
        self.arrival_time = None       # Will be assigned when the flight is scheduled
        self.aircraft_id = None        # Will be assigned when the flight is scheduled
        self.capacity = None           # Will be assigned when the flight is scheduled
        self.demands_assigned = []     # List of (demand, weight)