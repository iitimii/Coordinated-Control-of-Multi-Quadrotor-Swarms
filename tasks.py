import numpy as np
from enum import Enum
from typing import List, Dict, Tuple
from dataclasses import dataclass
import heapq

class DroneState(Enum):
    IDLE = 0
    MOVING_TO_PICKUP = 1
    PICKING = 2
    MOVING_TO_DROP = 3
    DROPPING = 4

@dataclass
class Position:
    x: float
    y: float
    z: float

@dataclass
class Task:
    pickup: Position
    dropoff: Position
    priority: int = 0
    
class SequenceCoordinator:
    def __init__(self, safety_distance: float = 1.0):
        self.drone_states: Dict[str, DroneState] = {}
        self.drone_positions: Dict[str, Position] = {}
        self.active_tasks: Dict[str, Task] = {}
        self.task_queue = []
        self.safety_distance = safety_distance
        
    def add_drone(self, drone_id: str, initial_position: Position):
        """Register a new drone to the system."""
        self.drone_states[drone_id] = DroneState.IDLE
        self.drone_positions[drone_id] = initial_position
        
    def add_task(self, task: Task):
        """Add a new pick-and-place task to the queue."""
        heapq.heappush(self.task_queue, (task.priority, task))
        
    def check_collision_risk(self, pos1: Position, pos2: Position) -> bool:
        """Check if two positions are too close."""
        distance = np.sqrt((pos1.x - pos2.x)**2 + 
                         (pos1.y - pos2.y)**2 + 
                         (pos1.z - pos2.z)**2)
        return distance < self.safety_distance
        
    def update_drone_position(self, drone_id: str, position: Position):
        """Update the known position of a drone."""
        self.drone_positions[drone_id] = position
        
    def get_next_waypoint(self, drone_id: str) -> Tuple[Position, bool]:
        """
        Get the next safe waypoint for a drone.
        Returns (waypoint, can_proceed)
        """
        if drone_id not in self.active_tasks:
            if not self.task_queue:
                return None, False
            _, task = heapq.heappop(self.task_queue)
            self.active_tasks[drone_id] = task
            self.drone_states[drone_id] = DroneState.MOVING_TO_PICKUP
            
        current_state = self.drone_states[drone_id]
        current_task = self.active_tasks[drone_id]
        
        # Determine target position based on state
        if current_state in [DroneState.IDLE, DroneState.MOVING_TO_PICKUP]:
            target = current_task.pickup
        else:
            target = current_task.dropoff
            
        # Check for collision risks with other drones
        for other_id, other_pos in self.drone_positions.items():
            if other_id != drone_id:
                if self.check_collision_risk(target, other_pos):
                    # Generate temporary waypoint to avoid collision
                    offset = self.safety_distance * 1.5
                    return Position(
                        target.x + offset,
                        target.y + offset,
                        target.z + 0.5
                    ), True
                    
        return target, True
        
    def complete_task(self, drone_id: str):
        """Mark current task as complete and reset drone state."""
        if drone_id in self.active_tasks:
            del self.active_tasks[drone_id]
        self.drone_states[drone_id] = DroneState.IDLE