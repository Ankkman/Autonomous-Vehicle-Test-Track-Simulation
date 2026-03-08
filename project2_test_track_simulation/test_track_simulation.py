"""
=============================================================================
PROJECT 2: Test Track Scenario Simulation for Autonomous Vehicle
=============================================================================
Author: Ankur Debnath
Description: Virtual test track with 6 event-triggered driving scenarios.
             An autonomous vehicle drives through each scenario, and an
             automated test evaluator grades performance as PASS/FAIL.

Scenarios:
    1. Pedestrian Crossing     — Must stop before crosswalk
    2. Stop Sign Compliance    — Must come to full stop before sign
    3. Slow Vehicle Following  — Must maintain safe following distance
    4. Emergency Braking       — Must stop before sudden obstacle
    5. Speed Limit Zone        — Must adapt to new speed limit
    6. Overtaking Maneuver     — Must safely pass slow vehicle

Architecture:
    Scenario Trigger → FSM Decision → Speed Control → Vehicle Dynamics
                                                    → Test Evaluator

Key Concepts: Scenario-Based Testing, FSM, Vehicle Dynamics, V-Model Testing
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

@dataclass
class TrackConfig:
    """
    All simulation parameters in one place.
    The track is a straight road divided into scenario zones.
    """
    # --- Time ---
    dt: float = 0.05  # 50ms timestep

    # --- Vehicle ---
    initial_speed: float = 0.0
    default_cruise_speed: float = 16.67  # 60 km/h
    max_speed: float = 22.22             # 80 km/h
    max_acceleration: float = 3.5        # m/s² (comfortable)
    max_deceleration: float = 7.0        # m/s² (hard braking)
    emergency_deceleration: float = 10.0 # m/s² (ABS-level)
    vehicle_length: float = 4.5

    # --- Track Layout (meters from start) ---
    track_total_length: float = 1200.0

    # Scenario 1: Pedestrian Crossing
    s1_zone_start: float = 80.0
    s1_crosswalk: float = 150.0    # Must stop BEFORE this line
    s1_zone_end: float = 200.0

    # Scenario 2: Stop Sign
    s2_zone_start: float = 250.0
    s2_sign: float = 320.0         # Must stop BEFORE this
    s2_zone_end: float = 370.0

    # Scenario 3: Slow Vehicle Following
    s3_zone_start: float = 420.0
    s3_slow_vehicle_start: float = 470.0  # Where slow car initially is
    s3_zone_end: float = 620.0
    s3_slow_speed: float = 8.33    # 30 km/h
    s3_min_follow: float = 10.0    # Minimum safe gap
    s3_max_follow: float = 35.0    # Max reasonable gap

    # Scenario 4: Emergency Braking
    s4_zone_start: float = 660.0
    s4_obstacle: float = 720.0     # Stationary obstacle
    s4_zone_end: float = 770.0
    s4_detection_range: float = 40.0

    # Scenario 5: Speed Limit Zone
    s5_zone_start: float = 810.0
    s5_sign: float = 830.0        # Speed limit sign
    s5_zone_end: float = 960.0
    s5_speed_limit: float = 8.33   # 30 km/h
    s5_tolerance: float = 1.5     # m/s tolerance

    # Scenario 6: Overtaking
    s6_zone_start: float = 1000.0
    s6_slow_vehicle_start: float = 1050.0
    s6_zone_end: float = 1180.0
    s6_slow_speed: float = 6.94   # 25 km/h

    # --- FSM Thresholds ---
    approach_distance: float = 50.0   # Start slowing when object < 50m
    stop_distance: float = 15.0       # Initiate full stop < 15m
    follow_distance: float = 40.0     # Enter follow mode < 40m
    clear_distance: float = 45.0      # Resume cruise > 45m

    # --- Control ---
    speed_control_gain: float = 0.8


# =============================================================================
# SECTION 2: VEHICLE STATES (FSM)
# =============================================================================

class VehicleState(Enum):
    IDLE = "Idle"
    ACCELERATING = "Accelerating"
    CRUISING = "Cruising"
    APPROACHING = "Approaching"
    FOLLOWING = "Following"
    STOPPING = "Stopping"
    STOPPED = "Stopped"
    OVERTAKING = "Overtaking"


# =============================================================================
# SECTION 3: TRACK OBJECTS AND SCENARIO RESULTS
# =============================================================================

@dataclass
class TrackObject:
    """An object placed on the test track."""
    name: str
    obj_type: str
    base_position: float
    speed: float = 0.0
    detection_range: float = 55.0
    active: bool = True
    activation_time: Optional[float] = None  # When car first enters this object's zone

    def current_position(self, sim_time: float) -> float:
        """Get current position accounting for movement."""
        if self.speed > 0 and self.activation_time is not None:
            elapsed = sim_time - self.activation_time
            return self.base_position + self.speed * max(0, elapsed)
        return self.base_position


@dataclass
class ScenarioResult:
    """Pass/fail result for one test scenario."""
    scenario_id: int
    name: str
    criteria: str
    passed: Optional[bool] = None
    actual_result: str = ""
    entry_time: Optional[float] = None
    exit_time: Optional[float] = None
    entry_speed: float = 0.0
    min_distance: float = float('inf')
    max_speed_in_zone: float = 0.0
    did_stop: bool = False
    stop_position: Optional[float] = None


# =============================================================================
# SECTION 4: SCENARIO MANAGER
# =============================================================================

class ScenarioManager:
    """
    Manages all 6 scenarios — places objects, detects which zone
    the car is in, and evaluates pass/fail criteria.
    """

    def __init__(self, config: TrackConfig):
        self.config = config
        self.c = config

        # Create the 6 scenario results
        self.results: List[ScenarioResult] = [
            ScenarioResult(1, "Pedestrian Crossing",
                           "Stop completely before crosswalk line"),
            ScenarioResult(2, "Stop Sign Compliance",
                           "Full stop before stop sign"),
            ScenarioResult(3, "Slow Vehicle Following",
                           f"Maintain {config.s3_min_follow}-{config.s3_max_follow}m gap"),
            ScenarioResult(4, "Emergency Braking",
                           "Stop before hitting sudden obstacle"),
            ScenarioResult(5, "Speed Limit Zone",
                           f"Speed ≤ {config.s5_speed_limit + config.s5_tolerance:.1f} m/s in zone"),
            ScenarioResult(6, "Overtaking Maneuver",
                           "Safely pass slow vehicle"),
        ]

        # Place objects on the track
        self.objects: List[TrackObject] = [
            TrackObject("Pedestrian", "pedestrian", config.s1_crosswalk,
                        speed=0.0, detection_range=55.0),
            TrackObject("Stop Sign", "stop_sign", config.s2_sign,
                        speed=0.0, detection_range=55.0),
            TrackObject("Slow Car (30 km/h)", "slow_vehicle", config.s3_slow_vehicle_start,
                        speed=config.s3_slow_speed, detection_range=60.0),
            TrackObject("Sudden Obstacle", "obstacle", config.s4_obstacle,
                        speed=0.0, detection_range=config.s4_detection_range),
            TrackObject("Speed Limit 30", "speed_sign", config.s5_sign,
                        speed=0.0, detection_range=50.0),
            TrackObject("Very Slow Car (25 km/h)", "overtake_vehicle", config.s6_slow_vehicle_start,
                        speed=config.s6_slow_speed, detection_range=60.0),
        ]

        # Tracking for scenario 3 (following distances)
        self.follow_distances: List[float] = []
        # Tracking for scenario 5 (speeds in zone after adjustment period)
        self.speeds_in_zone: List[float] = []
        # Tracking for scenario 6 (overtake progress)
        self.overtake_started: bool = False
        self.overtake_completed: bool = False

    # --- Zone boundaries for each scenario ---
    def _zones(self):
        c = self.c
        return [
            (c.s1_zone_start, c.s1_zone_end),
            (c.s2_zone_start, c.s2_zone_end),
            (c.s3_zone_start, c.s3_zone_end),
            (c.s4_zone_start, c.s4_zone_end),
            (c.s5_zone_start, c.s5_zone_end),
            (c.s6_zone_start, c.s6_zone_end),
        ]

    def get_active_zone(self, pos: float) -> Optional[int]:
        """Return which scenario zone (0-5) the car is in, or None."""
        for i, (start, end) in enumerate(self._zones()):
            if start <= pos <= end:
                return i
        return None

    def get_speed_limit(self, pos: float) -> float:
        """Return the speed limit at this position."""
        c = self.c
        if c.s5_sign <= pos <= c.s5_zone_end:
            return c.s5_speed_limit
        return c.default_cruise_speed

    def get_object_for_zone(self, zone_idx: int) -> Optional[TrackObject]:
        """Get the track object associated with a scenario zone."""
        if 0 <= zone_idx < len(self.objects):
            return self.objects[zone_idx]
        return None

    def get_distance_to_object(
        self, vehicle_pos: float, zone_idx: int, sim_time: float
    ) -> Tuple[Optional[TrackObject], float]:
        """
        Calculate distance from vehicle to the relevant object in this zone.
        Returns (object, distance). Distance is positive if object is ahead.
        """
        obj = self.get_object_for_zone(zone_idx)
        if obj is None or not obj.active:
            return None, float('inf')

        # Activate moving objects when car enters their zone
        if obj.speed > 0 and obj.activation_time is None:
            obj.activation_time = sim_time

        obj_pos = obj.current_position(sim_time)
        distance = obj_pos - vehicle_pos

        # Only return if object is ahead and within detection range
        if 0 < distance <= obj.detection_range:
            return obj, distance

        # If vehicle has passed the object
        if distance <= 0:
            return obj, distance

        return None, float('inf')

    def evaluate(
        self, zone_idx: int, vehicle_pos: float, vehicle_speed: float,
        distance: float, sim_time: float, state: VehicleState,
        obj: Optional[TrackObject]
    ):
        """Continuously evaluate scenario criteria while in zone."""
        r = self.results[zone_idx]
        c = self.c

        # Mark entry
        if r.entry_time is None:
            r.entry_time = sim_time
            r.entry_speed = vehicle_speed

        # Track metrics
        r.max_speed_in_zone = max(r.max_speed_in_zone, vehicle_speed)
        if obj is not None and distance < r.min_distance:
            r.min_distance = max(0, distance)

        # Did vehicle stop?
        if vehicle_speed < 0.05:
            if not r.did_stop:
                r.did_stop = True
                r.stop_position = vehicle_pos

        # --- Scenario-specific evaluation ---

        if zone_idx == 0:  # Pedestrian Crossing
            if r.did_stop and r.stop_position is not None and r.passed is None:
                if r.stop_position < c.s1_crosswalk - 1.0:  # 1m tolerance
                    r.passed = True
                    gap = c.s1_crosswalk - r.stop_position
                    r.actual_result = f"Stopped at {r.stop_position:.1f}m ({gap:.1f}m before crosswalk)"
                else:
                    r.passed = False
                    r.actual_result = f"Stopped too late at {r.stop_position:.1f}m (crosswalk at {c.s1_crosswalk}m)"

        elif zone_idx == 1:  # Stop Sign
            if r.did_stop and r.stop_position is not None and r.passed is None:
                if r.stop_position < c.s2_sign - 1.0:
                    r.passed = True
                    gap = c.s2_sign - r.stop_position
                    r.actual_result = f"Stopped at {r.stop_position:.1f}m ({gap:.1f}m before sign)"
                else:
                    r.passed = False
                    r.actual_result = f"Stopped too late at {r.stop_position:.1f}m (sign at {c.s2_sign}m)"

        elif zone_idx == 2:  # Following
            if obj is not None and 0 < distance < 60:
                self.follow_distances.append(distance)

        elif zone_idx == 3:  # Emergency Braking
            if r.did_stop and r.stop_position is not None and r.passed is None:
                if r.stop_position < c.s4_obstacle - 1.0:
                    r.passed = True
                    gap = c.s4_obstacle - r.stop_position
                    r.actual_result = f"Emergency stop at {r.stop_position:.1f}m ({gap:.1f}m before obstacle)"
                else:
                    r.passed = False
                    r.actual_result = "COLLISION — did not stop in time"

        elif zone_idx == 4:  # Speed Limit
            # Start recording after 40m into the zone (give time to slow down)
            if vehicle_pos > c.s5_sign + 40:
                self.speeds_in_zone.append(vehicle_speed)

        elif zone_idx == 5:  # Overtaking
            if state == VehicleState.OVERTAKING:
                self.overtake_started = True
            if self.overtake_started and state in [VehicleState.CRUISING, VehicleState.ACCELERATING]:
                self.overtake_completed = True

    def finalize_zone(self, zone_idx: int, exit_time: float):
        """Finalize evaluation when vehicle leaves a zone."""
        r = self.results[zone_idx]
        r.exit_time = exit_time
        c = self.c

        # Scenario 3: Following — evaluate collected distances
        if zone_idx == 2 and r.passed is None:
            if len(self.follow_distances) > 10:
                avg = np.mean(self.follow_distances)
                mn = min(self.follow_distances)
                if mn >= c.s3_min_follow * 0.8:  # 80% tolerance
                    r.passed = True
                    r.actual_result = f"Avg distance: {avg:.1f}m, Min: {mn:.1f}m (limit: {c.s3_min_follow}m)"
                else:
                    r.passed = False
                    r.actual_result = f"Too close! Min: {mn:.1f}m (limit: {c.s3_min_follow}m)"
            else:
                r.passed = True
                r.actual_result = "Maintained safe distance throughout"

        # Scenario 5: Speed Limit — evaluate collected speeds
        elif zone_idx == 4 and r.passed is None:
            if len(self.speeds_in_zone) > 10:
                mx = max(self.speeds_in_zone)
                limit = c.s5_speed_limit + c.s5_tolerance
                if mx <= limit:
                    r.passed = True
                    r.actual_result = (f"Max speed: {mx:.1f} m/s ({mx*3.6:.0f} km/h), "
                                       f"Limit: {limit:.1f} m/s ({limit*3.6:.0f} km/h)")
                else:
                    r.passed = False
                    r.actual_result = (f"EXCEEDED! Max: {mx:.1f} m/s, "
                                       f"Limit: {limit:.1f} m/s")
            else:
                r.passed = True
                r.actual_result = "Speed within limits"

        # Scenario 6: Overtaking
        elif zone_idx == 5 and r.passed is None:
            if self.overtake_completed:
                r.passed = True
                r.actual_result = "Successfully overtook slow vehicle"
            else:
                r.passed = False
                r.actual_result = "Failed to complete overtaking maneuver"

        # Scenarios 1, 2, 4: if never got evaluated, mark as fail
        if zone_idx in [0, 1, 3] and r.passed is None:
            r.passed = False
            r.actual_result = "Vehicle did not stop as required"


# =============================================================================
# SECTION 5: DECISION MODULE (FSM)
# =============================================================================

class DecisionModule:
    """
    FSM-based decision module for the test track.

    8 states with priority-based transitions.
    Each scenario triggers specific state behavior.
    """

    def __init__(self, config: TrackConfig):
        self.config = config
        self.state = VehicleState.IDLE
        self.prev_state = VehicleState.IDLE
        self.state_entry_time: float = 0.0
        self.transition_log: List[Dict] = []

        # Stopping/stopped tracking
        self.stopped_time: Optional[float] = None
        self.stop_wait_duration: float = 2.5  # Wait 2.5s at stops before resuming

        # Overtaking tracking
        self.overtake_phase: int = 0   # 0=none, 1=lane out, 2=passing, 3=lane back
        self.overtake_start_time: float = 0.0

        # Cleared obstacle tracking — so we don't re-trigger on passed objects
        self.cleared_zones: set = set()

    def _transition(self, new_state: VehicleState, t: float, pos: float,
                    speed: float, zone: Optional[int]):
        """Execute a state transition with logging."""
        if new_state != self.state:
            self.transition_log.append({
                'time': t, 'position': pos,
                'from': self.state.value, 'to': new_state.value,
                'speed': speed, 'zone': zone
            })
            self.prev_state = self.state
            self.state = new_state
            self.state_entry_time = t

            # Reset stopped timer on state change away from STOPPED
            if new_state != VehicleState.STOPPED:
                self.stopped_time = None

    def decide(
        self, pos: float, speed: float, t: float,
        speed_limit: float, zone: Optional[int],
        obj: Optional[TrackObject], distance: float
    ) -> Tuple[VehicleState, float]:
        """
        Core FSM evaluation. Returns (state, target_speed).

        Priority:
        1. Emergency / stopping conditions
        2. Scenario-specific behavior
        3. Normal cruising
        """
        target = speed_limit
        time_in_state = t - self.state_entry_time

        # ==============================================
        # If we already cleared this zone, just cruise through
        # ==============================================
        if zone is not None and zone in self.cleared_zones:
            if speed < speed_limit * 0.9:
                self._transition(VehicleState.ACCELERATING, t, pos, speed, zone)
                return VehicleState.ACCELERATING, speed_limit
            else:
                self._transition(VehicleState.CRUISING, t, pos, speed, zone)
                return VehicleState.CRUISING, speed_limit

        # ==============================================
        # CURRENTLY STOPPED — handle resume logic
        # ==============================================
        if self.state == VehicleState.STOPPED:
            if self.stopped_time is None:
                self.stopped_time = t

            waited = t - self.stopped_time
            if waited >= self.stop_wait_duration:
                # Done waiting — mark zone as cleared and resume
                if zone is not None:
                    self.cleared_zones.add(zone)
                self.stopped_time = None
                self._transition(VehicleState.ACCELERATING, t, pos, speed, zone)
                return VehicleState.ACCELERATING, speed_limit
            else:
                return VehicleState.STOPPED, 0.0

        # ==============================================
        # CURRENTLY STOPPING — check if fully stopped
        # ==============================================
        if self.state == VehicleState.STOPPING:
            if speed < 0.05:
                self._transition(VehicleState.STOPPED, t, pos, speed, zone)
                return VehicleState.STOPPED, 0.0
            return VehicleState.STOPPING, 0.0

        # ==============================================
        # SCENARIO-SPECIFIC BEHAVIOR
        # ==============================================
        if zone is not None and obj is not None and distance != float('inf'):

            obj_type = obj.obj_type

            # --- Pedestrian / Stop Sign: Must stop before the object ---
            if obj_type in ("pedestrian", "stop_sign"):
                if distance > 0:  # Object is still ahead
                    if distance < self.config.stop_distance:
                        # Close enough — hard stop
                        self._transition(VehicleState.STOPPING, t, pos, speed, zone)
                        return VehicleState.STOPPING, 0.0
                    elif distance < self.config.approach_distance:
                        # Approaching — gradual slowdown
                        # Target speed proportional to distance
                        decel_target = speed_limit * (distance / self.config.approach_distance) * 0.5
                        decel_target = max(2.0, min(decel_target, speed_limit * 0.6))
                        self._transition(VehicleState.APPROACHING, t, pos, speed, zone)
                        return VehicleState.APPROACHING, decel_target
                else:
                    # Passed the object — we should have stopped earlier
                    # Mark zone as cleared to avoid getting stuck
                    self.cleared_zones.add(zone)

            # --- Slow Vehicle: Follow at safe distance ---
            elif obj_type == "slow_vehicle":
                if 0 < distance < self.config.follow_distance:
                    # Adjust speed to maintain safe gap
                    # Target = slow vehicle speed + correction based on distance
                    correction = (distance - 20.0) * 0.3  # Aim for ~20m gap
                    follow_target = obj.speed + correction
                    follow_target = np.clip(follow_target, obj.speed * 0.7, obj.speed * 1.3)
                    self._transition(VehicleState.FOLLOWING, t, pos, speed, zone)
                    return VehicleState.FOLLOWING, follow_target
                elif distance <= 0:
                    # Passed slow vehicle somehow
                    self.cleared_zones.add(zone)

            # --- Sudden Obstacle: Emergency Stop ---
            elif obj_type == "obstacle":
                if 0 < distance < self.config.s4_detection_range:
                    self._transition(VehicleState.STOPPING, t, pos, speed, zone)
                    return VehicleState.STOPPING, 0.0
                elif distance <= 0:
                    self.cleared_zones.add(zone)

            # --- Speed Sign: Just adjust speed limit (handled by speed_limit param) ---
            elif obj_type == "speed_sign":
                if speed > speed_limit * 1.1:
                    self._transition(VehicleState.APPROACHING, t, pos, speed, zone)
                    return VehicleState.APPROACHING, speed_limit
                else:
                    self._transition(VehicleState.CRUISING, t, pos, speed, zone)
                    return VehicleState.CRUISING, speed_limit

            # --- Overtake Vehicle: Lane change maneuver ---
            elif obj_type == "overtake_vehicle":
                if 0 < distance < self.config.follow_distance and self.overtake_phase == 0:
                    # Start overtaking
                    self.overtake_phase = 1
                    self.overtake_start_time = t
                    self._transition(VehicleState.OVERTAKING, t, pos, speed, zone)
                    return VehicleState.OVERTAKING, speed_limit * 1.1

                if self.overtake_phase >= 1:
                    elapsed = t - self.overtake_start_time
                    if elapsed < 2.0:
                        self.overtake_phase = 1  # Lane change out
                    elif elapsed < 5.0:
                        self.overtake_phase = 2  # Passing
                    elif elapsed < 7.0:
                        self.overtake_phase = 3  # Lane change back
                    else:
                        self.overtake_phase = 0  # Complete
                        self.cleared_zones.add(zone)
                        self._transition(VehicleState.CRUISING, t, pos, speed, zone)
                        return VehicleState.CRUISING, speed_limit

                    self._transition(VehicleState.OVERTAKING, t, pos, speed, zone)
                    return VehicleState.OVERTAKING, speed_limit * 1.1

        # ==============================================
        # NORMAL DRIVING (no scenario or between zones)
        # ==============================================
        if speed < 0.1 and self.state == VehicleState.IDLE:
            self._transition(VehicleState.ACCELERATING, t, pos, speed, zone)
            return VehicleState.ACCELERATING, speed_limit

        if speed < speed_limit * 0.85:
            self._transition(VehicleState.ACCELERATING, t, pos, speed, zone)
            return VehicleState.ACCELERATING, speed_limit
        else:
            self._transition(VehicleState.CRUISING, t, pos, speed, zone)
            return VehicleState.CRUISING, speed_limit


# =============================================================================
# SECTION 6: VEHICLE DYNAMICS
# =============================================================================

class Vehicle:
    """Simple vehicle physics with proportional speed control."""

    def __init__(self, config: TrackConfig):
        self.config = config
        self.position: float = 0.0
        self.speed: float = config.initial_speed
        self.acceleration: float = 0.0
        self.lateral_offset: float = 0.0

    def update(self, target_speed: float, dt: float, is_emergency: bool = False):
        """Update vehicle speed and position."""
        error = target_speed - self.speed

        if is_emergency:
            self.acceleration = -self.config.emergency_deceleration
        elif target_speed < 0.1 and self.speed > 0.3:
            # Need to stop — apply strong braking
            self.acceleration = -self.config.max_deceleration
        elif error > 0.5:
            self.acceleration = min(self.config.speed_control_gain * error,
                                    self.config.max_acceleration)
        elif error < -0.5:
            self.acceleration = max(self.config.speed_control_gain * error,
                                    -self.config.max_deceleration)
        else:
            self.acceleration = self.config.speed_control_gain * error * 0.5

        self.speed += self.acceleration * dt
        self.speed = np.clip(self.speed, 0.0, self.config.max_speed)

        # Prevent tiny oscillations around zero
        if self.speed < 0.02 and target_speed == 0:
            self.speed = 0.0
            self.acceleration = 0.0

        self.position += self.speed * dt
        return self.speed, self.position, self.acceleration


# =============================================================================
# SECTION 7: MAIN SIMULATION
# =============================================================================

class TestTrackSimulation:
    """Main simulation — runs the car through the entire track."""

    def __init__(self):
        self.config = TrackConfig()
        self.vehicle = Vehicle(self.config)
        self.decision = DecisionModule(self.config)
        self.scenario_mgr = ScenarioManager(self.config)

        # Data recording
        self.time_log: List[float] = []
        self.pos_log: List[float] = []
        self.speed_log: List[float] = []
        self.accel_log: List[float] = []
        self.state_log: List[str] = []
        self.target_log: List[float] = []
        self.zone_log: List[Optional[int]] = []
        self.distance_log: List[Optional[float]] = []
        self.lateral_log: List[float] = []
        self.limit_log: List[float] = []

    def run(self):
        """Execute the full test track simulation."""
        print("=" * 72)
        print("  AUTONOMOUS VEHICLE — TEST TRACK SCENARIO SIMULATION")
        print("=" * 72)
        c = self.config
        print(f"\n  Track Length: {c.track_total_length}m")
        print(f"  Default Speed: {c.default_cruise_speed:.1f} m/s ({c.default_cruise_speed*3.6:.0f} km/h)")
        print(f"  Scenarios: 6")
        print(f"\n{'─'*72}")
        print(f"  {'TIME':>6} | {'POS':>6} | {'STATE':<14} | {'SPEED':>7} | "
              f"{'TARGET':>7} | {'ZONE':>4} | EVENT")
        print(f"{'─'*72}")

        sim_time = 0.0
        last_print = -1.0
        prev_zone = None
        max_time = 300.0  # Safety: max 5 minutes simulation

        while self.vehicle.position < c.track_total_length and sim_time < max_time:

            # 1. Which zone are we in?
            zone = self.scenario_mgr.get_active_zone(self.vehicle.position)

            # 2. Did we leave a zone?
            if prev_zone is not None and zone != prev_zone:
                self.scenario_mgr.finalize_zone(prev_zone, sim_time)

            # 3. Get relevant object and distance
            obj, distance = None, float('inf')
            if zone is not None:
                obj, distance = self.scenario_mgr.get_distance_to_object(
                    self.vehicle.position, zone, sim_time
                )

            # 4. Get speed limit
            speed_limit = self.scenario_mgr.get_speed_limit(self.vehicle.position)

            # 5. FSM Decision
            state, target = self.decision.decide(
                self.vehicle.position, self.vehicle.speed, sim_time,
                speed_limit, zone, obj, distance
            )

            # 6. Is this an emergency braking scenario?
            is_emergency = (
                state == VehicleState.STOPPING
                and obj is not None
                and obj.obj_type == "obstacle"
                and 0 < distance < c.s4_detection_range
            )

            # 7. Update vehicle
            speed, pos, accel = self.vehicle.update(target, c.dt, is_emergency)

            # 8. Update lateral offset for overtaking visualization
            if state == VehicleState.OVERTAKING:
                phase = self.decision.overtake_phase
                if phase == 1:
                    self.vehicle.lateral_offset = min(3.5,
                        self.vehicle.lateral_offset + 3.0 * c.dt)
                elif phase == 3:
                    self.vehicle.lateral_offset = max(0.0,
                        self.vehicle.lateral_offset - 3.0 * c.dt)
            else:
                self.vehicle.lateral_offset = max(0.0,
                    self.vehicle.lateral_offset - 2.0 * c.dt)

            # 9. Evaluate scenario
            if zone is not None:
                self.scenario_mgr.evaluate(
                    zone, pos, speed, distance, sim_time, state, obj
                )

            # 10. Record data
            self.time_log.append(sim_time)
            self.pos_log.append(pos)
            self.speed_log.append(speed)
            self.accel_log.append(accel)
            self.state_log.append(state.value)
            self.target_log.append(target)
            self.zone_log.append(zone)
            self.distance_log.append(distance if obj and distance != float('inf') else None)
            self.lateral_log.append(self.vehicle.lateral_offset)
            self.limit_log.append(speed_limit)

            # 11. Print every 1 second
            if int(sim_time) > last_print:
                last_print = int(sim_time)
                z_str = f"  S{zone+1}" if zone is not None else "  --"
                event = ""

                # Zone entry
                if zone is not None and zone != prev_zone:
                    names = ["Pedestrian", "Stop Sign", "Follow",
                             "Emergency", "Speed Limit", "Overtake"]
                    event = f"▶ {names[zone]} Zone"

                # Object detection
                if obj and 0 < distance < obj.detection_range and not event:
                    event = f"👁 {obj.name} at {distance:.0f}m"

                # State change
                if self.decision.transition_log:
                    lt = self.decision.transition_log[-1]
                    if abs(lt['time'] - sim_time) < 0.6 and not event:
                        event = f"→ {lt['to']}"

                print(f"  {sim_time:5.1f}s | {pos:5.0f}m | {state.value:<14} | "
                      f"{speed:5.1f}m/s | {target:5.1f}m/s | {z_str} | {event}")

            prev_zone = zone
            sim_time += c.dt

        # Finalize last zone
        if prev_zone is not None:
            self.scenario_mgr.finalize_zone(prev_zone, sim_time)

        print(f"{'─'*72}")
        self._print_report()

    def _print_report(self):
        """Print the final test report."""
        print(f"\n{'='*72}")
        print(f"  TEST TRACK — SCENARIO EVALUATION REPORT")
        print(f"{'='*72}")

        passed_count = 0
        for r in self.scenario_mgr.results:
            icon = "✅ PASS" if r.passed else "❌ FAIL" if r.passed is False else "⚠ N/A"
            if r.passed:
                passed_count += 1

            print(f"\n  Scenario {r.scenario_id}: {r.name}")
            print(f"  {'─'*50}")
            print(f"    Status:   {icon}")
            print(f"    Criteria: {r.criteria}")
            print(f"    Result:   {r.actual_result}")
            if r.entry_time is not None and r.exit_time is not None:
                print(f"    Duration: {r.exit_time - r.entry_time:.1f}s")
                print(f"    Entry Speed: {r.entry_speed:.1f} m/s ({r.entry_speed*3.6:.0f} km/h)")
            if r.min_distance < float('inf'):
                print(f"    Min Distance to Object: {r.min_distance:.1f}m")
            if r.did_stop and r.stop_position:
                print(f"    Stop Position: {r.stop_position:.1f}m")

        total = len(self.scenario_mgr.results)
        print(f"\n{'='*72}")
        print(f"  OVERALL: {passed_count}/{total} SCENARIOS PASSED")
        if passed_count == total:
            print(f"  ✅ VEHICLE CERTIFIED — All scenarios passed")
        else:
            print(f"  ❌ NOT CERTIFIED — {total - passed_count} scenario(s) failed")

        # FSM stats
        print(f"\n  FSM Transitions: {len(self.decision.transition_log)}")

        state_times: Dict[str, float] = {}
        for s in self.state_log:
            state_times[s] = state_times.get(s, 0) + self.config.dt
        total_time = self.time_log[-1] if self.time_log else 1

        print(f"\n  Time in Each State:")
        for s, tv in sorted(state_times.items(), key=lambda x: -x[1]):
            print(f"    {s:<14}: {tv:5.1f}s ({tv/total_time*100:4.1f}%)")

        print(f"\n  Vehicle Performance:")
        print(f"    Distance: {self.vehicle.position:.0f}m")
        print(f"    Time:     {self.time_log[-1]:.1f}s")
        print(f"    Max Speed: {max(self.speed_log):.1f} m/s ({max(self.speed_log)*3.6:.0f} km/h)")
        print(f"    Avg Speed: {np.mean(self.speed_log):.1f} m/s ({np.mean(self.speed_log)*3.6:.0f} km/h)")
        print(f"{'='*72}")

    def plot_results(self):
        """Generate 5-panel test track visualization."""
        fig, axes = plt.subplots(5, 1, figsize=(18, 24))
        fig.suptitle(
            'Autonomous Vehicle — Test Track Scenario Simulation\n'
            '6 Scenarios: Pedestrian | Stop Sign | Follow | Emergency | Speed Limit | Overtake',
            fontsize=14, fontweight='bold', y=0.98
        )

        pos = np.array(self.pos_log)
        speed = np.array(self.speed_log)
        accel = np.array(self.accel_log)

        state_colors = {
            'Idle': '#95a5a6',
            'Accelerating': '#3498db',
            'Cruising': '#2ecc71',
            'Approaching': '#f39c12',
            'Following': '#e67e22',
            'Stopping': '#e74c3c',
            'Stopped': '#c0392b',
            'Overtaking': '#9b59b6',
        }

        zone_colors = ['#FFE0B2', '#BBDEFB', '#C8E6C9', '#FFCDD2', '#F0F4C3', '#E1BEE7']
        zone_names = ['S1: Pedestrian', 'S2: Stop Sign', 'S3: Follow',
                      'S4: Emergency', 'S5: Speed Limit', 'S6: Overtake']
        zones = self.scenario_mgr._zones()
        c = self.config

        # ===== SUBPLOT 1: Track Layout =====
        ax1 = axes[0]
        ax1.set_title('Test Track Layout (Bird\'s Eye View)', fontsize=13, fontweight='bold')
        ax1.set_xlabel('Track Position (meters)', fontsize=11)
        ax1.set_ylabel('Lane', fontsize=11)

        # Road
        ax1.axhspan(-1.5, 1.5, color='#E0E0E0', alpha=0.5)
        ax1.axhline(y=0, color='white', linestyle='--', linewidth=1, alpha=0.8)

        # Scenario zones
        for (start, end), color, name in zip(zones, zone_colors, zone_names):
            ax1.axvspan(start, end, alpha=0.3, color=color)
            ax1.text((start+end)/2, 2.2, name, ha='center', fontsize=8, fontweight='bold')

        # Vehicle path
        lat = np.array(self.lateral_log)
        ax1.plot(pos, lat, 'b-', linewidth=2, alpha=0.7, label='Vehicle Path')

        # Object markers
        obj_positions = [
            (c.s1_crosswalk, "🚶 Pedestrian", 'red'),
            (c.s2_sign, "🛑 Stop Sign", 'red'),
            (c.s3_slow_vehicle_start, "🚗 Slow Car", 'orange'),
            (c.s4_obstacle, "⚠ Obstacle", 'darkred'),
            (c.s5_sign, "30 km/h", 'blue'),
            (c.s6_slow_vehicle_start, "🚗 Very Slow", 'orange'),
        ]
        for obj_pos, label, color in obj_positions:
            ax1.annotate(label, xy=(obj_pos, 0), fontsize=7, ha='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.3))

        ax1.set_xlim(0, c.track_total_length)
        ax1.set_ylim(-2.5, 4)
        ax1.set_yticks([0, 3.5])
        ax1.set_yticklabels(['Main Lane', 'Overtake Lane'])
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3, axis='x')

        # ===== SUBPLOT 2: Speed Profile =====
        ax2 = axes[1]
        ax2.set_title('Speed Profile Along Track', fontsize=13, fontweight='bold')
        ax2.set_ylabel('Speed (m/s)', fontsize=11)

        for (start, end), color in zip(zones, zone_colors):
            ax2.axvspan(start, end, alpha=0.2, color=color)

        ax2.plot(pos, speed, 'b-', linewidth=2, label='Actual Speed')
        ax2.plot(pos, self.target_log, 'r--', linewidth=1.5, label='Target', alpha=0.7)
        ax2.plot(pos, self.limit_log, 'g:', linewidth=1.5, label='Speed Limit', alpha=0.7)

        # Pass/fail markers
        for i, r in enumerate(self.scenario_mgr.results):
            mid = (zones[i][0] + zones[i][1]) / 2
            mark = '✅' if r.passed else '❌' if r.passed is False else '⚠'
            ax2.annotate(mark, xy=(mid, max(speed)+1), fontsize=14, ha='center')

        ax2.set_xlim(0, c.track_total_length)
        ax2.set_ylim(-1, max(speed)+3)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)

        ax2r = ax2.twinx()
        ax2r.set_ylim(-1*3.6, (max(speed)+3)*3.6)
        ax2r.set_ylabel('km/h', fontsize=10)

        # ===== SUBPLOT 3: FSM Timeline =====
        ax3 = axes[2]
        ax3.set_title('FSM State Timeline', fontsize=13, fontweight='bold')

        prev_st = self.state_log[0]
        start_i = 0
        for i in range(1, len(self.state_log)):
            if self.state_log[i] != prev_st or i == len(self.state_log)-1:
                col = state_colors.get(prev_st, '#bdc3c7')
                ax3.axvspan(pos[start_i], pos[min(i, len(pos)-1)], alpha=0.6, color=col)
                w = pos[min(i, len(pos)-1)] - pos[start_i]
                if w > 25:
                    mid = (pos[start_i] + pos[min(i, len(pos)-1)]) / 2
                    ax3.text(mid, 0.5, prev_st, ha='center', va='center',
                            fontsize=7, fontweight='bold',
                            transform=ax3.get_xaxis_transform())
                prev_st = self.state_log[i]
                start_i = i

        for (start, end), color in zip(zones, zone_colors):
            ax3.axvspan(start, end, alpha=0.1, color=color)

        patches = [mpatches.Patch(color=c, label=s, alpha=0.6)
                   for s, c in state_colors.items()]
        ax3.legend(handles=patches, loc='upper right', fontsize=7, ncol=4)
        ax3.set_xlim(0, c.track_total_length)
        ax3.set_yticks([])
        ax3.grid(True, alpha=0.3, axis='x')

        # ===== SUBPLOT 4: Acceleration =====
        ax4 = axes[3]
        ax4.set_title('Acceleration / Braking Profile', fontsize=13, fontweight='bold')
        ax4.set_ylabel('Acceleration (m/s²)', fontsize=11)

        for (start, end), color in zip(zones, zone_colors):
            ax4.axvspan(start, end, alpha=0.2, color=color)

        ax4.fill_between(pos, 0, accel, where=(accel >= 0),
                         color='green', alpha=0.4, label='Throttle')
        ax4.fill_between(pos, 0, accel, where=(accel < 0),
                         color='red', alpha=0.4, label='Braking')
        ax4.plot(pos, accel, 'k-', linewidth=0.5, alpha=0.5)

        ax4.axhline(y=0, color='gray', alpha=0.3)
        ax4.axhline(y=-self.config.emergency_deceleration, color='darkred',
                    linestyle=':', alpha=0.3, label=f'Emergency ({self.config.emergency_deceleration} m/s²)')

        ax4.set_xlim(0, c.track_total_length)
        ax4.legend(loc='lower right', fontsize=9)
        ax4.grid(True, alpha=0.3)

        # ===== SUBPLOT 5: Results Summary =====
        ax5 = axes[4]
        ax5.set_title('Test Results Summary', fontsize=13, fontweight='bold')
        ax5.set_xlim(0, 7)
        ax5.set_ylim(0, 4)
        ax5.axis('off')

        short_names = ['Pedestrian', 'Stop Sign', 'Follow',
                       'Emergency', 'Speed Limit', 'Overtake']
        for i, r in enumerate(self.scenario_mgr.results):
            x = i + 0.5
            if r.passed:
                bc, tc = '#C8E6C9', '#2E7D32'
                status = "✅ PASS"
            elif r.passed is False:
                bc, tc = '#FFCDD2', '#C62828'
                status = "❌ FAIL"
            else:
                bc, tc = '#FFF9C4', '#F57F17'
                status = "⚠ N/A"

            rect = FancyBboxPatch((x-0.4, 0.5), 0.8, 3.0,
                                  boxstyle="round,pad=0.1",
                                  facecolor=bc, edgecolor=tc, linewidth=2)
            ax5.add_patch(rect)
            ax5.text(x, 3.2, f"S{r.scenario_id}", ha='center',
                    fontsize=11, fontweight='bold', color=tc)
            ax5.text(x, 2.6, short_names[i], ha='center',
                    fontsize=8, fontweight='bold', color=tc)
            ax5.text(x, 1.8, status, ha='center',
                    fontsize=10, fontweight='bold', color=tc)

            if r.stop_position and r.scenario_id in [1, 2, 4]:
                metric = f"Stopped\n{r.stop_position:.0f}m"
            elif r.scenario_id == 3 and r.actual_result:
                metric = f"Min gap\n{r.min_distance:.0f}m"
            elif r.actual_result:
                words = r.actual_result.split()[:3]
                metric = ' '.join(words)
            else:
                metric = ""
            ax5.text(x, 1.0, metric, ha='center', fontsize=7, color=tc)

        pc = sum(1 for r in self.scenario_mgr.results if r.passed)
        tot = len(self.scenario_mgr.results)
        oc = '#2E7D32' if pc == tot else '#C62828'
        ot = "CERTIFIED ✅" if pc == tot else "NOT CERTIFIED ❌"
        fig.text(0.5, 0.02, f"Overall: {pc}/{tot} Passed — {ot}",
                ha='center', fontsize=14, fontweight='bold', color=oc,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F5',
                         edgecolor=oc, linewidth=2))

        plt.tight_layout(rect=[0, 0.04, 1, 0.96])
        plt.savefig('test_track_results.png', dpi=150, bbox_inches='tight')
        print(f"\n  📊 Graph saved as: test_track_results.png")
        plt.show()


# =============================================================================
# SECTION 8: RUN
# =============================================================================

if __name__ == "__main__":
    print("\n  Initializing Test Track Simulation...")
    print("  6 Scenarios | 8-State FSM | Automated Pass/Fail Evaluation\n")

    sim = TestTrackSimulation()
    sim.run()
    sim.plot_results()

    print("\n  ✅ Test track simulation complete!")
    print("  Files generated:")
    print("    - test_track_results.png")
    print("\n  Methodology: Scenario-Based Testing (V-Model validation)\n")