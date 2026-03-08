# Test Track Scenario Simulation for Autonomous Vehicle

## Overview
Virtual test track with 6 event-triggered driving scenarios for autonomous 
vehicle validation. Automated pass/fail evaluation tests the vehicle's 
FSM-based decision module against specific safety criteria.

## Scenarios
| # | Scenario | Pass Criteria |
|---|----------|---------------|
| 1 | Pedestrian Crossing | Stop before crosswalk line |
| 2 | Stop Sign | Full stop before sign |
| 3 | Slow Vehicle | Maintain safe following distance (10-35m) |
| 4 | Emergency Braking | Stop before sudden obstacle |
| 5 | Speed Limit Zone | Comply with reduced 30 km/h limit |
| 6 | Overtaking | Safely pass slow vehicle |

## Architecture

<p align="center">
  <img width="178" height="425" alt="architecture_diagram" src="https://github.com/user-attachments/assets/56975a88-49d1-48a3-aa17-7ab4d2dc37e1" />
</p>



## How to Run
pip install numpy matplotlib  

python test_track_simulation.py

## Key Concepts

Scenario-Based Testing: Industry-standard validation method for ADAS/AV    
FSM Decision Making: 8-state behavioral planner    
Vehicle Dynamics: Physics-based speed and position simulation    
V-Model Testing: This project represents the validation phase     

## Results
6/6 scenarios passed — Vehicle certified ✅
- Pedestrian: Stopped 13.8m before crosswalk
- Stop Sign: Stopped 13.7m before sign  
- Following: Avg 23.5m gap, Min 19.3m (limit: 10m)
- Emergency: Stopped 26.4m before obstacle
- Speed Limit: Max 8.8 m/s in 9.8 m/s zone
- Overtaking: Completed in 7 seconds

## Performance
- Track: 1200m completed in 113.9s
- Avg Speed: 10.5 m/s (38 km/h)
- FSM Transitions: 25 (clean, no flickering)
