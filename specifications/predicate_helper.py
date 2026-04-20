import numpy as np

def distance_to_goal(position, GOAL_CENTER, GOAL_RADIUS):
    """Compute distance from position to goal boundary.
    
    Positive = inside goal
    Negative = outside goal
    """
   
    dist_to_center = np.linalg.norm(position - np.array(GOAL_CENTER))
    return GOAL_RADIUS - dist_to_center