import numpy as np


def avoid_collision(sensors, left_motor, right_motor):
    speed_factor = (abs(left_motor) + abs(right_motor)) / 2
    movement_factor = 1 - np.sqrt(abs(left_motor - right_motor) / 2)
    proximity_factor = 1 - np.max(sensors)
    return speed_factor * movement_factor * proximity_factor
