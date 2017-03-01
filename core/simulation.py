import warnings
from ctypes import cdll, c_int, POINTER, c_float, create_string_buffer, c_double
from core.constants import KHEPERA_LIB


class Simulation:
    _dll = cdll.LoadLibrary(KHEPERA_LIB)
    _dll.createSimulation.restype = POINTER(c_int)
    _dll.getRobot.restype = POINTER(c_int)
    _dll.getSensorState.restype = c_float
    _dll.getSensorCount.restype = c_int
    _dll.cloneSimulation.restype = POINTER(c_int)
    _dll.getXCoord.restype = c_int
    _dll.getYCoord.restype = c_int

    def __init__(self, wd_path=None):
        if wd_path is not None:
            self.sim = Simulation._dll.createSimulation(create_string_buffer(wd_path.encode()), False)
            self.initial_state = Simulation._dll.cloneSimulation(self.sim)
        self.robot = None
        self.robot_id = None
        self.sensor_states = None
        self.is_copy = wd_path is None

    def copy(self):
        sim = Simulation()
        sim.sim = Simulation._dll.cloneSimulation(self.sim)
        sim.initial_state = self.initial_state
        sim.set_controlled_robot(self.robot_id)
        sim.sensor_states = self.sensor_states
        return sim

    def reset(self):
        Simulation._dll.removeSimulation(self.sim)
        self.sim = Simulation._dll.cloneSimulation(self.initial_state)
        self.set_controlled_robot(self.robot_id)
        self.sensor_states = None

    def set_controlled_robot(self, robot_id):
        self.robot = Simulation._dll.getRobot(self.sim, robot_id)
        self.robot_id = robot_id

    def set_robot_speed(self, left_motor_speed, right_motor_speed):
        if self.robot is None:
            warnings.warn('No robot to control. Use Simulation.set_controlled_robot first.')
        else:
            Simulation._dll.setRobotSpeed(self.robot, c_double(left_motor_speed), c_double(right_motor_speed))

    def simulate(self, steps):
        Simulation._dll.updateSimulation(self.sim, steps)
        self.sensor_states = None

    def get_sensor_states(self):
        if self.robot is not None:
            warnings.warn('No robot to control. Use Simulation.set_controlled_robot first.')
        elif self.sensor_states is None:
            count = Simulation._dll.getSensorCount(self.robot)
            self.sensor_states = [Simulation._dll.getSensorState(self.robot, i) for i in range(count)]
        return self.sensor_states

    def get_robot_position(self):
        return Simulation._dll.getXCoord(self.robot), Simulation._dll.getYCoord(self.robot)

    def move_robot_random(self):
        Simulation._dll.moveRandom(self.sim, self.robot)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.sim is not None:
            Simulation._dll.removeSimulation(self.sim)
        if self.initial_state is not None and not self.is_copy:
            Simulation._dll.removeSimulation(self.initial_state)


class SimulationList:

    def __init__(self, path, num_sim, num_per_ctrl, robot_id):
        self.list = []
        self.num_per_ctrl = num_per_ctrl
        self.init_sim = Simulation(path)
        self.init_sim.set_controlled_robot(robot_id)
        for _ in range(num_sim):
            self.list.append([self.init_sim.copy() for _ in range(num_per_ctrl)])

    def reset(self):
        for sims in self.list:
            for sim in sims:
                sim.reset()

    def randomize(self):
        init_pos_sims = []
        for _ in range(self.num_per_ctrl):
            sim = self.init_sim.copy()
            sim.move_robot_random()
            init_pos_sims.append(sim)

        for i in range(len(self.list)):
            for sim in self.list[i]:
                sim.__exit__(None, None, None)
            self.list[i] = [sim.copy() for sim in init_pos_sims]

    def __getitem__(self, item):
        return self.list[item]

    def __iter__(self):
        return iter(self.list)

    def __len__(self):
        return len(self.list)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.init_sim.__exit__(exc_type, exc_val, exc_tb)
        for sims in self.list:
            for sim in sims:
                sim.__exit__(exc_type, exc_val, exc_tb)

