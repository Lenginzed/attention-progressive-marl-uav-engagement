import numpy as np

from .termination_condition_base import BaseTerminationCondition


class OutOfBounds(BaseTerminationCondition):
    def __init__(self, config):
        super().__init__(config)
        self.max_distance_from_center_m = getattr(config, "max_distance_from_center_m", 25000.0)

    def get_termination(self, task, env, agent_id, info={}):
        position = env.agents[agent_id].get_position()
        distance = np.linalg.norm(position[:2])
        done = distance >= self.max_distance_from_center_m
        if done:
            env.agents[agent_id].crash()
            info = self.record_reason(info, agent_id, "out_of_bounds")
            self.log(f"{agent_id} is out of bounds! Total Steps={env.current_step}")
        success = False
        return done, success, info
