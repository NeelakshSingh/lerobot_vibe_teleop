from abc import ABC, abstractmethod
from lerobothackathonenv.lerobot_types import *

from numpy import clip, array, exp
from numpy.linalg import norm
import numpy as np

import mujoco
from gymnasium import spaces

class ExtendedTask(Task, ABC):
    """
    This class represents one "variation" of the
    LeRobot environment. Subclass this to create
    new variations.
    """
    XML_PATH: Path
    ACTION_SPACE: Space
    OBSERVATION_SPACE: Space

    @abstractmethod
    def get_reward(self, physics: Physics) -> float:
        raise NotImplementedError("get_sim_metadata is not implemented!")

    @abstractmethod
    def get_observation(self, physics: Physics) -> Obs:
        raise NotImplementedError("get_sim_metadata is not implemented!")

    @abstractmethod
    def get_sim_metadata(self,) -> Dict:
        raise NotImplementedError("get_sim_metadata is not implemented!")

# ~ Define a task without reward yet that uses the
#   a generated mujoco xml file for the physics

class ExampleTask(ExtendedTask):
    XML_PATH = (
            Path(__file__).parent
                / "models"
                / "xml"
                / "so101_tabletop_manipulation_generated.xml"
        )
    ACTION_SPACE = spaces.Box(low=-1, high=1, shape=(6,), dtype=float64)

    RANGE_QPOS = (-3.0, 3.0)
    RANGE_QVEL = (-10.0, 10.0)
    RANGE_AF = (-3.35, 3.35)
    RANGE_GRIPPER = (-1.5, 1.5)
    OBSERVATION_SPACE = spaces.Dict(
        dict(
            qpos=spaces.Box(*RANGE_QPOS, shape=(27,), dtype=float64),
            qvel=spaces.Box(*RANGE_QVEL, shape=(24,), dtype=float64),
            actuator_force=spaces.Box(*RANGE_AF, shape=(6,), dtype=float64),
            gripper_pos=spaces.Box(*RANGE_GRIPPER, shape=(3,), dtype=float64),
        )
    )

    def __init__(self, random=None):
        super().__init__(random=random)

    def get_observation(
        self,
        physics: Physics
    ) -> Obs:
        data = physics.data
        gripper_site_id = mujoco.mj_name2id(
            physics.model._model,
            mujoco.mjtObj.mjOBJ_SITE.value,
            "gripperframe"
        )
        gripper_pos = data.site_xpos[gripper_site_id]
        obs = dict(
            qpos=clip(data.qpos, *self.RANGE_QPOS),
            qvel=clip(data.qvel, *self.RANGE_QVEL),
            actuator_force=clip(data.actuator_force, *self.RANGE_AF),
            gripper_pos=clip(gripper_pos, *self.RANGE_GRIPPER)
        )
        return obs

# ~ Take the task above and make it a reaching task
#   by defining the appropriate reward function

class ExampleReachTask(ExampleTask):
    def __init__(self, random=None, target_pos: NDArray = array([0, 0, 1.1])):
        super(ExampleTask, self).__init__(random=random)
        self.target_pos = target_pos

    def get_reward(
        self,
        physics: Physics
    ) -> float:
        obs = self.get_observation(physics)
        gripper_pos = obs.get("gripper_pos")
        delta = self.target_pos - gripper_pos
        sigma = 0.1
        reward = exp(-norm(delta) ** 2 / ( 2 * sigma ))
        return reward

    def get_sim_metadata(self,):
        return {"target_pos": self.target_pos.copy()}

class GoalConditionedObjectPlaceTask(ExampleTask):
    DELTA = 0.05
    TABLE_HEIGHT = 0.6
    RANGE_TARGET_POS = (-0.4, 0.4)
    # ~ Body ids of objects to be manipulated
    MANIPULATABLES: List[int] = [
        9, # milk_0
        11, # bread_1
        13 # cereal_2
    ]
    OBSERVATION_SPACE = spaces.Dict(
        dict(
            qpos=spaces.Box(*ExampleTask.RANGE_QPOS, shape=(27,), dtype=float64),
            qvel=spaces.Box(*ExampleTask.RANGE_QVEL, shape=(24,), dtype=float64),
            actuator_force=spaces.Box(*ExampleTask.RANGE_AF, shape=(6,), dtype=float64),
            gripper_pos=spaces.Box(*ExampleTask.RANGE_GRIPPER, shape=(3,), dtype=float64),
            target_pos=spaces.Box(*RANGE_TARGET_POS, shape=(3,), dtype=float64),
            object_index=spaces.Box(0, 1, shape=(len(MANIPULATABLES), ), dtype=float64)
        )
    )


    def __init__(self, random=None):
        super(ExampleTask, self).__init__(random=random)
        self.resample_goal()

    def resample_goal(self,):
        x, y = np.random.uniform(*self.RANGE_TARGET_POS, (2,))
        z = self.TABLE_HEIGHT + self.DELTA
        self.target_pos = array([x, y, z])
        self.focus_object = np.random.randint(
            len(self.MANIPULATABLES)
        )

    def get_reward(
        self,
        physics: Physics
    ) -> float:
        data = physics.data
        object_pos = data.xpos[self.MANIPULATABLES[self.focus_object]]
        print(object_pos)
        cost = norm(object_pos - self.target_pos)
        return -float(cost)

    @staticmethod
    def one_hot(index: int, size: int) -> NDArray[float64]:
        vector = np.zeros(size)
        vector[index] = 1.0
        return vector

    def get_sim_metadata(self,):
        return {"target_pos": self.target_pos.copy()}

    def get_observation(
        self,
        physics: Physics
    ) -> Obs:
        data = physics.data
        gripper_site_id = mujoco.mj_name2id(
            physics.model._model,
            mujoco.mjtObj.mjOBJ_SITE.value,
            "gripperframe"
        )
        gripper_pos = data.site_xpos[gripper_site_id]
        obs = dict(
            qpos=clip(data.qpos, *self.RANGE_QPOS).copy(),
            qvel=clip(data.qvel, *self.RANGE_QVEL).copy(),
            actuator_force=clip(data.actuator_force, *self.RANGE_AF).copy(),
            gripper_pos=clip(gripper_pos, *self.RANGE_GRIPPER).copy(),
            target_pos=self.target_pos.copy(),
            object_index=self.one_hot(
                self.focus_object,
                len(self.MANIPULATABLES)
            )
        )
        return obs

# Define more tasks here...
