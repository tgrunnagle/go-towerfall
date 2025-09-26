"""Environment module for RL training."""

from rl_bot_system.environment.game_environment import GameEnvironment, TrainingMode
from rl_bot_system.environment.state_processors import (
    StateProcessor, StateRepresentationType, StateProcessorFactory,
    RawCoordinateProcessor, GridBasedProcessor, FeatureVectorProcessor,
    ABTestingFramework
)
from rl_bot_system.environment.action_spaces import (
    ActionSpaceConfig, ActionSpaceType, ActionSpaceFactory,
    DiscreteActionSpace, ContinuousActionSpace, HybridActionSpace,
    MultiDiscreteActionSpace, ActionTimingController
)
from rl_bot_system.environment.reward_functions import (
    RewardFunction, RewardType, RewardFunctionFactory,
    SparseRewardFunction, DenseRewardFunction, ShapedRewardFunction,
    MultiObjectiveRewardFunction, HorizonBasedRewardFunction,
    RewardTuningFramework
)

__all__ = [
    # Core environment
    'GameEnvironment',
    'TrainingMode',
    
    # State processors
    'StateProcessor',
    'StateRepresentationType',
    'StateProcessorFactory',
    'RawCoordinateProcessor',
    'GridBasedProcessor',
    'FeatureVectorProcessor',
    'ABTestingFramework',
    
    # Action spaces
    'ActionSpaceConfig',
    'ActionSpaceType',
    'ActionSpaceFactory',
    'DiscreteActionSpace',
    'ContinuousActionSpace',
    'HybridActionSpace',
    'MultiDiscreteActionSpace',
    'ActionTimingController',
    
    # Reward functions
    'RewardFunction',
    'RewardType',
    'RewardFunctionFactory',
    'SparseRewardFunction',
    'DenseRewardFunction',
    'ShapedRewardFunction',
    'MultiObjectiveRewardFunction',
    'HorizonBasedRewardFunction',
    'RewardTuningFramework'
]