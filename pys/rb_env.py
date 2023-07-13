import gymnasium as gym
from gymnasium import spaces

class RbEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, tick_path, render_mode=None, history_size = 70):
        self._render_mode = render_mode
        self.observation_space= spaces.Dict(
            {
                "hold": spaces.Box(0, 4, shape=(2,), dtype=int),
                "history": spaces.Box(0, history_size, shape=(5,), dtype=float)
            }
        )
        self.action_space = spaces.Discrete(3) # up, hold, down

    def _gen_obs(self):
        return {
            "notImplemented"
        }

    def _gen_info(self):
        return "notImplemented"    


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self._render_mode == "human":
            self._render_frame()
        return self._gen_obs(), self._gen_info()


    def step(self, action):
        reward = 0
        terminated = False

        if self._render_mode == "human":
            self._render_frame()
        return self._gen_obs(), reward, terminated, False, self._gen_info()
    

    
