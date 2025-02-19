from gymnasium.envs.registration import register

register(
    id="gymnasium_env/MazeEnv-v0",
    entry_point="gymnasium_env.envs.simple_maze:MazeEnv",
)

register(
    id="gymnasium_env/MazeEnv-v1",
    entry_point="gymnasium_env.envs.simple_maze:EnrichMazeEnv",
)

register(
    id="gymnasium_env/VariableMazeEnv-v0",
    entry_point="gymnasium_env.envs.simple_maze:VariableMazeEnv",
)

register(
    id="gymnasium_env/VariableMazeEnv-v1",
    entry_point="gymnasium_env.envs.simple_maze:EnrichVariableMazeEnv",
)
