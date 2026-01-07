import gymnasium as gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv

# 创建环境 (指定一个平面跑道地图)
env = DuckietownEnv(
    map_name = "loop_empty",
    domain_rand = False,
    draw_curve = False,
    draw_bbox = False
)

obs = env.reset()
for _ in range(1000):
    # 随机动作: [线速度, 转向角]
    action = env.action_space.sample() 
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        env.reset()

env.close()