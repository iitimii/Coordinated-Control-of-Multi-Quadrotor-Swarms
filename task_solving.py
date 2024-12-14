from envs.TaskAviary import TaskAviary
import numpy as np

def run():
    env = TaskAviary(gui=True)

    env.reset()
    action = np.zeros((1,4))
    for i in range(100000000):
        env.step(action)
        env.render()


if __name__ == "__main__":
    run()