import numpy as np
from scipy.misc import imresize
import gym
import matplotlib.pyplot as plt

def preprocess_frame(frame, v_crop=(0, 0), h_crop=(0, 0)):
    """
    Preprocess image for faster computation

    Parameters
    ----------
    frame : ndarray
        Color image, of shape (H,W,C)

    v_crop : tuple, optional
        Defines how many rows of the image to remove from top and bottom, of shape (top, bottom)

    h_crop: tuple, optional
        Defines how many columns of the image to remove from left and right, of shape (left, right)

    Returns
    -------
    m : ndarray
        Greyscale image of shape(H, W)

    """

    heigth, width, _ = frame.shape
    frame = np.mean(frame, axis=2) / 255.0
    frame = frame[v_crop[0]:heigth - v_crop[1], h_crop[0]:width - h_crop[1]]
    frame = imresize(frame, size=(80, 80), interp='nearest')

    return frame

env = gym.make('BreakoutDeterministic-v4')
obs = env.reset()
plt.imshow(obs)
plt.imshow(preprocess_frame(obs))
for i in range(10):
    observation, _, _, _ = env.step(np.random.randint(1, env.action_space.n))

plt.imshow(preprocess_frame(observation))


