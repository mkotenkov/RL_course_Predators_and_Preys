from IPython.display import Image, display
import ffmpy
import os
import shutil
import imageio
import numpy as np
import cv2

from modules.preprocess import preprocess
from modules.reward import Reward
from modules.utils import get_env


def __extend_image_width(image: np.ndarray, width: int, color):
    image_extended = np.ndarray((image.shape[0], image.shape[1] + width, image.shape[2]), dtype=image.dtype)
    image_extended[:, :image.shape[1]] = image
    image_extended[:, image.shape[1]:] = color
    return image_extended


def __paint_text_on_image(image, text, position, font_face=cv2.FONT_HERSHEY_SIMPLEX,
                          font_scale=1, color=(200, 0, 0), thickness=2):
    image_with_text = image.copy()
    cv2.putText(image_with_text, text, position, font_face, font_scale, color, thickness)
    return image_with_text


def create_gif(wrapped_env, gif_path, duration, text_info, EXTRA_WIDTH=480):
    temp_path_name = 'temp_path'
    wrapped_env.render(dir=temp_path_name)
    images = []
    lst = sorted(os.listdir(temp_path_name),
                 key=lambda x: int(x.split('.')[0]))

    for filename, text_group in zip(lst, text_info):
        file_path = os.path.join(temp_path_name, filename)
        image = imageio.imread(file_path)
        image = cv2.resize(image, (EXTRA_WIDTH, EXTRA_WIDTH), interpolation=cv2.INTER_AREA)
        image = __extend_image_width(image, EXTRA_WIDTH, (0, 0, 0))

        for text_kwargs in text_group:
            text_kwargs['position'] = (text_kwargs['position'][0] + EXTRA_WIDTH, text_kwargs['position'][1])
            image = __paint_text_on_image(image, **text_kwargs)

        images.append(image)

    imageio.mimsave(gif_path, images, duration=duration)
    shutil.rmtree(temp_path_name)


def create_and_display_gif(wrapped_env, gif_path, duration, text_info):
    create_gif(wrapped_env, gif_path, duration, text_info)
    display(Image(gif_path))


def get_text_info(r, reward, info, env):
    """r is Reward object"""
    return [
        dict(text=f'step: {env.realm.step_num}',
             position=(300, 50),
             font_face=cv2.FONT_HERSHEY_SIMPLEX,
             font_scale=0.7,
             color=(150, 150, 150),
             thickness=2),

        dict(text=f'd_prey_or_enemy: {int(r.dist_difference[0, 0]) if reward is not None else None}',
             position=(30, 50),
             font_face=cv2.FONT_HERSHEY_SIMPLEX,
             font_scale=0.7,
             color=(200, 0, 0),
             thickness=2),

        dict(text=f'd_teammate: {int(r.dist_difference[0, 1]) if reward is not None else None}',
             position=(30, 100),
             font_face=cv2.FONT_HERSHEY_SIMPLEX,
             font_scale=0.7,
             color=(200, 0, 0),
             thickness=2),

        dict(text=f'd_bonus:    {int(r.dist_difference[0, 2]) if reward is not None else None}',
             position=(30, 150),
             font_face=cv2.FONT_HERSHEY_SIMPLEX,
             font_scale=0.7,
             color=(200, 0, 0),
             thickness=2),

        dict(text=f'kills:    {r.kills[0] if reward is not None else None}',
             position=(30, 200),
             font_face=cv2.FONT_HERSHEY_SIMPLEX,
             font_scale=0.7,
             color=(200, 0, 0),
             thickness=2),

        dict(text=f'reward: {round(reward[0], 3) if reward is not None else None}',
             position=(30, 300),
             font_face=cv2.FONT_HERSHEY_SIMPLEX,
             font_scale=0.7,
             color=(50, 0, 255),
             thickness=2),

        dict(text=f'scores: {info["scores"]}',
             position=(30, 400),
             font_face=cv2.FONT_HERSHEY_SIMPLEX,
             font_scale=0.7,
             color=(0, 200, 0),
             thickness=2),
    ]


def create_video_from_gif(gif_path):
    video_path = gif_path[:-4] + '.mp4'

    if os.path.exists(video_path):
        os.remove(video_path)

    ffmpy.FFmpeg(
        inputs={gif_path: None},
        outputs={video_path: None},
        global_options='-loglevel quiet'
    ).run()


def simulate_episode_and_create_gif(model, difficulty, n_predators, cfg, gif_path):
    env = get_env(n_predators, difficulty, cfg.max_steps_for_episode, render_gif=True)
    state, info = env.reset()
    processed_state = preprocess(state, info)
    done = False
    r = Reward(n_predators=n_predators, **cfg.reward_weights)
    text_info = [get_text_info(r, None, info, env)]

    while not done:
        actions = model.get_actions(processed_state)
        next_state, done, next_info = env.step(actions)
        next_processed_state = preprocess(next_state, next_info)
        reward = r(processed_state, info, next_processed_state, next_info)        
        info, processed_state = next_info, next_processed_state
        text_info.append(get_text_info(r, reward, next_info, env))  # for display

    create_and_display_gif(env, gif_path, duration=1., text_info=text_info)
    create_video_from_gif(gif_path)

    return info['scores'][0] - info['scores'][1]
