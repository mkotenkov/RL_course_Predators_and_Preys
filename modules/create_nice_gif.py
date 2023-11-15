"""
Usage:
1. после инициализации среды, перед симуляцией эпизода создаём text_info = [get_text_info(YOUR_ARGS)]
2. после каждого степа text_info.append(get_text_info(YOUR_ARGS))
3. после окончания эпизода вызываем create_gif(wrapped_env, gif_path, duration, text_info)

* YOUR_ARGS - любые ваши аргументы, содержащую инфу, которую нужно отображать на гифке справа
* Внутрь get_text_info добавляете сколько угодно аналогичных словариков, меняете position, text, color...

регулировка duration не работает ))
"""
from IPython.display import Image, display
import ffmpy

import os
import shutil
import imageio
import numpy as np
import cv2

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

def create_video_from_gif(gif_path):
    video_path = gif_path[:-4] + '.mp4'

    if os.path.exists(video_path):
        os.remove(video_path)

    ffmpy.FFmpeg(
        inputs={gif_path: None},
        outputs={video_path: None},
        global_options='-loglevel quiet'
    ).run()

def get_text_info(r, info, env, potential_rewards):
    """r is Reward object"""        
    return [
        dict(text=f'step: {env.realm.step_num}',
             position=(300, 50),
             font_face=cv2.FONT_HERSHEY_SIMPLEX,
             font_scale=0.7,
             color=(150, 150, 150),
             thickness=2),

        dict(text=f'left:    {round(potential_rewards["left"][0], 2)}',
             position=(30, 150),
             font_face=cv2.FONT_HERSHEY_SIMPLEX,
             font_scale=0.7,
             color=(200, 0, 0),
             thickness=2),

        dict(text=f'right:    {round(potential_rewards["right"][0], 2)}',
             position=(30, 175),
             font_face=cv2.FONT_HERSHEY_SIMPLEX,
             font_scale=0.7,
             color=(200, 0, 0),
             thickness=2),

        dict(text=f'up:    {round(potential_rewards["up"][0], 2)}',
             position=(30, 200),
             font_face=cv2.FONT_HERSHEY_SIMPLEX,
             font_scale=0.7,
             color=(200, 0, 0),
             thickness=2),

        dict(text=f'down:    {round(potential_rewards["down"][0], 2)}',
             position=(30, 225),
             font_face=cv2.FONT_HERSHEY_SIMPLEX,
             font_scale=0.7,
             color=(200, 0, 0),
             thickness=2),
             

        dict(text=f'kills:    {r.kills[0] if r.result is not None else None}',
             position=(30, 300),
             font_face=cv2.FONT_HERSHEY_SIMPLEX,
             font_scale=0.7,
             color=(200, 0, 0),
             thickness=2),

        dict(text=f'reward: {round(r.result[0], 3) if r.result is not None else None}',
             position=(30, 350),
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

