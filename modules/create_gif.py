"""
Usage:
1. после инициализации среды, перед симуляцией эпизода создаём text_info = [get_text_info(YOUR_ARGS)]
2. после каждого степа text_info.append(get_text_info(YOUR_ARGS))
3. после окончания эпизода вызываем create_gif(wrapped_env, gif_path, duration, text_info)

* YOUR_ARGS - любые ваши аргументы, содержащую инфу, которую нужно отображать на гифке справа
* Внутрь get_text_info добавляете сколько угодно аналогичных словариков, меняете position, text, color...

регулировка duration не работает ))
"""
import ffmpy
import os
import shutil
import imageio
import numpy as np
import cv2

from modules.reward import RewardBasedModel


def __extend_image_width(image: np.ndarray, width: int, color):
    image_extended = np.ndarray((image.shape[0], image.shape[1] + width, image.shape[2]), dtype=image.dtype)
    image_extended[:, :image.shape[1]] = image
    image_extended[:, image.shape[1]:] = color
    return image_extended


def __paint_text_on_image(image, text, position, font_face=cv2.FONT_HERSHEY_SIMPLEX,
                          font_scale=0.7, color=(200, 0, 0), thickness=2):
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


def create_video_from_gif(gif_path):
    video_path = gif_path[:-4] + '.mp4'

    if os.path.exists(video_path):
        os.remove(video_path)

    ffmpy.FFmpeg(
        inputs={gif_path: None},
        outputs={video_path: None},
        global_options='-loglevel quiet'
    ).run()


def get_text_info(r, info, env, model):
    """r is Reward object"""
    out = []

    out.extend([
        dict(text=f'step: {env.realm.step_num}',
             position=(330, 50),                          
             color=(170, 170, 170)),          

        dict(text=f'bonus_count: {info["predators"][0]["bonus_count"]}',
             position=(30, 100),                          
             color=(200, 200, 0))
    ])

    if isinstance(model, RewardBasedModel):
        out.extend([
            dict(text=f'Expected rewards:',
                 position=(30, 140),                                  
                 color=(0, 180, 180)),                 

            dict(text=f'left:    {round(model.expected_info["left"][0], 2) if len(model.expected_info) > 0 else None}',
                 position=(30, 175),                                  
                 color=(0, 180, 180)),                 

            dict(text=f'right:    {round(model.expected_info["right"][0], 2)if len(model.expected_info) > 0 else None}',
                 position=(30, 200),                                  
                 color=(0, 180, 180)),                 

            dict(text=f'up:    {round(model.expected_info["up"][0], 2)if len(model.expected_info) > 0 else None}',
                 position=(30, 225),                                  
                 color=(0, 180, 180)),                 

            dict(text=f'down:    {round(model.expected_info["down"][0], 2)if len(model.expected_info) > 0 else None}',
                 position=(30, 250),                                  
                 color=(0, 180, 180))                 

        ])
    else:
        left = model.q_values[0][2].item() if model.q_values is not None else None
        right = model.q_values[0][1].item() if model.q_values is not None else None
        up = model.q_values[0][3].item() if model.q_values is not None else None
        down = model.q_values[0][4].item() if model.q_values is not None else None

        def get_q_value_color(x): 
          if x is None:
               return (0, 180, 180)
          if x == max(left, right, up, down):
               return (0, 240, 240)
          return (0, 180, 180)

        q_values_block_y = 175
        q_values_block_x = 110

        out.extend([
            dict(text=f'Q values:',
                 position=(30, q_values_block_y),                                  
                 color=(0, 180, 180)),


            dict(text=f'left:',
                 position=(30, q_values_block_y + 35),                                  
                 color=get_q_value_color(left)),    

            dict(text=f'{round(left, 2) if model.q_values is not None else None }',
                 position=(q_values_block_x, q_values_block_y + 35),                                  
                 color=get_q_value_color(left)),    


            dict(text=f'right:',
                 position=(30, q_values_block_y + 60),                                  
                 color=get_q_value_color(right)),     

            dict(text=f'{round(right, 2) if model.q_values is not None else None }',
                 position=(q_values_block_x, q_values_block_y + 60),                                  
                 color=get_q_value_color(right)),  


            dict(text=f'up:',
                 position=(30, q_values_block_y + 85),                                  
                 color=get_q_value_color(up)),     

            dict(text=f'{round(up, 2) if model.q_values is not None else None }',
                 position=(q_values_block_x, q_values_block_y + 85),                                  
                 color=get_q_value_color(up)), 


            dict(text=f'down:',
                 position=(30, q_values_block_y + 110),                                  
                 color=get_q_value_color(down)),      

            dict(text=f'{round(down, 2) if model.q_values is not None else None }',
                 position=(q_values_block_x, q_values_block_y + 110),                                  
                 color=get_q_value_color(down)),            
        ])

    out.extend([
        dict(text=f'reward: {round(r.result[0], 3) if r.result is not None else None}',
             position=(30, 375),                          
             color=(200, 0, 200)),             

        dict(text=f'scores: {info["scores"]}',
             position=(30, 425),                          
             color=(0, 200, 0))             
    ])

    return out
