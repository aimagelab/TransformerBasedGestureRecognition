import cv2
import numpy as np

from pathlib import Path

def load_split_nvgesture(file_with_split='./nvgesture_train_correct.lst', list_split=list()):
    with open(file_with_split, 'rt') as f:
        dict_name = file_with_split[file_with_split.rfind('/') + 1:]
        dict_name = dict_name[:dict_name.find('_')]

        for line in f:
            params = line.split(' ')
            params_dictionary = dict()

            params_dictionary['dataset'] = dict_name

            path = params[0].split(':')[1]
            for param in params[1:]:
                parsed = param.split(':')
                key = parsed[0]
                if key == 'label':
                    # make label start from 0
                    label = int(parsed[1]) - 1
                    params_dictionary['label'] = label
                elif key in ('depth', 'color', 'duo_left'):
                    # first store path
                    params_dictionary[key] = path + '/' + parsed[1]
                    # store start frame
                    params_dictionary[key + '_start'] = int(parsed[2])

                    params_dictionary[key + '_end'] = int(parsed[3])

            params_dictionary['duo_right'] = params_dictionary['duo_left'].replace('duo_left', 'duo_right')
            params_dictionary['duo_right_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_right_end'] = params_dictionary['duo_left_end']

            params_dictionary['duo_disparity'] = params_dictionary['duo_left'].replace('duo_left', 'duo_disparity')
            params_dictionary['duo_disparity_start'] = params_dictionary['duo_left_start']
            params_dictionary['duo_disparity_end'] = params_dictionary['duo_left_end']

            list_split.append(params_dictionary)

    return list_split


def load_data_from_file(data_path, example_config, sensor, image_width, image_height, nogesture = False):
    path = example_config[sensor] + ".avi"
    path = Path(data_path) / path[path.find('/') + 1:]
    start_frame = example_config[sensor + '_start']
    end_frame = example_config[sensor + '_end']
    label = example_config['label']

    if end_frame - start_frame > 80:
        new_start = (end_frame - start_frame) // 2 - 40 + start_frame
        new_end = (end_frame - start_frame) // 2 + 40 + start_frame
        start_frame = new_start
        end_frame = new_end

    chnum = 3 if sensor == "color" else 1

    video_container = np.zeros((image_height, image_width, chnum, 160 if nogesture else 80), dtype=np.uint8)

    cap = cv2.VideoCapture(str(path))

    if nogesture:
        start_offset = 40 if start_frame >= 40 else start_frame
        end_offset = 40 if int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - end_frame >= 40 \
            else int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - end_frame
        frames_to_load = range(start_frame - start_offset, end_frame + end_offset)
    else:
        frames_to_load = range(start_frame, end_frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for indx, frameIndx in enumerate(frames_to_load):
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (image_width, image_height))
            if sensor != "color":
                frame = frame[..., 0]
                frame = frame[..., np.newaxis]
            video_container[..., indx] = frame
        else:
            print("Could not load frame")

    cap.release()

    if nogesture:
        return video_container, label, (start_offset, end_offset)
    else:
        return video_container, label, None