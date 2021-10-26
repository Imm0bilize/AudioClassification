from glob import glob

CLASS_NAME = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']
DELIMITER = ';'
IDX_CLASS_NAME_IN_PATH = 1

with open('dataset.csv', 'a') as file:
    file.write(f"index{DELIMITER}path{DELIMITER}class_name{DELIMITER}class_number\n")
    for idx, path_to_file in enumerate(glob('Emotions/*/*.wav')):
        class_name = path_to_file.split('/')[IDX_CLASS_NAME_IN_PATH]
        class_number = CLASS_NAME.index(class_name)
        file.write(f'{idx}{DELIMITER}./datasets/{path_to_file}{DELIMITER}{class_name}{DELIMITER}{class_number}\n')
