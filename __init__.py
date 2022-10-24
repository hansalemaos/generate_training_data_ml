import shutil
import sys
import xml.etree.ElementTree as ET
import regex
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
from math import ceil
from time import time
from imgaug import augmenters as iaa
import random
import numpy as np
import cv2
from pascal_voc_writer import Writer
import PILasOPENCV
from configparser import ConfigParser
import pandas as pd
from a_pandas_ex_less_memory_more_speed import pd_add_less_memory_more_speed

pd_add_less_memory_more_speed()


def add_pixxelborder(img, loop=10):
    looppercentage = loop
    loop = int(img.shape[0] / 100 * loop)
    if loop < 2:
        loop = 2
    image0 = img.copy()
    ending = ceil(255 / loop)
    try:
        for x in range(loop):
            x = x + 1
            bord = image0[:, :, 3][x : x + 1]
            test = random.randrange(x, x * ending) * np.random.random_sample(
                bord.shape
            ) + random.randrange(x, x * ending) * np.random.random_sample(bord.shape)
            test[test > 255] = 255
            test = test.astype(np.uint8)
            image0[:, :, 3][x : x + 1] = test
            image0[:, :, 3][image0.shape[0] - x - 1 : image0.shape[0] - x] = test
        image0 = np.rot90(image0)
        loop = looppercentage
        loop = int(image0.shape[0] / 100 * loop)
        if loop < 2:
            loop = 2
        ending = ceil(255 / loop)
        for x in range(loop):
            x = x + 1
            bord = image0[:, :, 3][x : x + 1]
            test = random.randrange(x, x * ending) * np.random.random_sample(
                bord.shape
            ) + random.randrange(x, x * ending) * np.random.random_sample(bord.shape)
            test[test > 255] = 255
            test = test.astype(np.uint8)
            image0[:, :, 3][x : x + 1] = test
            image0[:, :, 3][image0.shape[0] - x - 1 : image0.shape[0] - x] = test
        image0 = np.rot90(np.rot90(np.rot90(image0)))
        image0[:, :, 3][..., :2] = 0
        image0[:, :, 3][..., -2:] = 0
        image0[:, :, 3][:2] = 0
        image0[:, :, 3][-2:] = 0
        return image0
    except ValueError:
        return add_pixxelborder(img=img, loop=looppercentage - 1)


def merge_image(back, front, x, y, alphablur=130):
    if back.shape[2] == 3:
        back = cv2.cvtColor(back, cv2.COLOR_BGR2BGRA)
    if front.shape[2] == 3:
        front = cv2.cvtColor(front, cv2.COLOR_BGR2BGRA)
    bh, bw = back.shape[:2]
    fh, fw = front.shape[:2]
    x1, x2 = max(x, 0), min(x + fw, bw)
    y1, y2 = max(y, 0), min(y + fh, bh)
    front_cropped = front[y1 - y : y2 - y, x1 - x : x2 - x]
    back_cropped = back[y1:y2, x1:x2]

    alpha_front = front_cropped[:, :, 3:4] / (255 + alphablur)
    result = back.copy()
    result[y1:y2, x1:x2, :3] = (
        alpha_front * front_cropped[:, :, :3]
        + (1 - alpha_front) * back_cropped[:, :, :3]
    )
    result[y1:y2, x1:x2, 3:4] = 255

    return result


def intersects(box1, box2):
    return not (
        box1[2] < box2[0] or box1[0] > box2[2] or box1[1] > box2[3] or box1[3] < box2[1]
    )


def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles


def move_files_to_folder(list_of_files, destination_folder, concatfolder):
    destination_folder = os.path.join(concatfolder, destination_folder)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for f in list_of_files:
        try:
            shutil.move(f, destination_folder)
        except Exception as fe:
            print(f'{fe}', end='\r')
            continue
    return regex.sub(r"[\\/]+", "/", destination_folder).rstrip("/") + "/"


def get_config_dfs(configfile):
    config = ConfigParser()
    config.read(configfile)
    dfgeneral = pd.DataFrame(pd.Series(config.__dict__["_sections"]["GENERAL"]))
    dfgeneral = dfgeneral.T
    df = pd.DataFrame.from_records(
        [
            x[1]
            for x in config.__dict__["_sections"].items()
            if regex.search(r"BUTTON\d+", x[0]) is not None
        ]
    )
    for col in dfgeneral:
        try:
            dfgeneral[col] = dfgeneral[col].str.replace("""['"]+""", "", regex=True)
        except Exception as fe:
            pass
    for col in df:
        try:
            df[col] = df[col].str.replace("""['"]+""", "", regex=True)
        except Exception:
            pass
    df = (
        df.ds_string_to_mixed_dtype()
        .ds_reduce_memory_size(verbose=False)
        .ds_reduce_memory_size(verbose=False)
    )
    dfgeneral = (
        dfgeneral.ds_string_to_mixed_dtype()
        .ds_reduce_memory_size(verbose=False)
        .ds_reduce_memory_size(verbose=False)
    )
    return df, dfgeneral


def get_filters(
    percentagecrop,
    startcrop,
    endcrop,
    random_crop,
    percentageblur,
    startblur,
    endblur,
    random_blur,
):
    random_crop_ = True
    random_blur_ = True
    if random_crop == 0:
        random_crop_ = False
    if random_blur == 0:
        random_blur_ = False
    image_generation_settings = {
        "Crop": {
            "activated": random_crop_,
            "frequency": percentagecrop / 100,
            "command": (iaa.Crop(px=(startcrop, endcrop))),
        },
        "GaussianBlur1": {
            "activated": random_blur_,
            "frequency": percentageblur / 100,
            "command": (iaa.GaussianBlur(sigma=(startblur, endblur))),
        },
    }
    sequentialsettings = []
    for key, item in image_generation_settings.items():
        if item["activated"]:
            sequentialsettings.append(iaa.Sometimes(item["frequency"], item["command"]))
    seq = iaa.Sequential(sequentialsettings)
    return seq


def get_button_filter(dfbuttons):
    dfbuttons["aa_filter"] = dfbuttons.apply(
        lambda x: get_filters(
            percentagecrop=x.random_crop_percentage,
            startcrop=x.random_crop_min,
            endcrop=x.random_crop_max,
            random_crop=x.random_crop,
            percentageblur=x.random_blur_percentage,
            startblur=x.random_blur_min,
            endblur=x.random_blur_max,
            random_blur=x.random_crop,
        ),
        axis=1,
    )
    return dfbuttons


def extract_info_from_xml(xml_file, mapping_dict_for_classes):
    # from https://blog.paperspace.com/train-yolov5-custom-data/
    root = ET.parse(xml_file).getroot()
    class_name_to_id_mapping = {}
    counter = 0
    # Initialise the info dict
    info_dict = {}
    info_dict["bboxes"] = []

    # Parse the XML Tree
    for elem in root:
        # Get the file name
        if elem.tag == "filename":
            info_dict["filename"] = elem.text

        # Get the image size
        elif elem.tag == "size":
            image_size = []
            for subelem in elem:
                image_size.append(int(subelem.text))

            info_dict["image_size"] = tuple(image_size)

        # Get details of the bounding box
        elif elem.tag == "object":
            bbox = {}
            for subelem in elem:
                if subelem.tag == "name":
                    bbox["class"] = subelem.text
                    class_name_to_id_mapping[subelem.text] = mapping_dict_for_classes[
                        subelem.text
                    ]
                    counter = counter + 1

                elif subelem.tag == "bndbox":
                    for subsubelem in subelem:
                        bbox[subsubelem.tag] = int(subsubelem.text)
            info_dict["bboxes"].append(bbox)

    return class_name_to_id_mapping, info_dict


def convert_to_yolov5(info_dict, sav_f, class_name_to_id_mapping):
    # from https://blog.paperspace.com/train-yolov5-custom-data/

    print_buffer = []

    # For each bounding box
    for b in info_dict["bboxes"]:
        try:
            class_id = class_name_to_id_mapping[b["class"]]
        except KeyError:
            print("Invalid Class. Must be one from ", class_name_to_id_mapping.keys())

        # Transform the bbox co-ordinates as per the format required by YOLO v5
        b_center_x = (b["xmin"] + b["xmax"]) / 2
        b_center_y = (b["ymin"] + b["ymax"]) / 2
        b_width = b["xmax"] - b["xmin"]
        b_height = b["ymax"] - b["ymin"]

        # Normalise the co-ordinates by the dimensions of the image
        image_w, image_h, image_c = info_dict["image_size"]
        b_center_x /= image_w
        b_center_y /= image_h
        b_width /= image_w
        b_height /= image_h

        # Write the bbox details to the file
        print_buffer.append(
            "{} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                class_id, b_center_x, b_center_y, b_width, b_height
            )
        )

    # Name of the file which we have to save
    save_file_name = os.path.join(sav_f, info_dict["filename"].replace("png", "txt"))

    # Save the annotation to disk
    print("\n".join(print_buffer), file=open(save_file_name, "w"))


def read_button_images(dfbuttons, dfgeneral):
    allbuttons_images = {}
    for key, item in dfbuttons.iterrows():
        buttonpath = os.path.join(dfgeneral.image_button_folder.iloc[0], str(key))
        allbuttons_images[key] = [
            cv2.imread(x) for x in getListOfFiles(buttonpath) if str(x).endswith(".png")
        ]
    return allbuttons_images


def read_background_images(dfgeneral):
    image_background_folder_pic = [
        cv2.imread(x)
        for x in getListOfFiles(dfgeneral.image_background_folder.iloc[0])
        if str(x).endswith(".png")
    ]
    allims = []
    newsize = 640
    for bi in image_background_folder_pic:
        if bi.shape[0] != 360 and bi.shape[1] != 640:
            imbutton1 = PILasOPENCV.Image(bi)
            imbutton1 = imbutton1.resize(
                (newsize, int(newsize / imbutton1.size[0] * imbutton1.size[1]))
            )
            allims.append(imbutton1.getim().copy())
        else:
            allims.append(bi)
    return allims


def get_number_of_buttons_on_img(dfgeneral):
    buttons_on_pic = 1
    if int(dfgeneral.maximum_buttons_on_pic.iloc[0]) > 1:
        buttons_on_pic = random.randrange(
            1, int(dfgeneral.maximum_buttons_on_pic.iloc[0])
        )
    return buttons_on_pic


def get_sam_buttons(dfbuttons, number_of_buttons):
    sampleb = dfbuttons.sample(number_of_buttons)
    return sampleb


def generate_training_data(config_file):
    dfbuttons, dfgeneral = get_config_dfs(configfile=config_file)
    dfbuttons = get_button_filter(dfbuttons)
    all_button_images = read_button_images(dfbuttons, dfgeneral)
    all_background_images = read_background_images(dfgeneral)
    save_path_generated_pics = dfgeneral.save_path_generated_pics.iloc[0]
    if not os.path.exists(save_path_generated_pics):
        os.makedirs(save_path_generated_pics)
    picsmax = dfgeneral["number_of_pictures_to_generate"].iloc[0]
    generated_pic_folder = dfgeneral["save_path_generated_pics_separate"].iloc[0]

    personal_yaml = dfgeneral["yaml_file"].iloc[0]
    yolovyamel = os.path.join(generated_pic_folder, personal_yaml)

    counter = 0
    annotations = []
    images = []
    annotationstxt = []
    while True:
        number_of_buttons = get_number_of_buttons_on_img(dfgeneral)
        sampleb = get_sam_buttons(dfbuttons, number_of_buttons)
        random_background = random.choice(all_background_images).copy()
        allpicstouse = []
        for key, item in sampleb.iterrows():
            allpicstouse.append([key, random.choice(all_button_images[key]).copy()])

        random_background_as_pilcv = PILasOPENCV.Image(random_background)
        random_background_as_pilcv = random_background_as_pilcv.convert("RGBA")

        picsavepath = (
            os.path.join(save_path_generated_pics, str(counter).zfill(8)) + ".png"
        )
        picsavepathxml = (
            os.path.join(save_path_generated_pics, str(counter).zfill(8)) + ".xml"
        )
        picsavepathtxt = (
            os.path.join(save_path_generated_pics, str(counter).zfill(8)) + ".txt"
        )

        writer = Writer(
            picsavepath,
            random_background_as_pilcv.size[0],
            random_background_as_pilcv.size[1],
        )

        overlappingcheck = [(50000, 50000, 50001, 50001)]

        for ini, bu in enumerate(allpicstouse):
            pic_to_paste = bu[1]
            button1 = pic_to_paste.copy()
            key_for_infos = bu[0]
            tmpdfinfos = sampleb.loc[key_for_infos].T

            button_maxsizex = int(
                (random_background_as_pilcv.size[0] / 100) * tmpdfinfos["max_x"]
            )

            button_minsizex = int(
                (random_background_as_pilcv.size[0] / 100) * tmpdfinfos["min_x"]
            )

            try:
                final_width = random.randrange(button_minsizex, button_maxsizex)
            except Exception:
                final_width = random.randrange(button_minsizex, button_maxsizex + 2)

            ratio = final_width / button1.shape[1]
            final_height = int(button1.shape[0] * ratio)
            timeout = time() + 2
            while True:
                if timeout < time():
                    imbutton = PILasOPENCV.Image(button1)
                    break
                try:
                    imbutton = PILasOPENCV.Image(
                        tmpdfinfos["aa_filter"](images=[button1])[0].copy()
                    )
                    break
                except Exception as das:
                    pass
            imbutton = imbutton.resize((final_width, final_height))
            imbutton = imbutton.convert("RGBA")

            pixboarder = tmpdfinfos["add_pixelboarder"]

            if pixboarder:
                add_pixelboarder_percentage = tmpdfinfos["add_pixelboarder_percentage"]
                add_pixelboarder_dec = (
                    random.choices(
                        [True, False],
                        weights=[
                            add_pixelboarder_percentage,
                            100 - add_pixelboarder_percentage,
                        ],
                        k=1,
                    )
                )[0]
                if add_pixelboarder_dec:
                    imbutton._instance = add_pixxelborder(
                        imbutton._instance,
                        loop=random.randrange(
                            tmpdfinfos["unsharp_border"],
                            tmpdfinfos["unsharp_border_percentage"],
                        ),
                    )

            imbutton = imbutton.rotate(
                random.randrange(
                    tmpdfinfos["max_negativ_rotation"],
                    tmpdfinfos["max_positiv_rotation"],
                ),
                expand=True,
            )

            image_x_min = abs(
                ceil(
                    (random_background_as_pilcv.size[0] / 100)
                    * tmpdfinfos["allowed_min_distance_from_zero_x"]
                )
            )
            image_y_min = abs(
                ceil(
                    (random_background_as_pilcv.size[1] / 100)
                    * tmpdfinfos["allowed_min_distance_from_zero_y"]
                )
            )
            image_x_max = abs(
                ceil(
                    (random_background_as_pilcv.size[0] / 100)
                    * tmpdfinfos["allowed_max_distance_from_zero_x"]
                )
            )
            image_y_max = abs(
                ceil(
                    (random_background_as_pilcv.size[1] / 100)
                    * tmpdfinfos["allowed_max_distance_from_zero_y"]
                )
            )

            try:
                position_x_of_picture = random.randrange(image_x_min, image_x_max)
            except Exception:
                position_x_of_picture = random.randrange(image_x_min, image_x_max + 2)
            try:
                position_y_of_picture = random.randrange(image_y_min, image_y_max)
            except Exception:
                position_y_of_picture = random.randrange(image_y_min, image_y_max + 2)

            newpicstartx = position_x_of_picture
            newpicstarty = position_y_of_picture
            newpicsendx = position_x_of_picture + imbutton.size[0]
            newpicsendy = position_y_of_picture + imbutton.size[1]
            coordsofnewpic = newpicstartx, newpicstarty, newpicsendx, newpicsendy
            if True in ([intersects(coordsofnewpic, x) for x in overlappingcheck]):
                for __ in range(tmpdfinfos["max_overlapping_avoid"]):
                    try:
                        position_x_of_picture = random.randrange(
                            image_x_min, image_x_max
                        )
                    except Exception:
                        position_x_of_picture = random.randrange(
                            image_x_min, image_x_max + 2
                        )
                    try:
                        position_y_of_picture = random.randrange(
                            image_y_min, image_y_max
                        )
                    except Exception:
                        position_y_of_picture = random.randrange(
                            image_y_min, image_y_max + 2
                        )

                    newpicstartx = position_x_of_picture
                    newpicstarty = position_y_of_picture
                    newpicsendx = position_x_of_picture + imbutton.size[0]
                    newpicsendy = position_y_of_picture + imbutton.size[1]
                    coordsofnewpic = (
                        newpicstartx,
                        newpicstarty,
                        newpicsendx,
                        newpicsendy,
                    )
                    if True not in (
                        [intersects(coordsofnewpic, x) for x in overlappingcheck]
                    ):
                        overlappingcheck.append(coordsofnewpic)
                        break
                    else:
                        pass

            imbuttonnew1 = imbutton.getim()
            background1 = merge_image(
                back=random_background_as_pilcv.getim(),
                front=imbuttonnew1,
                x=position_x_of_picture,
                y=position_y_of_picture,
                alphablur=random.randrange(
                    tmpdfinfos["transparency_min"], tmpdfinfos["transparency_max"]
                ),
            )
            random_background_as_pilcv = PILasOPENCV.Image(background1)
            random_background_as_pilcv = random_background_as_pilcv.convert("RGBA")

            writer.addObject(
                tmpdfinfos["class_name"],
                position_x_of_picture,
                position_y_of_picture,
                position_x_of_picture + imbutton.size[0],
                position_y_of_picture + imbutton.size[1],
            )

        try:
            cv2.imwrite(picsavepath, random_background_as_pilcv.getim())
            writer.save(picsavepathxml)
            annotations.append(picsavepathxml)
            images.append(picsavepath)
            annotationstxt.append(picsavepathtxt)
            print(f"Pictures written: {str(counter).zfill(10)} / {picsmax}", end="\r")
            counter = counter + 1

            if counter > picsmax:
                break
        except Exception as fe:
            continue

    mapping_dict_for_classes = dfbuttons.class_name.to_dict()
    mapping_dict_for_classes = {v: k for k, v in mapping_dict_for_classes.items()}

    class_name_to_id_mapping = {}
    allyolovfiles = []
    for ann in tqdm(annotations):
        class_name_to_id_mappingtmp, info_dict = extract_info_from_xml(
            ann, mapping_dict_for_classes
        )
        for key, item in class_name_to_id_mappingtmp.items():
            class_name_to_id_mapping[key] = item
        allyolovfiles.append(info_dict)
    for info_dict in allyolovfiles:
        convert_to_yolov5(
            info_dict,
            class_name_to_id_mapping=class_name_to_id_mapping,
            sav_f=save_path_generated_pics,
        )

    train_images, val_images, train_annotations, val_annotations = train_test_split(
        images, annotationstxt, test_size=0.2, random_state=1
    )
    val_images, test_images, val_annotations, test_annotations = train_test_split(
        val_images, val_annotations, test_size=0.5, random_state=1
    )

    train_images_path = move_files_to_folder(
        train_images, "images/train", generated_pic_folder
    )
    val_images_path = move_files_to_folder(
        val_images, "images/val/", generated_pic_folder
    )
    test_images_path = move_files_to_folder(
        test_images, "images/test/", generated_pic_folder
    )
    move_files_to_folder(train_annotations, "labels/train/", generated_pic_folder)
    move_files_to_folder(val_annotations, "labels/val/", generated_pic_folder)
    move_files_to_folder(test_annotations, "labels/test/", generated_pic_folder)

    fileinfosmodel = fr"""
    train: {train_images_path} 
    val:  {val_images_path} 
    test: {test_images_path} 

    # number of classes
    nc: {len(mapping_dict_for_classes)}

    # class names
    names: {repr(list(mapping_dict_for_classes.keys())).replace("'", '"')}
    """
    with open(yolovyamel, encoding="utf-8", mode="w") as f:
        f.write(fileinfosmodel)

    for file in annotations:
        try:
            os.remove(file)
        except Exception:
            continue
    for file in images:
        try:
            os.remove(file)
        except Exception:
            continue
    for file in annotationstxt:
        try:
            os.remove(file)
        except Exception:
            continue

def print_config_file():
    r"""

    Example of a config file;

    [GENERAL]
    image_background_folder="C:\Users\Gamer\anaconda3\envs\dfdir\alltrainingdata"
    image_button_folder = "C:\yolovtest\buttonimages"
    save_path_generated_pics = "C:\trainingset\generated_pics"
    save_path_generated_pics_separate = "C:\trainingset\generated_pics_sep"
    maximum_buttons_on_pic=3
    number_of_pictures_to_generate=100
    max_overlapping_avoid=50000
    yaml_file="royal_halloween.yaml"

    [TRAINING]  #not necessary for generating training data
    model_file ='yolov5s.yaml'
    personal_yaml="royal_halloween.yaml"
    hypfile = "hyp.scratch-low.yaml"
    resolutionsize=640
    batch=30
    epochs=4
    ptfile="C:\Users\Gamer\anaconda3\envs\dfdir\yolov5\yolov5s.pt"
    workers=4
    generated_pic_folder = "C:\trainingset\generated_pics_sep"
    name_for_set = "royal_halloweennew"
    save_period=10


    [BUTTON0]
    class_name="play_apple_game"
    allowed_min_distance_from_zero_x=1
    allowed_min_distance_from_zero_y=1
    allowed_max_distance_from_zero_x=70
    allowed_max_distance_from_zero_y=70
    max_x=25
    max_y=25
    min_x=15
    min_y=15
    transparency_min=1
    transparency_max=50
    max_negativ_rotation=-10
    max_positiv_rotation=10
    add_pixelboarder=1
    add_pixelboarder_percentage=10
    unsharp_border=1
    unsharp_border_percentage=10
    random_crop=1
    random_crop_percentage=30
    random_crop_min=0
    random_crop_max=2
    random_blur=1
    random_blur_percentage=20
    random_blur_min=0.001
    random_blur_max=0.05


    [BUTTON1]
    class_name="won_apples"
    allowed_min_distance_from_zero_x=1
    allowed_min_distance_from_zero_y=1
    allowed_max_distance_from_zero_x=70
    allowed_max_distance_from_zero_y=70
    max_x=50
    max_y=50
    min_x=40
    min_y=40
    transparency_min=1
    transparency_max=10
    max_negativ_rotation=-10
    max_positiv_rotation=10
    add_pixelboarder=1
    add_pixelboarder_percentage=10
    unsharp_border=1
    unsharp_border_percentage=10
    random_crop=0
    random_crop_percentage=30
    random_crop_min=0
    random_crop_max=1
    random_blur=0
    random_blur_percentage=10
    random_blur_min=0.001
    random_blur_max=0.05


    Explanation

    Download https://github.com/ultralytics/yolov5 to /yolov5/ in your env, install all requirements
    [GENERAL] image_background_folder - folder where your background images are located
    [GENERAL] image_button_folder - folder where the buttons that you want to detect are located. Each button can have several different images. Each button's images must be in it's own folder. Folders must be consecutively numbered
    [GENERAL] save_path_generated_pics - Temp folder for generated files
    [GENERAL] save_path_generated_pics_separate - Finished generated training data
    [GENERAL] maximum_buttons_on_pic - Max number of random buttons on a generated image
    [GENERAL] number_of_pictures_to_generate - Total number of training images
    [GENERAL] max_overlapping_avoid - Number of times to try to not overlap buttons (if maximum_buttons_on_pic > 1)
    [GENERAL] yaml_file - choose a filename with ending '.yaml', e.g. 'mygeneratedfiles.yaml'
    [TRAINING] model_file - One of https://github.com/ultralytics/yolov5#pretrained-checkpoints  , you might have to download them and put them into the yolov5 folder, maybe they get downloaded automatically
    [TRAINING] personal_yaml - copy what you wrote in [GENERAL] yaml_file
    [TRAINING] hypfile - I usually use "hyp.scratch-low.yaml" - check out the official documentation: https://github.com/ultralytics/yolov5
    [TRAINING] resolutionsize - Use 640, I haven't tested it with other values
    [TRAINING] batch - I use 30 with a RTX 2060 8 GB
    [TRAINING] epochs - 100 is good to start with - check out the official documentation: https://github.com/ultralytics/yolov5
    [TRAINING] ptfile - I start new models with yolov5s.pt and use later my own pretrained files - check out: https://github.com/ultralytics/yolov5#pretrained-checkpoints
    [TRAINING] workers - number of CPUs to use
    [TRAINING] generated_pic_folder - copy what you wrote in [GENERAL] save_path_generated_pics_separate
    [TRAINING] name_for_set - choose name for the set
    [BUTTON0] - each button must have its own section named BUTTON + next consecutively number
    [BUTTON0] class_name - choose a unique class name
    [BUTTON0] allowed_min_distance_from_zero_x - the minimum x distance in percent that the button can show up on the picture
    [BUTTON0] allowed_min_distance_from_zero_y - the minimum y distance in percent that the button can show up on the picture
    [BUTTON0] allowed_max_distance_from_zero_x - the maximum x distance in percent that the button can show up on the picture
    [BUTTON0] allowed_max_distance_from_zero_y - the maximum y distance in percent that the button can show up on the picture
    [BUTTON0] max_x - the max x size of the button in percent relative to the background picture, e.g. if you put 10 and your image has a width of 640, the max x size is 64
    [BUTTON0] max_y - the max y size of the button in percent relative to the background picture
    [BUTTON0] min_x - the min x size of the button in percent relative to the background picture, e.g. if you put 10 and your image has a width of 640, the min x size is 64
    [BUTTON0] min_y - the min y size of the button in percent relative to the background picture
    [BUTTON0] transparency_min  for random transparency, value will be substracted from alpha channel
    [BUTTON0] transparency_max - for random transparency, value will be substracted from alpha channel
    [BUTTON0] max_negativ_rotation - degrees to rotate button for random.randrange
    [BUTTON0] max_positiv_rotation - degrees to rotate button for random.randrange
    [BUTTON0] add_pixelboarder - 1 to enable, 0 to disable (for fuzzy border)
    [BUTTON0] add_pixelboarder_percentage - percentage of images to add the fuzzy border to
    [BUTTON0] unsharp_border - how many percent of the picture should become a fuzzy border (minimum)
    [BUTTON0] unsharp_border_percentage - how many percent of the picture should become a fuzzy border (maximum)
    [BUTTON0] random_crop - 1 to enable, 0 to disable, don't disable it for now, might cause problems
    [BUTTON0] random_crop_percentage - percentage of all images the crop should be applied to
    [BUTTON0] random_crop_min - how many pixels should be cropped (minimum)
    [BUTTON0] random_crop_max - how many pixels should be cropped (maximum)
    [BUTTON0] random_blur - 1 to enable, 0 to disable
    [BUTTON0] random_blur_percentage - percentage of all images the blur should be applied to
    [BUTTON0] random_blur_min - 0.001
    [BUTTON0] random_blur_max - 0.005

    """

if __name__ == "__main__":
    try:
        generate_training_data(config_file=sys.argv[1])
    except Exception:
        print('There is something wrong! Is your config file okay?')
        print_config_file()
