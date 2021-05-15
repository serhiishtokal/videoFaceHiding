import glob
import ntpath
from os import path, makedirs

from face_anonymize_video_to_images import anonymize_video_to_images, DetectionCreator, AnonymizeMethod

input_folder = "d:\\STUDY Univer\\PolitechnikaLubelska\\DYPLOM\\Datasets\\TikTok\\input\\"
output_folder = 'd:\\STUDY Univer\\PolitechnikaLubelska\\DYPLOM\\Datasets\\TikTok\\output\\'
videos_extension = 'mp4'
detector=DetectionCreator.ULFD(128)
anonymizer= AnonymizeMethod.PIXELATE

input_videos_paths = glob.glob(input_folder + '*.' + videos_extension)
print(input_videos_paths)

for input_video_path in input_videos_paths:
    head, tail = ntpath.split(input_video_path)
    file_name = path.splitext(tail)[0]
    output_images_folder = output_folder + f"{file_name}\\"

    if not path.exists(output_images_folder):
        makedirs(output_images_folder)
    anonymize_video_to_images(input_video_path, output_images_folder, detector, anonymizer)
