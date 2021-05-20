import glob
import ntpath

from face_anonymize_video import anonymize_video, DetectionCreator, AnonymizeMethod


# detector = DetectionCreator.ULFD(input_img_width=320)
# detector = DetectionCreator.DNN_CAFFE()
# detector = DetectionCreator.CASCADE()

# anonymizationMethod = AnonymizeMethod.BLUR
# anonymizationMethod = AnonymizeMethod.PIXELATE
# anonymizationMethod = AnonymizeMethod.BlackRectangle

input_folder = "d:\\STUDY Univer\\PolitechnikaLubelska\\DYPLOM\\Datasets\\TikTok\\input\\"
output_folder = 'd:\\STUDY Univer\\PolitechnikaLubelska\\DYPLOM\\Datasets\\TikTok\\output\\'
videos_extension = 'mp4'

input_videos_paths = glob.glob(input_folder+'*.'+videos_extension)
output_video_paths = []

for input_video_path in input_videos_paths:
    head, tail = ntpath.split(input_video_path)
    output_video_path=output_folder+'out_'+tail
    anonymize_video(input_video_path, output_video_path, DetectionCreator.ULFD(320),AnonymizeMethod.PIXELATE)