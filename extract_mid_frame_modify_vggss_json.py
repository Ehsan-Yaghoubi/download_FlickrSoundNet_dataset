import json
import glob
import os
import subprocess as sp
import sys
import numpy as np
import cv2


class vggss:

    def __init__(self, json_dir):
        self.json_dir = json_dir
        with open(self.json_dir, 'r') as vggss:
            self.vggss_dicts = json.load(vggss)  # list of dicts that have the annotations inside

    def names(self):
        return [dict['file'][:-7] for dict in self.vggss_dicts]

    def annot(self, name):
        for dict in self.vggss_dicts:
            if name in list(dict.values())[0][:11]:
                return dict
        return None

    def annot_show(self, annots, path_jpg):
        """
        # [In this link](https://www.robots.ox.ac.uk/~vgg/research/lvs/) the authors have mentioned that the annotations
         are as: `Video clip name, class, bounding box labels : {Xmin, Ymin, Xmax, Ymax}. `
        :param annots: dictionary
        :param path_to_img: path to the annotated frame
        :return:
        """
        basename = annots['file'][0:11]
        clas = annots['class']
        bbox = annots['bbox']

        #path_jpg = glob.glob(os.path.join(path_to_img, '{}_*[0-9]*.jpg'.format(basename)))[0]
        img = cv2.imread(path_jpg)
        if img is None:
            sys.exit('could not read/find the image')

        for index, bb in enumerate(bbox):
            h, w, c = np.shape(img)
            bbox1 = bb[0]*w, bb[1]*h, bb[2]*w, bb[3]*h

            bbox = []
            for pt in bbox1:
                if pt<0:
                    pt=0
                bbox.append(int(pt))

            point1 = bbox[0], bbox[1]
            point2 = bbox[2], bbox[3]
            print('img shape: ', w, h, c)
            print('annotation class: ', clas)
            print('bbox: ', bbox)
            print('-----'*10)

            cv2.rectangle(img, point1, point2, (255, 0, 0), 2)
            if len(bbox) == index+1:
                break

        cv2.imshow('display', img)
        k = cv2.waitKey(0)


def my_ffmpeg(ffmpeg_arguments):
    """
       ffmpeg guide list: https://gist.github.com/tayvano/6e2d456a9897f55025e25035478a3a50
       '-n': never overwrite output files
       '-i': Specify the input video path
       '-vf': Specify the target_frame to be extracted #  e.g., '-vf', "select=eq(n\,{})".format(target_frame),
       """

    proc = sp.Popen(ffmpeg_arguments, stdout=sp.PIPE, stderr=sp.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        print(stderr)
    else:
        print("The file is saved at " + ffmpeg_arguments[-1])


def extract_frame(ffmpeg_args, vggss_clss, show_annot):
    my_ffmpeg(ffmpeg_args)
    # show annotations
    if show_annot:
        base_name = ffmpeg_args[-1].split('/')[-1].split('_')[0]
        vggss_clss.annot_show(vggss_clss.annot(base_name), ffmpeg_args[-1])


if __name__ == '__main__':
    #input folders
    vggss_videos = '/data2/datasets/vggss/video'
    vggss_json_dir = '/data2/datasets/vggss/vggss.json'  # annotations of the vggss dataset
    #output folders
    vggss_frames_saving_path = '/data2/datasets/vggss/frames_samename'
    vggss_json_saving_path = '/data2/datasets/vggss/modified_vggss.json'
    os.makedirs(vggss_frames_saving_path, exist_ok=True)

    vggss_instance = vggss(vggss_json_dir)

    videos = os.listdir(vggss_videos)
    for index, video_name in enumerate(videos):
        video_path = os.path.join(vggss_videos, video_name)
        #video_base_name = video_name[:11]
        ffmpeg_args = ['/usr/bin/ffmpeg',
                       '-n',
                       "-loglevel",
                       "warning",
                       '-ss', '00:00:05',
                       '-i', video_path,
                       os.path.join(vggss_frames_saving_path, '{}.jpg'.format(video_name.split(".mp4")[0]))]

        extract_frame(ffmpeg_args, vggss_instance, show_annot=True)






