import subprocess
import json
import sys
import numpy as np
import cv2
import multiprocessing


class VGGSS:

    def __init__(self, json_dir):
        self.json_dir = json_dir
        with open(self.json_dir, 'r') as vggss:
            self.vggss_dicts = json.load(vggss)  # list of dicts that have the annotations inside

    def basenames(self):
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

def download_video(video_id, start_time, end_time, output_path, full_name):
    command = ["ffmpeg", '-n', "-loglevel", "warning", "-i", f"https://www.youtube.com/watch?v={video_id}", "-ss", str(start_time), "-to", str(end_time), "-c", "copy", f"{output_path}{full_name}.mp4"]
    subprocess.call(command)

if __name__ == '__main__':
    vggss_path = "./vggss.json"
    output_path = "/data2/datasets/vggss/vggss_labeled/video/"
    num_processes = 12
    vggss = VGGSS(vggss_path)
    vggss_samples = vggss.basenames()

    for index, video_id in enumerate(vggss_samples):
        annotations = vggss.annot(video_id)
        full_name = annotations["file"]
        start_time = int(full_name[-6:])
        end_time = int(start_time + 10)


        processes = []
        for i in range(num_processes):
            p = multiprocessing.Process(target=download_video, args=(video_id, start_time, end_time, output_path, full_name))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()


