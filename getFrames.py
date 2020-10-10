# 转发服务器上的12345端口数据到本地的12346端口： ssh -L 12346:127.0.0.1:12345 zh@10.21.25.237
# 本地访问：127.0.0.1:12346
# ces yixia
#
#####################
# 这是我写的！！！  #
#####################


#
import os,time
import subprocess
from moviepy.editor import VideoFileClip

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_img(videos_src_path, output_path, num_per_sec=1):
    videos = os.listdir(videos_src_path)
    # 不知道为啥， filter没法用 for 遍历，之前的OK的！！！
    # videos = filter(lambda x: x.endswith('avi'), videos)  # 暂时只处理avi
    #
    # print("videos: ", list(videos))
    videos = list(videos)
    # print(type(videos), len(videos))

    for each_video in videos:
        # each_video =  list(videos)[0]
        # each_video = videos[0]
        print("hhhhh", each_video)
        each_video_name, _ = each_video.split('.')
        os.mkdir(output_path + '/' + each_video_name)  # 为当前video创建output_path
        each_video_save_full_path = os.path.join(output_path, each_video_name)  # 生成save_path
        each_video_full_path = os.path.join(videos_src_path, each_video)  # 当前video的完整path

        start_time = '00:00:00'  #
        run_time = get_file_times(each_video_full_path)
        # num_per_sec = 20  # fps {we get is from the video}
        # os.mkdir(each_video_save_full_path)
        save_full_path = each_video_save_full_path
        comm_str = "ffmpeg " \
                   "-i {} " \
                   " -r {}  " \
                   "-ss {} " \
                   " -t {} " \
                   " -q:v 2 " \
                   " {}/%05d.jpg ".format(
            each_video_full_path, num_per_sec, start_time, run_time, save_full_path)
        print(comm_str)
        os.system(comm_str)

def get_file_times(filename):
        u"""
        获取视频时长（s:秒）
        """
        clip = VideoFileClip(filename)
        file_time = timeConvert(clip.duration)
        return file_time

def timeConvert(size):# 单位换算
        M, H = 60, 60**2
        if size < M:
            return ('00:00:{}').format(size)
        if size < H:
            return ('00:{}:{}').format( int(size/M) ,int(size%M) )
        else:
            hour = int(size/H)
            mine = int(size%H/M)
            second = int(size%H%M)
            tim_srt = ('{}:{}:{}').format(hour, mine, second)
            return tim_srt

def test_get_file_times():
    pwd = os.path.join( os.getcwd(), 'video_src' )
    filename = 'test.mp4'
    path = os.path.join(pwd, filename)
    print(  get_file_times(path) )

    filename = 'test_out.mp4'
    path = os.path.join(pwd, filename)
    print(  get_file_times(path) )


def test_get_img():
    videos_src_path = "/home/zh/Papers_Code/ano_pred_cvpr2018_sist/" \
                      "ano_pred_cvpr2018/Data/shanghaitech/training/frames"
    output_path = "/home/zh/Papers_Code/ano_pred_cvpr2018_sist/" \
                  "ano_pred_cvpr2018/Data/shanghaitech/training/frames"
    num_per_sec = 50
    get_img(videos_src_path, output_path, num_per_sec)

if __name__ == '__main__':
    # 内测：

    # test_get_file_times()
    # test_get_file_times()

    videos_src_path = "/home/zh/Papers_Code/CVPR2019_pytorch_VAD/novelty-detection/data/shanghaitech/training/videos"
    output_path = "/home/zh/Papers_Code/CVPR2019_pytorch_VAD/novelty-detection/data/shanghaitech/training/frames"
    # 注：由于函数写法原因，一定要先在正确路径手动创建 frames目录
    num_per_sec = 24 # 直接使用 ffmpeg -i xxx.avi 肉眼查看的，后面考虑用函数读
    get_img(videos_src_path, output_path, num_per_sec)

