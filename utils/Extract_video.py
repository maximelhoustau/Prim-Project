from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-d", "--video_destination", help="path to the video file extracted")
ap.add_argument("-s", "--start",type=int, default=0, help="Time to start in minutes")
ap.add_argument("-e", "--end", type=int, help="Time to end extraction in minutes")

args = vars(ap.parse_args())

video_path = "../videos/"+args["video"]
video_path_dest = "../images/"+args["video_destination"]
start_time = args["start"]
end_time = args["end"]

start_time = start_time*60
end_time = end_time*60

ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=video_path_dest)
