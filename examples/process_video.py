
from moviepy.editor import VideoFileClip
from CarND_Advanced_Lane_Lines.core import process_image
vid_output = 'examples/project_video_out.mp4'
# To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# To do so add .subclip(start_second,end_second) to the end of the line below
# Where start_second and end_second are integer values representing the start and end of the subclip
# You may also uncomment the following line for a subclip of the first 5 seconds

clip1 = VideoFileClip("examples/project_video.mp4")
# NOTE: this function expects color images!!
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(vid_output, audio=False)
