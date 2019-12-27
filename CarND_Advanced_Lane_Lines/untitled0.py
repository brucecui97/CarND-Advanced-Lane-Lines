from moviepy.editor import VideoFileClip
from IPython.display import HTML
vid_output = 'project_video_out.mp4'
# To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
# To do so add .subclip(start_second,end_second) to the end of the line below
# Where start_second and end_second are integer values representing the start and end of the subclip
# You may also uncomment the following line for a subclip of the first 5 seconds

clip1 = VideoFileClip("project_video.mp4")
# NOTE: this function expects color images!!
white_clip = clip1.fl_image(process_image)
# %time white_clip.write_videofile(vid_output, audio=False)

clip2 = VideoFileClip("project_video.mp4").subclip(39, 42)
# NOTE: this function expects color images!!
white_clip = clip2.fl_image(process_image)
# %time white_clip.write_videofile("projecet_out_snip.mp4", audio=False)
