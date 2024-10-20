implementation example:


from spatter import *

my_filter = MySpatter()
my_filter.set_params({})
my_filter.apply_filter_video(input_path='videos/crowd_run_short_1920x1080_50.yuv', output_path='videos/tired.yuv')
