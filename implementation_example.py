from spatter import *

my_filter = MySpatter()
my_filter.set_params({})
my_filter.apply_filter_video(input_path='videos/crowd_run_short_1920x1080_50.yuv', output_path='videos/tired.yuv')

tpe_sampler = optuna.samplers.TPESampler(
        n_startup_trials= 8, # объем разведки. Всегда настраивайте!
        n_ei_candidates=15, # влияет на "точность шага"
)
study = start_optimization(partial(my_filter.get_objective, input_path='super_short.yuv', needed_psnr=33), n_trials = 150, n_jobs = -1,
                           tpe_sampler = tpe_sampler)
