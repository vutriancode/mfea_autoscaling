import os

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))
CORE_DATA_DIR = PROJECT_DIR + '/{}'.format('data')

ENV_LIST = ['development', 'experiment']
ENV_DEFAULT = 'experiment'

ENV = ENV_DEFAULT


class Config:
    if ENV == 'development':
        DATA_EXPERIMENT = 'google_trace'  # grid, traffic, google_trace

        GOOGLE_TRACE_DATA_CONFIG = {
            'train_data_type': 'mem',  # cpu_mem, uni_mem, uni_cpu
            'predict_data': 'mem',
            'data_type': 'all_jobs',  # 1_job, all_jobs
            'time_interval': 5,
            'file_data_name': '/input_data/google_trace/{}/{}_mins.csv',
            'data_path': CORE_DATA_DIR + '{}'
        }

    elif ENV == 'experiment':
        DATA_EXPERIMENT = 'google_trace'  # grid, traffic, google_trace
        GOOGLE_TRACE_DATA_CONFIG = {
            'train_data_type': 'mem',  # cpu_mem, uni_mem, uni_cpu
            'predict_data': 'mem',
            'data_type': 'all_jobs',  # 1_job, all_jobs
            'time_interval': 5,
            'file_data_name': '/input_data/google_trace/{}/{}_mins.csv',
            'data_path': CORE_DATA_DIR + '{}'
        }
    else:
        raise EnvironmentError
