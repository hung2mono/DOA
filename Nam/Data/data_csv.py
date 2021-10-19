from Sinh_Data import *

# Para
# # array signal parameters
fc = 2e9  # Tần số tín hiệu đến
c = 3e8  # Vận tốc ánh sáng
M = 10  # Số phần tử anten
N = 400  # snapshot number
wavelength = c / fc  # Bước sóng
d = 0.5 * wavelength  # Khoảng cách giữa các phần tử anten
K_ss = 2  # signal number
doa_min = -60  # DOA max (degree)
doa_max = 60  # DOA min (degree)
# # spatial filter training parameters

grid_sf = 1  # DOA step (degree) for generating different scenarios
# GRID_NUM_SF = int((doa_max - doa_min) / grid_sf)
GRID_NUM_SF = int((doa_max - doa_min) / grid_sf)
SF_NUM = 6  # number of spatial filters
SF_SCOPE = (doa_max - doa_min) / SF_NUM  # spatial scope of each filter
SNR_sf = 10
NUM_REPEAT_SF = 1  # number of repeated sampling with random noise

noise_flag_sf = 1  # 0: noise-free; 1: noise-present
amp_or_phase = 0  # show filter amplitude or phase: 0-amplitude; 1-phase
NUM_GRID_SS = 121
# # training set parameters
# SS_SCOPE = SF_SCOPE / SF_NUM   # scope of signal directions
step_ss = 1  # DOA step (degree) for generating different scenarios

doa_delta = np.array(np.arange(20) + 1) * 0.1 * SF_SCOPE
# doa_delta = np.array(np.arange(122)) # inter-signal direction differences
SNR_ss = np.array([10, 10, 10]) + 0
NUM_REPEAT_SS = 100  # number of repeated sampling with random noise

noise_flag_ss = 1  # 0: noise-free; 1: noise-present

# # DNN parameters
grid_ss = 1  # inter-grid angle in spatial spectrum

input_size_ss = M * (M - 1)  # 90
batch_size_ss = 32
learning_rate_ss = 0.001
num_epoch_ss = 400
n_hidden = 256
n_classes = 121

data_train_ss = generate_training_data_ss_AI(M, N, K_ss, d, wavelength, SNR_ss, doa_min, doa_max, step_ss, doa_delta, NUM_REPEAT_SS, grid_ss,
                                 NUM_GRID_SS)