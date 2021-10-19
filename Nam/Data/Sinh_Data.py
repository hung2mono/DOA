import numpy as np

def generate_target_spectrum(DOA, doa_min, grid, NUM_GRID):
    K = len(DOA)
    target_vector = 0
    for ki in range(K):
        doa_i = DOA[ki]
        target_vector_i_ = []
        grid_idx = 0
        while grid_idx < NUM_GRID:
            grid_pre = doa_min + grid * grid_idx
            grid_post = doa_min + grid * (grid_idx + 1)
            if grid_pre <= doa_i and grid_post > doa_i:
                expand_vec = np.array([grid_post - doa_i, doa_i - grid_pre]) / grid
                grid_idx += 2
            else:
                expand_vec = np.array([0.0])
                grid_idx += 1
            target_vector_i_.extend(expand_vec)
        if len(target_vector_i_) >= NUM_GRID:
            target_vector_i = target_vector_i_[:NUM_GRID]
        else:
            expand_vec = np.zeros(NUM_GRID - len(target_vector_i_))
            target_vector_i = target_vector_i_
            target_vector_i.extend(expand_vec)
        target_vector += np.asarray(target_vector_i)
    # target_vector /= K

    return target_vector

def generate_training_data_ss_AI(M, N, K, d, wavelength, SNR, doa_min, doa_max, step, doa_delta, NUM_REPEAT_SS, grid_ss,
                                 NUM_GRID_SS):
    data_train_ss = {}
    data = []
    data_train_ss['input_nf'] = []
    data_train_ss['input'] = []
    data_train_ss['target_spec'] = []
    for delta_idx in range(len(doa_delta)):
        delta_curr = doa_delta[delta_idx]  # inter-signal direction differences
        delta_cum_seq_ = [delta_curr]  # doa differences w.r.t first signal
        delta_cum_seq = np.concatenate([[0], delta_cum_seq_])  # the first signal included
        delta_sum = np.sum(delta_curr)  # direction difference between first and last signals
        NUM_STEP = int((doa_max - doa_min - delta_sum) / step)  # number of scanning steps

        for step_idx in range(NUM_STEP):
            doa_first = doa_min + step * step_idx
            DOA = delta_cum_seq + doa_first

            for rep_idx in range(NUM_REPEAT_SS):
                add_noise = np.random.randn(M, N) + 1j * np.random.randn(M, N)
                array_signal = 0
                for ki in range(K):
                    signal_i = 10 ** (SNR[ki] / 20) * (np.random.randn(1, N) + 1j * np.random.randn(1, N))
                    # phase_shift_unit = 2 * np.pi * d / wavelength * np.sin(DOA / 180 * np.pi)
                    array_geom = np.expand_dims(np.array(np.arange(M)), axis=-1) * d
                    phase_shift_array = 2 * np.pi * array_geom / wavelength * np.sin(DOA[ki] / 180 * np.pi)
                    a_i = np.cos(phase_shift_array) + 1j * np.sin(phase_shift_array)
                    array_signal_i = np.matmul(a_i, signal_i)
                    array_signal += array_signal_i

                array_output_nf = array_signal + 0 * add_noise  # noise-free output
                array_output = array_signal + 1 * add_noise

                array_covariance_nf = 1 / N * (np.matmul(array_output_nf, np.matrix.getH(array_output_nf)))
                array_covariance = 1 / N * (np.matmul(array_output, np.matrix.getH(array_output)))
                cov_vector_nf_ = []
                cov_vector_ = []
                for row_idx in range(M):
                    cov_vector_nf_.extend(array_covariance_nf[row_idx, (row_idx + 1):])
                    cov_vector_.extend(array_covariance[row_idx, (row_idx + 1):])
                cov_vector_nf_ = np.asarray(cov_vector_nf_)
                cov_vector_nf_ext = np.concatenate([cov_vector_nf_.real, cov_vector_nf_.imag])
                cov_vector_nf = 1 / np.linalg.norm(cov_vector_nf_ext) * cov_vector_nf_ext
                data_train_ss['input_nf'].append(cov_vector_nf)
                cov_vector_ = np.asarray(cov_vector_)
                cov_vector_ext = np.concatenate([cov_vector_.real, cov_vector_.imag])
                cov_vector = 1 / np.linalg.norm(cov_vector_ext) * cov_vector_ext
                data_train_ss['input'].append(cov_vector)
                # construct spatial spectrum target
                target_spectrum = generate_target_spectrum(DOA, doa_min, grid_ss, NUM_GRID_SS)
                data_train_ss['target_spec'].append(target_spectrum)
                cov_vector = np.concatenate([cov_vector, target_spectrum])
                data.append(cov_vector)
    print(np.shape(data))
    np.savetxt('data',data)
    return data_train_ss