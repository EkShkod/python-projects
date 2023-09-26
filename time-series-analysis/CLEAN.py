import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def clean(f, t, dt, q, g, delta, chi=4):
    """ Operates CLEAN algorithm for even or uneven time series. 
    	See Roberst et al, 1987 for detailed description.
        f: 	list or one-dimensional numpy array
        	time series values
        t: 	list or one-dimensional numpy array
        	time points
        dt:	list or one-dimensional numpy array
        	time steps
        q:	float
        	critical value (false alarm probability)
        g:	float, [0:1]
        	specifies harmonic signal in 'dirty' spectrum
        delta: 	str, {‘min’, ‘mean’}
        	specifies way to compute upper limit on frequency interval
        chi: 	float, default = 4
        	parameter for estrimating edges of arrays of spectral estimates
        	value 4 is set according to Roberst et al, 1987. """

    N = len(t)
    plt.figure(figsize=(12, 4))
    plt.scatter(t, f, color='white', linewidth=0.7, edgecolors='black')
    plt.xlabel('Time, s', fontsize=10)
    plt.ylabel('a - Given time series', fontsize=10)
    plt.xlim([0, t[-1]])
    plt.show()
    
    # 1-2. Minimal distance between data points and upper limit on frequency interval
    if delta == 'min':
        v_max = 1/(2*min(dt))
    elif delta == 'mean':
        v_max = 1/(2*np.mean(dt))
    else:
        print('Parameter v_max can not be defined correctly. Please set delta = "min" or "mean"')
        return

    # 3. Parameter m, which is used to calculate edges of arrays of spectral estimates
    m = int(chi * v_max * (t[-1]-t[0]))

    # 4. "Dirty" spectrum
    j_1 = np.arange(-m, m+1, 1)
    v_1 = np.zeros(len(j_1))
    D = np.zeros(len(j_1), dtype=complex)
    for j in j_1:
        v_1[j] = (j/m)*v_max
        for k in range(N):
            D[j] += (1 / N) * f[k] * np.exp(-(0+1j) * 2*np.pi * v_1[j] * t[k])

    # 5. Window function
    j_2 = np.arange(-2*m, 2*m+1, 1)
    v_2 = np.zeros(len(j_2))
    W = np.zeros(len(j_2), dtype=complex)
    for j in j_2:
        v_2[j] = (j/m)*v_max
        for k in range(N):
            W[j] += (1 / N) * np.exp(-(0+1j) * 2*np.pi * v_2[j] * t[k])

    plt.figure(figsize=(12, 4))
    plt.plot(v_2[:len(v_2) // 4], np.power(np.abs(W), 2)[:len(v_2) // 4], color='black', linewidth=0.7)
    plt.xlabel('Frequency, 1/s', fontsize=10)
    plt.ylabel('b - Spectral window', fontsize=10)
    # plt.xlim([0, 0.5])
    plt.ylim(0)
    plt.show()
    file.write('v_2: ' + str(v_2[:len(v_2) // 4]) + '\n')
    file.write('W: ' + str(np.power(np.abs(W), 2)[:len(v_2) // 4]) + '\n')

    # 6. Zeros of superresolution spectrum
    C = np.zeros(len(j_1), dtype=complex)

    # 7. Threshold on detection of a signal in noise
    X_q = -np.log(q)
    D_mean = 0
    D_abs = np.power(np.abs(D), 2)
    for j in range(m+1):
        D_mean += (1/(m+1)) * D_abs[j]
    D_q = D_mean * X_q
    D_q_arr = np.ones(len(v_1)) * D_q

    plt.figure(figsize=(12, 4))
    plt.plot(v_1[:len(v_1) // 2], np.power(np.abs(D), 2)[:len(v_1) // 2], color='black', linewidth=0.7)
    plt.plot(v_1[:len(v_1) // 2], D_q_arr[:len(v_1) // 2], '--', color='black', linewidth=0.7)
    plt.xlabel('Frequency, 1/s', fontsize=10)
    plt.ylabel('c - "Dirty" periodogram', fontsize=10)
    # plt.xlim([0, 0.5])
    plt.ylim(0)
    plt.show()
    file.write('v_1: ' + str(v_1[:len(v_1) // 2]) + '\n')
    file.write('D: ' + str(np.power(np.abs(D), 2)[:len(v_1) // 2]) + '\n')

    # 8. Iteratively changing dirty spectrum array. Computation cycle starts here
    i = 1

    # 9. Maximum count D_max of D and its number J
    J = np.argmax(D_abs)
    D_max = D_abs[J]

    # 10. If D_max < D_q - move to step 16. If not - move to step 11.
    if D_max > D_q:

        while D_max > D_q:
            # 11. Complex amplitide
            a = (D[J] - D[J] * W[2*J]) / (1 - np.power(np.abs(W[2*J]), 2))

            # 12. Abstracting corresponding harmonic from a "dirty" spectrum
            for j in j_1:
                D[j] = D[j] - g * (a*W[j-J] + np.conj(a)*W[j+J])

            # 13. Writing contribution of abstracted harmonic in array C
            C[J] += g * a
            C[-J] += g * np.conj(a)

            # 14. Maximum count D_max of D_i and its number J
            D_abs = np.power(np.abs(D), 2)
            J = np.argmax(D_abs)
            D_max = D_abs[J]

            # 15. Moving to step 10
            i += 1

    # 16. Moving to step 17 or raising a warning
    else:
        if i > 1:
            pass
        else:
            print('This time sequence has no harmonic components with probability 1-q')


    # 17. "Clear" spectral window with even time points
    B = np.zeros(len(j_2), dtype=complex)
    t_e = np.zeros(len(dt))
    for j in j_2:
        for k in range(N):
            t_e[k] = (t[-1] - t[0]) * k / (N-1)
            B[j] += (1 / N) * np.exp(-(0+1j) * 2*np.pi * v_2[j] * t_e[k])

    # 18. "Clear" spectrum
    S = np.zeros(len(j_1), dtype=complex)
    for j in j_1:
        for k in j_1:
            S[j] += C[k]*B[j-k]

    plt.figure(figsize=(12, 4))
    plt.plot(v_1[:len(v_1) // 2], np.power(np.abs(S), 2)[:len(v_1) // 2], color='black', linewidth=0.7)
    plt.plot(v_1[:len(v_1)//2], D_q_arr[:len(v_1)//2], '--', color='black', linewidth=0.7)
    plt.xlabel('Frequency, 1/s', fontsize=10)
    plt.ylabel('d - Clear spectrum', fontsize=10)
    # plt.xlim([0, 0.5])
    plt.ylim(0)
    plt.show()
    file.write('S: ' + str(np.power(np.abs(S), 2)[:len(v_1) // 2]) + '\n')

    # 19. Correlogram (shifted estimation of correlation function)
    Corr = np.zeros(N, dtype=complex)
    f_e = np.zeros(N, dtype=complex)
    for k in range(N):
        for j in j_1:
            Corr[k] += (N/m) * np.power(np.abs(S[j]), 2) * np.exp((0+1j) * 2*np.pi * v_1[j] * t_e[k])

    # 20. Time series on even time points grid
            f_e[k] += (N/m) * S[j] * np.exp((0+1j) * 2*np.pi * v_1[j] * t_e[k])

    plt.figure(figsize=(12, 4))
    plt.plot(t_e, Corr, color='black', linewidth=0.7)
    plt.xlabel('Time, s', fontsize=10)
    plt.ylabel('e - Correlogram', fontsize=10)
    plt.xlim([0, t_e[-1]])
    plt.show()
    file.write('t_e: ' + str(t) + '\n')
    file.write('Corr: ' + str(Corr) + '\n')

    plt.figure(figsize=(12, 4))
    plt.plot(t_e, f_e, color='black', linewidth=0.7)
    plt.scatter(t, f, color='white', linewidth=0.7, edgecolors='black')
    plt.xlabel('Time, s', fontsize=10)
    plt.ylabel('f - Even time series', fontsize=10)
    plt.xlim([0, t_e[-1]])
    plt.show()
    file.write('f_e: ' + str(f_e) + '\n')

    # 21. End of the algorithm
    data = {'Time, t': t, 'Series, f': f,
            'Frequency -m...0...m, v_1': v_1, '"Dirty" periodogram, D': np.power(np.abs(D), 2),
            'Frequency -2m...0...2m, v_2': v_2, 'Window function, W': np.power(np.abs(W), 2),
            'Clean periodogram, S': np.power(np.abs(S), 2),
            'Even time, t_e': t_e, 'Even series, f_e': f_e, 'Correlogram, Corr': Corr}
    df = pd.DataFrame(data.values(), data.keys()).T
    df.to_csv('result.csv', index=False)
    file.close()
    return
    
