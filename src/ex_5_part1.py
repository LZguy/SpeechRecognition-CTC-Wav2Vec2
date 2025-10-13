import sys
import numpy as np


def ctc_probability(y, label, alpha_chars):
    """
    Computes P(label | x) via the CTC forward recursion.

    Parameters
    ----------
    y : np.ndarray of shape (T, K)
        The model's output probabilities over K symbols.
        We assume the last column is the blank symbol.
    label : str
        The label sequence whose probability we want, e.g. "aaabb".
    alpha_chars : str
        The 'alphabet' of real tokens, e.g. "abc".  We assume:
            index 0 -> alpha_chars[0]
            index 1 -> alpha_chars[1]
            ...
            index len(alpha_chars) -> blank
    Returns
    -------
    float
        The CTC forward probability P(label|x).
    """

    # Number of real characters in the alphabet:
    num_real_chars = len(alpha_chars)
    # The blank index is the last column (K-1):
    blank_index = num_real_chars  # i.e. if alpha_chars="abc", blank_index=3

    # -------------------------------------------------------
    # 1) Map each character in alpha_chars to a column index in y
    #    columns: 0..(num_real_chars-1) for real chars, then blank_index for blank
    # -------------------------------------------------------
    char_to_index = {}
    for i, ch in enumerate(alpha_chars):
        char_to_index[ch] = i
    # We define the blank symbol to be the last column
    char_to_index["blank"] = blank_index

    # -------------------------------------------------------
    # 2) Build the “extended” label sequence z:
    #    z = [blank, l1, blank, l2, blank, ..., ln, blank]
    # -------------------------------------------------------
    z = [blank_index]  # start with blank
    for c in label:
        z.append(char_to_index[c])  # real token
        z.append(blank_index)  # blank
    S = len(z)  # length of extended sequence
    T = y.shape[0]  # number of time steps

    # -------------------------------------------------------
    # 3) Allocate alpha (forward) matrix, shape = (S x T).
    #    alpha[s,t] = probability of seeing z[:s+1] after t+1 frames
    # -------------------------------------------------------
    alpha = np.zeros((S, T), dtype=np.float64)

    # -------------------------------------------------------
    # 4) Initialization (time t=0)
    #    alpha[0,0] = y[0, z[0]] (prob of first symbol, which is blank)
    #    alpha[1,0] = y[0, z[1]] if label is not empty
    # -------------------------------------------------------
    alpha[0, 0] = y[0, z[0]]
    if S > 1:
        alpha[1, 0] = y[0, z[1]]

    # -------------------------------------------------------
    # 5) Fill in the DP for t = 1..T-1, s = 0..S-1
    #
    #    If z_s = blank or z_s == z_{s-2}, then
    #       alpha[s,t] = (alpha[s, t-1] + alpha[s-1, t-1]) * y[t, z[s]]
    #    else:
    #       alpha[s,t] = (alpha[s, t-1] + alpha[s-1, t-1] + alpha[s-2, t-1]) * y[t, z[s]]
    #
    # -------------------------------------------------------
    for t in range(1, T):
        for s in range(S):
            emit_prob = y[t, z[s]]

            a_s = alpha[s, t - 1] if s >= 0 else 0.0
            a_s_1 = alpha[s - 1, t - 1] if s - 1 >= 0 else 0.0
            a_s_2 = alpha[s - 2, t - 1] if s - 2 >= 0 else 0.0

            # If s=0, there's no s-1 or s-2 to consider except alpha[s,t-1].
            if s == 0:
                alpha[s, t] = a_s * emit_prob
            else:
                # If z[s] is blank or if it equals z[s-2], skip the (s-2) transition.
                if z[s] == blank_index or (s >= 2 and z[s] == z[s - 2]):
                    alpha[s, t] = (a_s + a_s_1) * emit_prob
                else:
                    alpha[s, t] = (a_s + a_s_1 + a_s_2) * emit_prob

    # -------------------------------------------------------
    # 6) Final probability = alpha[S-1, T-1] + alpha[S-2, T-1]
    # -------------------------------------------------------
    if S == 1:
        # degenerate case: label was empty, or one symbol
        ctc_prob = alpha[0, T - 1]
    else:
        ctc_prob = alpha[S - 1, T - 1] + alpha[S - 2, T - 1]

    return ctc_prob


if __name__ == "__main__":
    # Command-line usage:
    #   python ex_5_part1.py /path/to/matrix.npy label_str alphabet
    #
    path_to_npy = sys.argv[1]
    label_str = sys.argv[2]  # e.g. "aaabb"
    alphabet = sys.argv[3]  # e.g. "abc"

    # 1) Load the T×K probability matrix
    y = np.load(path_to_npy)

    # 2) Compute the probability
    prob_value = ctc_probability(y, label_str, alphabet)

    # 3) Write result to out.txt, rounding to two decimals
    with open("out.txt", "w") as f:
        f.write(str(round(prob_value, 2)) + "\n")
