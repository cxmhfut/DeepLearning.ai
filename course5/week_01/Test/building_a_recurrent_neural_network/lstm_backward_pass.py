from course5.week_01.Test.building_a_recurrent_neural_network.lstm_network import *

def lstm_cell_backward(da_next, dc_next, cache):
    """
    Implement the backward pass for the LSTM-cell (single time-step).

    Arguments:
    da_next -- Gradients of next hidden state, of shape (n_a, m)
    dc_next -- Gradients of next cell state, of shape (n_a, m)
    cache -- cache storing information from the forward pass

    Returns:
    gradients -- python dictionary containing:
                        dxt -- Gradient of input data at time-step t, of shape (n_x, m)
                        da_prev -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dc_prev -- Gradient w.r.t. the previous memory state, of shape (n_a, m, T_x)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the output gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the output gate, of shape (n_a, 1)
    """

    # Retrieve information from "cache"
    (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters) = cache

    ### START CODE HERE ###
    # Retrieve dimensions from xt's and a_next's shape (≈2 lines)
    n_x, m = xt.shape
    n_a, m = a_next.shape

    print(xt.shape)
    print(a_next.shape)

    # Compute gates related derivatives, you can find their values can be found by looking carefully at equations (7) to (10) (≈4 lines)
    dot = da_next * np.tanh(c_next) * ot * (1 - ot)
    dcct = dc_next * it + ot * (1 - np.tanh(c_next) ** 2) * it * da_next * cct * (1 - np.tanh(cct) ** 2)
    dit = dc_next * cct + ot * (1 - np.tanh(c_next) ** 2) * cct * da_next * it * (1 - it)
    dft = dc_next * c_prev + ot * (1 - np.tanh(c_next) ** 2) * c_prev * da_next * ft * (1 - ft)

    # Code equations (7) to (10) (≈4 lines)
    # dit = None
    # dft = None
    # dot = None
    # dcct = None

    # Concatenate a_prev and xt (≈3 lines)
    concat = np.zeros((n_a + n_x, m))
    concat[: n_a, :] = a_prev
    concat[n_a:, :] = xt

    print(concat.shape)
    print(dot.shape)

    # Compute parameters related derivatives. Use equations (11)-(14) (≈8 lines)
    dWf = None
    dWi = None
    dWc = None
    # dWo = dot * concat.T
    dWo = None
    dbf = np.sum(dft, keepdims=True, axis=1)
    dbi = np.sum(dit, keepdims=True, axis=1)
    dbc = np.sum(dcct, keepdims=True, axis=1)
    dbo = np.sum(dot, keepdims=True, axis=1)

    # Compute derivatives w.r.t previous hidden state, previous memory state and input. Use equations (15)-(17). (≈3 lines)
    da_prev = None
    dc_prev = None
    dxt = None
    ### END CODE HERE ###

    # Save gradients in dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients