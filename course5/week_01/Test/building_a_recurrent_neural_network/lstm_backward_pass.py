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

    # Compute gates related derivatives, you can find their values can be found by looking carefully at equations (7) to (10) (≈4 lines)
    dot = None
    dcct = None
    dit = None
    dft = None

    # Code equations (7) to (10) (≈4 lines)
    dit = None
    dft = None
    dot = None
    dcct = None

    # Compute parameters related derivatives. Use equations (11)-(14) (≈8 lines)
    dWf = None
    dWi = None
    dWc = None
    dWo = None
    dbf = None
    dbi = None
    dbc = None
    dbo = None

    # Compute derivatives w.r.t previous hidden state, previous memory state and input. Use equations (15)-(17). (≈3 lines)
    da_prev = None
    dc_prev = None
    dxt = None
    ### END CODE HERE ###

    # Save gradients in dictionary
    gradients = {"dxt": dxt, "da_prev": da_prev, "dc_prev": dc_prev, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients


def lstm_backward(da, caches):
    """
    Implement the backward pass for the RNN with LSTM-cell (over a whole sequence).

    Arguments:
    da -- Gradients w.r.t the hidden states, numpy-array of shape (n_a, m, T_x)
    dc -- Gradients w.r.t the memory states, numpy-array of shape (n_a, m, T_x)
    caches -- cache storing information from the forward pass (lstm_forward)

    Returns:
    gradients -- python dictionary containing:
                        dx -- Gradient of inputs, of shape (n_x, m, T_x)
                        da0 -- Gradient w.r.t. the previous hidden state, numpy array of shape (n_a, m)
                        dWf -- Gradient w.r.t. the weight matrix of the forget gate, numpy array of shape (n_a, n_a + n_x)
                        dWi -- Gradient w.r.t. the weight matrix of the update gate, numpy array of shape (n_a, n_a + n_x)
                        dWc -- Gradient w.r.t. the weight matrix of the memory gate, numpy array of shape (n_a, n_a + n_x)
                        dWo -- Gradient w.r.t. the weight matrix of the save gate, numpy array of shape (n_a, n_a + n_x)
                        dbf -- Gradient w.r.t. biases of the forget gate, of shape (n_a, 1)
                        dbi -- Gradient w.r.t. biases of the update gate, of shape (n_a, 1)
                        dbc -- Gradient w.r.t. biases of the memory gate, of shape (n_a, 1)
                        dbo -- Gradient w.r.t. biases of the save gate, of shape (n_a, 1)
    """

    # Retrieve values from the first cache (t=1) of caches.
    (caches, x) = caches
    (a1, c1, a0, c0, f1, i1, cc1, o1, x1, parameters) = caches[0]

    ### START CODE HERE ###
    # Retrieve dimensions from da's and x1's shapes (≈2 lines)
    n_a, m, T_x = da.shape
    n_x, m = x1.shape

    # initialize the gradients with the right sizes (≈12 lines)
    dx = np.zeros((n_x, m, T_x))
    da0 = np.zeros((n_a, m))
    da_prevt = None
    dc_prevt = None
    dWf = np.zeros((n_a, n_a + n_x))
    dWi = np.zeros((n_a, n_a + n_x))
    dWc = np.zeros((n_a, n_a + n_x))
    dWo = np.zeros((n_a, n_a + n_x))
    dbf = np.zeros((n_a, 1))
    dbi = np.zeros((n_a, 1))
    dbc = np.zeros((n_a, 1))
    dbo = np.zeros((n_a, 1))

    # loop back over the whole sequence
    for t in reversed(range(T_x)):
        # Compute all gradients using lstm_cell_backward
        gradients = lstm_cell_forward(da_prevt, dc_prevt, caches[t])
        # Store or add the gradient to the parameters' previous step's gradient
        dx[:, :, t] = gradients["dxt"]
        dWf = gradients["dWf"]
        dWi = gradients["dWi"]
        dWc = gradients["dWc"]
        dWo = gradients["dWo"]
        dbf = gradients["dbf"]
        dbi = gradients["dbi"]
        dbc = gradients["dbc"]
        dbo = gradients["dbo"]
    # Set the first activation's gradient to the backpropagated gradient da_prev.
    da0 = da_prevt

    ### END CODE HERE ###

    # Store the gradients in a python dictionary
    gradients = {"dx": dx, "da0": da0, "dWf": dWf, "dbf": dbf, "dWi": dWi, "dbi": dbi,
                 "dWc": dWc, "dbc": dbc, "dWo": dWo, "dbo": dbo}

    return gradients


if __name__ == '__main__':
    np.random.seed(1)
    xt = np.random.randn(3, 10)
    a_prev = np.random.randn(5, 10)
    c_prev = np.random.randn(5, 10)
    Wf = np.random.randn(5, 5 + 3)
    bf = np.random.randn(5, 1)
    Wi = np.random.randn(5, 5 + 3)
    bi = np.random.randn(5, 1)
    Wo = np.random.randn(5, 5 + 3)
    bo = np.random.randn(5, 1)
    Wc = np.random.randn(5, 5 + 3)
    bc = np.random.randn(5, 1)
    Wy = np.random.randn(2, 5)
    by = np.random.randn(2, 1)

    parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

    a_next, c_next, yt, cache = lstm_cell_forward(xt, a_prev, c_prev, parameters)

    da_next = np.random.randn(5, 10)
    dc_next = np.random.randn(5, 10)
    gradients = lstm_cell_backward(da_next, dc_next, cache)
    # print("gradients[\"dxt\"][1][2] =", gradients["dxt"][1][2])
    # print("gradients[\"dxt\"].shape =", gradients["dxt"].shape)
    # print("gradients[\"da_prev\"][2][3] =", gradients["da_prev"][2][3])
    # print("gradients[\"da_prev\"].shape =", gradients["da_prev"].shape)
    # print("gradients[\"dc_prev\"][2][3] =", gradients["dc_prev"][2][3])
    # print("gradients[\"dc_prev\"].shape =", gradients["dc_prev"].shape)
    # print("gradients[\"dWf\"][3][1] =", gradients["dWf"][3][1])
    # print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)
    # print("gradients[\"dWi\"][1][2] =", gradients["dWi"][1][2])
    # print("gradients[\"dWi\"].shape =", gradients["dWi"].shape)
    # print("gradients[\"dWc\"][3][1] =", gradients["dWc"][3][1])
    # print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
    # print("gradients[\"dWo\"][1][2] =", gradients["dWo"][1][2])
    # print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)
    print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
    print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
    print("gradients[\"dbi\"][4] =", gradients["dbi"][4])
    print("gradients[\"dbi\"].shape =", gradients["dbi"].shape)
    print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
    print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
    print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
    print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)

    x = np.random.randn(3, 10, 7)
    a0 = np.random.randn(5, 10)
    Wf = np.random.randn(5, 5 + 3)
    bf = np.random.randn(5, 1)
    Wi = np.random.randn(5, 5 + 3)
    bi = np.random.randn(5, 1)
    Wo = np.random.randn(5, 5 + 3)
    bo = np.random.randn(5, 1)
    Wc = np.random.randn(5, 5 + 3)
    bc = np.random.randn(5, 1)

    parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}

    a, y, c, caches = lstm_forward(x, a0, parameters)

    da = np.random.randn(5, 10, 4)
    gradients = lstm_backward(da, caches)

    print("gradients[\"dx\"][1][2] =", gradients["dx"][1][2])
    print("gradients[\"dx\"].shape =", gradients["dx"].shape)
    print("gradients[\"da0\"][2][3] =", gradients["da0"][2][3])
    print("gradients[\"da0\"].shape =", gradients["da0"].shape)
    print("gradients[\"dWf\"][3][1] =", gradients["dWf"][3][1])
    print("gradients[\"dWf\"].shape =", gradients["dWf"].shape)
    print("gradients[\"dWi\"][1][2] =", gradients["dWi"][1][2])
    print("gradients[\"dWi\"].shape =", gradients["dWi"].shape)
    print("gradients[\"dWc\"][3][1] =", gradients["dWc"][3][1])
    print("gradients[\"dWc\"].shape =", gradients["dWc"].shape)
    print("gradients[\"dWo\"][1][2] =", gradients["dWo"][1][2])
    print("gradients[\"dWo\"].shape =", gradients["dWo"].shape)
    print("gradients[\"dbf\"][4] =", gradients["dbf"][4])
    print("gradients[\"dbf\"].shape =", gradients["dbf"].shape)
    print("gradients[\"dbi\"][4] =", gradients["dbi"][4])
    print("gradients[\"dbi\"].shape =", gradients["dbi"].shape)
    print("gradients[\"dbc\"][4] =", gradients["dbc"][4])
    print("gradients[\"dbc\"].shape =", gradients["dbc"].shape)
    print("gradients[\"dbo\"][4] =", gradients["dbo"][4])
    print("gradients[\"dbo\"].shape =", gradients["dbo"].shape)
