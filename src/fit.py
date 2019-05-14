from __future__ import division

import numpy as np
import plateau_function as pf

from scipy.optimize import curve_fit


def fit_connected_component(softmax_scores, n_frames, cc_start, cc_end, tau, min_w=5, min_s=0.05):
    """
    Fits the plateau function to the connected component of softmax scores p such that p > tau

    :param softmax_scores: the softmax scores, should be a 1D numpy array of scores matching the untrimmed video length
    :param n_frames: number of frames in the corresponding untrimmed video
    :param cc_start: frame index where the connected component starts in softmax_scores
    :param cc_end: frame index where the connected component ends in softmax_scores
    :param tau: the threshold used to obtain the connected component
    :param min_w: minimum w to avoid noisy fits
    :param min_s: minimum s to avoid noisy fits
    :return: (q, connected_component) where q is the fitted plateau function
    """
    x = np.arange(cc_start, cc_end + 1)
    y = softmax_scores[cc_start:cc_end + 1]
    max_idx = np.argmax(y)

    # initial parameters guess #

    # initial value for plateau centre, will be the arg max of max softmax_scores(c, connected_component)
    c_init = x[max_idx]

    # initial value for plateau width, set to the number of frames whose softmax_scores is larger than threshold.
    # / 2 because plateau width is 2*w
    w_init = np.where(y > tau)[0].size / 2

    s_init = y[max_idx]  # initial value for slope steepness, set to maximum value of the softmax score

    parameters_guess = [c_init, w_init, s_init]

    # parameter bounds
    c_bounds = [x[0], x[-1]]
    w_bounds = [0, x.size / 2]
    s_bounds = [0, 1]

    bounds = ([c_bounds[0], w_bounds[0], s_bounds[0]], [c_bounds[1], w_bounds[1], s_bounds[1]])

    try:
        fitted_parameters, _ = curve_fit(pf.PlateauFunction.function, x, y, p0=parameters_guess, bounds=bounds)

        c_fit = fitted_parameters[0]
        w_fit = fitted_parameters[1]
        s_fit = fitted_parameters[2]

        # discarding garbage fit
        if w_fit < min_w or s_fit < min_s:
            return None, None

        q = pf.PlateauFunction(c_fit, w_fit, s_fit, n_frames, generate_xy=False)
        q.is_proposal = True
    except Exception:
        return None, None

    connected_component = np.array([cc_start, cc_end])

    return q, connected_component


def fit_proposals(softmax_scores, min_tau_for_cc, max_tau_for_cc, min_cc_length, tau_step=0.1):
    """
    Produces update proposals fitting softmax scores

    :param softmax_scores: the softmax scores. Should be a (m x n) NumPy array where m is the number of classes and n is
     the number of frames in the untrimmed video
    :param min_tau_for_cc: the minimum tau to calculate the connected components of scores
    :param max_tau_for_cc: the maximum tau to calculate the connected components of scores
    :param min_cc_length: minimum connected component length to avoid noisy fits
    :param tau_step: the tau step to range from min_tau to max_tau
    :return: (proposals, connected_components). Proposals is a dictionary whose keys are class indices and values
     are list of fitted proposals. Connected_components is a dictionary whose keys are class indices and values are list
     of dictionaries containing keys `cc` (connected component) and `tau`
    """
    (n_classes, n_frames) = softmax_scores.shape

    proposals = {}
    connected_components = {}

    # for each class separately
    for class_index in range(n_classes):
        for tau in np.arange(min_tau_for_cc, max_tau_for_cc, tau_step):
            # calculating connected components where class_index is higher than tau
            scores_indices = np.where(softmax_scores[class_index] > tau)[0]
            (cc_starts, cc_ends, cc_lengths, components) = find_connected_components_in_indices(scores_indices)

            for i, cc_start in enumerate(cc_starts):
                if cc_lengths[i] < min_cc_length:
                    continue

                (q, cc) = fit_connected_component(softmax_scores[class_index], n_frames, cc_start, cc_ends[i], tau)

                if q is None:
                    continue

                q.connected_component = cc
                q.tau = tau
                q.is_proposal = True

                if class_index in proposals:
                    duplicate = False

                    for gg in proposals[class_index]:
                        if q.has_same_parameters_as(gg, eps_c=1, eps_w=1, eps_s=0.05):
                            duplicate = True
                            break

                    if not duplicate:
                        proposals[class_index].append(q)
                        connected_components[class_index].append({'cc': cc, 'tau': tau})
                else:
                    proposals[class_index] = [q]
                    connected_components[class_index] = [{'cc': cc, 'tau': tau}]

    return proposals, connected_components


def calculate_proposal_confidence(g, q, softmax_scores):
    """
    Calculates the confidence of an update proposal, given the plateau function to be updated.
    :param g: the plateau function to be updated
    :param q: the update proposal
    :param softmax_scores: the softmax scores
    :return: the confidence of the proposal
    """
    assert g.label is not None, 'No label set for this g! {}'.format(g)
    assert q.label is not None, 'No label set for this q! {}'.format(q)
    assert g.label == q.label, 'Calculating confidence between g and q with different labels! {}, {}'.format(g.label,
                                                                                                             q.label)

    g_cc = g.to_connected_component(cheap=True)['cc']
    q_cc = q.to_connected_component(cheap=True)['cc']

    mean_g = np.mean(softmax_scores[g.label, g_cc])
    mean_q = np.mean(softmax_scores[q.label, q_cc])
    confidence = mean_q - mean_g

    return confidence


def find_connected_components_in_indices(y):
    """
    Calculates connected components in an array **y** containing the indices where a condition **p** is satisfied over
    an other array **x**.

    This function returns:

    - cc_starts: array containing the indices, relative to x, where each connected component starts, that is the start
      of the connected component in x where condition p is satisfied.
    - cc_ends: same as cc_starts, but containing the end of each connected component. Ends are inclusive.
    - cc_lengths: the length of each connected component
    - connected_components: the actual connected components of y, i.e.
      connected_components[i] = y[cc_starts[i]]:y[cc_ends[i]].

    For example, let:

    - x = [0, 0, 0.1, 0.2, 0.5, 0.5, 0, 0.2, 0.2]
    - y = np.where(x > 0.1) = [3, 4, 5, 7, 8]

    Then:

    - cc_starts = [3, 7]
    - cc_ends = [5, 8]
    - cc_lengths = [3, 2]
    - connected_components = [[3, 4, 5], [7, 8]

    :param y: the array of indices. Must be a NumPy array contain monotonic integer values, with no repetitions
    :return: (cc_starts, cc_ends, cc_lengths, connected_components), all as NumPy arrays
    """
    if y.size == 0:
        cc_starts = np.array([])
        cc_ends = np.array([])
        cc_lengths = np.array([])
        connected_components = np.array([])
    else:
        assert np.unique(y).shape == y.shape, 'y contains repeated elements!'
        assert np.array_equal(np.sort(y), y), 'y is not monotonically increasing!'

        a = np.diff(y)
        b = np.where(np.append(a, np.inf) > 1)[0]
        cc_lengths = np.diff(np.insert(b, 0, -1))
        cc_ends = np.cumsum(cc_lengths) - 1
        cc_starts = np.insert(cc_ends[:-1], 0, -1) + 1

        connected_components = [y[cc_starts[i]:cc_ends[i]+1] for i, _ in enumerate(cc_starts)]
        cc_starts = y[cc_starts]
        cc_ends = y[cc_ends]

    return cc_starts, cc_ends, cc_lengths, connected_components
