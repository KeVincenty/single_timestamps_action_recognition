import numpy as np
import pickle

import fit
import tqdm
import sys

import single_timestamps


def update_plateaus(plateaus_per_video, softmax_scores, update_proposals_per_video, settings, update_info):
    """
    Updates plateaus

    :param plateaus_per_video: dictionary whose keys are video ids and values are list of plateau objects
    :param softmax_scores: dictionary whose keys are video ids and values are softmax scores
    :param update_proposals_per_video: dictionary whose keys are video ids and values are list of update proposals
    :param settings: settings dictionary
    :param update_info: dictionary storing update info
    :return: number of plateaus that have been updated
    """
    all_plateaus = single_timestamps.get_all_plateaus_in_dataset(plateaus_per_video)

    update_candidates = []
    all_proposals = {}
    save_all_proposals = settings['save_all_proposals'] if 'save_all_proposals' in settings else True

    bar = tqdm.tqdm(desc='-> Assigning update proposals...', total=len(all_plateaus), file=sys.stdout)

    # match proposals to the corresponding plateaus
    for g_id, g in all_plateaus.items():
        matching_proposals = []

        if g.label in update_proposals_per_video[g.video]:
            # label and index will be set for all q in matching_proposals
            matching_proposals = g.match_to_proposals(update_proposals_per_video[g.video][g.label])

        if matching_proposals:
            for mp in matching_proposals:
                # setting the proposal confidence
                confidence = fit.calculate_proposal_confidence(g, mp, softmax_scores[g.video])
                mp.set_confidence(confidence)

            all_proposals[g_id] = matching_proposals

        if g.video not in update_info:
            update_info[g.video] = {}

        if g_id not in update_info[g.video]:
            update_info[g.video][g_id] = {}

        if save_all_proposals:
            update_info[g.video][g_id]['all_proposals'] = list(all_proposals.values())

        bar.update()

    bar.close()

    # discarding proposals altering the order of the actions, as well as those with confidence <= 0

    bar = tqdm.tqdm(desc='-> Ranking update proposals...', total=len(all_proposals), file=sys.stdout)

    for g_id, proposals in all_proposals.items():
        g = all_plateaus[g_id]
        plateaus_in_video = sorted(plateaus_per_video[g.video], key=lambda x: x.index)  # sort according to index
        n_plateaus = len(plateaus_in_video)
        prop = None

        left_constraint = 0 if g.index == 0 else plateaus_in_video[g.index-1].c
        right_constraint = np.inf if g.index == n_plateaus-1 else plateaus_in_video[g.index+1].c

        update_info[g.video][g_id]['left_constraint'] = left_constraint
        update_info[g.video][g_id]['right_constraint'] = right_constraint

        for prop in proposals[:]:
            if prop.confidence <= 0 or (prop.c <= left_constraint or prop.c >= right_constraint):
                proposals.remove(prop)

        if proposals:
            proposals.sort(key=lambda x: x.confidence, reverse=True)
            prop = proposals[0]  # take the proposal with highest confidence
            prop.id = g_id

            update_candidates.append(
                {
                    'g_id': g_id,
                    'confidence': prop.confidence,
                    'proposal': prop
                }
            )

        if save_all_proposals:
            update_info[g.video][g_id]['valid_proposals'] = list(proposals)

        update_info[g.video][g_id]['chosen_proposal'] = prop
        update_info[g.video][g_id]['updated'] = False

        bar.update()

    bar.close()

    update_candidates.sort(key=lambda x: x['confidence'], reverse=True)
    n_to_update = min(len(update_candidates), round(len(all_plateaus) * settings['z']))

    with tqdm.tqdm(desc='-> Applying updates...', total=n_to_update, file=sys.stdout) as bar:
        for i in range(n_to_update):
            chosen_proposal = update_candidates[i]['proposal']
            g = all_plateaus[chosen_proposal.id]
            g.update_parameters(chosen_proposal, settings['lc'], settings['lw'], settings['ls'])
            update_info[g.video][g.id]['updated'] = True
            bar.update()

    return n_to_update


def write_update_info_to_file(update_info, path):
    with open('{}.pkl'.format(path), 'wb') as f:
        pickle.dump(update_info, f)
