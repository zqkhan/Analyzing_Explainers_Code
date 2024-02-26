import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import squareform, hamming
from random import random
from scipy.stats import pearsonr, entropy


np.random.seed(0)


def rise_generate_masks(N=1000, d=10):

    masks = {key:None for key in range(d)}
    for dim in range(d):
        masks[dim] = np.random.choice([0, 1], size=(N, d), p=[1. / 2, 1. / 2])
        masks[dim][ :, dim] = 1
        masks[dim] = np.unique(masks[dim], axis=0)
    return masks

def rise_explain(model, input, masks, batch_size=1024, NN=True):
    num_points = input.shape[0]
    num_dim = input.shape[1]
    # masks = np.stack(list(masks.values()), axis=0)
    # masks = np.reshape(masks, (masks.shape[0]*masks.shape[1], -1))
    explanations = np.zeros(shape=(num_points, num_dim,))
    if NN:
        model_prediction = model.predict(input, batch_size=batch_size).argmax(axis=-1)
        for d in range(num_dim):
            model_prediction_tiled = np.tile(model_prediction, (len(masks[d])))
            masked_input = np.tile(input, (len(masks[d]), 1)) * np.tile(masks[d], (len(input), 1))
            pred = model.predict(masked_input, batch_size=batch_size)
            prediction = pred[np.arange(len(pred)), model_prediction_tiled]
            prediction = np.reshape(prediction, (len(masks[d]), len(input)))
            explanations[:, d] = prediction.mean(axis=0)
    else:
        for d in range(num_dim):
            prediction = []
            for mask in masks[d]:
                pred = model.predict_proba(input * mask)
                prediction.append(pred)
            prediction = np.array(prediction)
            explanations[:, d] = prediction.mean(axis=0).max(axis=1)
        
    return explanations
# def rise_explain(model, input, masks, batch_size=1024, NN=True):
#
#     num_points = input.shape[0]
#     num_dim = input.shape[1]
#     explanations = np.zeros(shape=(num_points, num_dim,))
#
#     for d in range(num_dim):
#         prediction = []
#         for mask in masks[d]:
#             if NN:
#                 pred = model.predict(input * mask, batch_size=batch_size).max(axis=-1)
#             else:
#                 pred = model.predict_proba(input * mask)
#             prediction.append(pred)
#         prediction = np.array(prediction)
#         explanations[:, d] = prediction.mean(axis=0).max(axis=1)
#
#     return explanations

def calculate_robust_astute(data, explanation, explanation_type='selection', num_points=10, ball_r=2, epsilon=0,
                            kdtree=None):

    if kdtree is not None:
        KDTree_val = kdtree
    else:
        KDTree_val = KDTree(data)
    robust_indicator = []
    if num_points == len(data):
        range_iter = range(len(data))
    else:
        range_iter = np.random.choice(len(data), num_points, replace=False)
    ball_indices = KDTree_val.query_ball_point(data, ball_r)
    for i in range_iter:
        if len(ball_indices[i]) > 0:
            if explanation_type == 'selection':
                d_explanation = (explanation[i] != explanation[ball_indices[i]]).mean(axis=-1)
            elif explanation_type == 'attribution':
                d_explanation = (np.sqrt(((explanation[i] - explanation[ball_indices[i]]) ** 2).sum(axis=-1)))
            else:
                ValueError ("explanation type not supported")
            robust_indicator.append(np.all(d_explanation <= epsilon))
    robust_indicator = np.array(robust_indicator)
    astuteness = robust_indicator.mean()

    return robust_indicator, astuteness

def uniform_sample_ball(center, radius, num_samples=1000):

    d = center.shape[-1]
    points = np.zeros(shape=(num_samples, d))

    U = np.random.normal(0, 1, (num_samples, d))
    norm = np.sum(U ** 2, axis=1) ** 0.5
    r = np.random.random(size=(num_samples)) ** (1.0 / d)
    points = radius * (np.expand_dims(r / norm, -1) * U) + center

    return points

def get_explanation(model, model_type, data, k=2, NN=True, masks=None):

    if model_type == 'L2X':
        scores_val = model.predict(data, verbose=0, batch_size=1000)
        on_indices_val = np.argsort(scores_val, axis=-1)[:, -k:]
        explanation = np.zeros_like(scores_val)
        for i in range(len(explanation)):
            explanation[i, on_indices_val[i]] = 1

    elif model_type == 'invase':
        g_hat = model.importance_score(data)
        explanation = 1. * (g_hat > 0.5)

    elif model_type == 'lime':
        explainer_model = model[0]
        bbox_model = model[1]
        explanation = [np.array(explainer_model.explain_instance(x, bbox_model.predict,
                                                            num_features=data.shape[-1]).as_list())[:, -1].astype(float)
                        for x in data]
        explanation = np.abs(np.array(explanation))

    elif model_type == 'shap':
        if NN:
            shap_values = model.shap_values(data, ranked_outputs=1)
            explanation = np.abs(shap_values[0][0])
        else:
            shap_values = model.shap_values(data, nsamples=10)
            explanation = np.abs(shap_values[0])
    elif model_type == 'rise':
        explanation = rise_explain(model=model, input=data,
                                    masks=masks, batch_size=1024, NN=NN)
    elif model_type == 'cxplain':
        # explanation = model.explain(data.reshape((len(data), 10, 10, 3)), confidence_level=0.80)
        explanation = model.explain(data, confidence_level=0.80)
        explanation = np.abs(explanation)

    return explanation

def calculate_gt_astuteness(data, gt_explanation, num_points=10, ball_r=2, epsilon=0, kdtree=None):

    if kdtree is not None:
        KDTree_val = kdtree
    else:
        KDTree_val = KDTree(data)

    if num_points == len(data):
        range_indices = range(len(data))
    else:
        range_indices = np.random.choice(len(data), num_points, replace=False)

    data_explanation = gt_explanation[range_indices]
    ball_indices = KDTree_val.query_ball_point(data[range_indices], ball_r)
    robust_indicator = []
    for i in range(len(ball_indices)):
        if len(ball_indices[i]) > 0:
            d_explanation = (data_explanation[i] != gt_explanation[ball_indices[i]]).mean(axis=-1)
        robust_indicator.append(np.all(d_explanation <= epsilon))
    robust_indicator = np.array(robust_indicator)
    astuteness = robust_indicator.mean()

    return robust_indicator, astuteness


def calculate_prob_lipschitz(data, model, r=1, L_range=[0.1, 1, 2, 3, 4], num_points=10, NN=True):

    if num_points == len(data):
        range_indices = range(len(data))
    else:
        range_indices = np.random.choice(len(data), num_points, replace=False)
    num_samples = 1000
    lipschitz_indicator = np.zeros(shape=(len(L_range), len(range_indices) * num_samples))
    if NN:
        data_prediction = model.predict(data,)
    else:
        data_prediction = model.predict_proba(data,)
    count = 0
    for i, ind in enumerate(range_indices):
        ball_points = uniform_sample_ball(center=data[ind], radius=r, num_samples=num_samples)
        if NN:
            ball_points_prediction = model.predict(ball_points,)
        else:
            ball_points_prediction = model.predict_proba(ball_points)
        d_prediction = (np.sqrt(((data_prediction[ind] - ball_points_prediction) ** 2).sum(axis=-1)))
        d_data = (np.sqrt(((data[ind] - ball_points) ** 2).sum(axis=-1)))
        for l in range(len(L_range)):
            lipschitz_indicator[l, count : count + num_samples] = np.all(d_prediction <= L_range[l] * d_data)
        count = count + num_samples

    p_lip = lipschitz_indicator.mean(axis=1)
    return p_lip


def calculate_robust_astute_sampled(data, explainer, explainer_type,
                                    explanation_type='selection',
                                    num_points=10, ball_r=2, epsilon=0, k=2,
                                    exponentiate=0, calculate_astuteness=True, NN=True,
                                    data_explanation=None):

    if explainer_type == 'rise':
        masks = rise_generate_masks(10, d=data.shape[1])
    else:
        masks = None
    robust_indicator = []
    if num_points == len(data):
        range_indices = range(len(data))
    else:
        range_indices = np.random.choice(len(data), num_points, replace=False)
    if data_explanation is None:
        data_explanation = get_explanation(model=explainer,
                                           model_type=explainer_type,
                                           data=data, k=k, NN=NN, masks=masks)
    if len(data_explanation.shape) > 2:
        data_explanation = data_explanation.reshape((len(data_explanation), -1))
    if calculate_astuteness:
        for ind in range_indices:
            ball_points = uniform_sample_ball(center=data[ind], radius=ball_r, num_samples=10)
            ball_points_explanation = get_explanation(model=explainer,
                                                      model_type=explainer_type,
                                                      data=ball_points, k=k, NN=NN, masks=masks)
            if len(ball_points_explanation.shape) > 2:
                ball_points_explanation = ball_points_explanation.reshape((len(ball_points_explanation), -1))
            d_data = (np.sqrt(((data[ind] - ball_points) ** 2).sum(axis=-1)))
            if explanation_type == 'selection':
                d_explanation = (data_explanation[ind] != ball_points_explanation).mean(axis=-1)
            elif explanation_type == 'attribution':
                d_explanation = (np.sqrt(((data_explanation[ind] - ball_points_explanation) ** 2).sum(axis=-1)))
            else:
                ValueError ("explanation type not supported")
            if exponentiate:
                d_explanation = np.exp(-d_explanation)
                robust_indicator.append(np.all(d_explanation >= epsilon))
            else:
                if explanation_type == 'attribution':
                    robust_indicator.append(np.all(d_explanation <= epsilon * d_data))
                else:
                    robust_indicator.append(np.all(d_explanation <= epsilon))
        robust_indicator = np.array(robust_indicator)
        astuteness = robust_indicator.mean()

        return robust_indicator, astuteness, data_explanation
    else:
        return data_explanation

def calculate_stability(explanations, explanation_type):

    check_ij = np.eye(len(explanations))
    d_explanations = np.zeros_like(check_ij)
    if explanation_type == 'selection':
        for i in range(len(explanations)):
            for j in range(len(explanations)):
                if not check_ij[i, j]:
                    d_explanations[i, j] = (explanations[i] != explanations[j]).mean(axis=-1).mean()
                    check_ij[i, j] = 1
                    check_ij[j, i] = 1
                    d_explanations[j, i] = d_explanations[i, j]
    elif explanation_type == 'attribution':
        for i in range(len(explanations)):
            for j in range(len(explanations)):
                if not check_ij[i, j]:
                    d_explanations[i, j] = np.sqrt(((explanations[i] - explanations[j]) ** 2).sum(axis=-1)).mean()
                    check_ij[i, j] = 1
                    check_ij[j, i] = 1
                    d_explanations[j, i] = d_explanations[i, j]
    d_explanations = squareform(d_explanations)
    return d_explanations.mean(), d_explanations.std()

def calculate_correlations(explanations, explanation_type):

    check_ij = np.eye(len(explanations))
    d_explanations = np.zeros_like(check_ij)
    if explanation_type == 'selection':
        for i in range(len(explanations)):
            for j in range(len(explanations)):
                if not check_ij[i, j]:
                    # d_explanations[i, j] = (explanations[i] != explanations[j]).mean(axis=-1).mean()
                    d_explanations[i, j] = 1 + np.mean([pearsonr(explanations[i][k], explanations[j][k])[0] for
                                                    k in range(len(explanations[i]))])
                    check_ij[i, j] = 1
                    check_ij[j, i] = 1
                    d_explanations[j, i] = d_explanations[i, j]
    elif explanation_type == 'attribution':
        for i in range(len(explanations)):
            for j in range(len(explanations)):
                if not check_ij[i, j]:
                    # d_explanations[i, j] = np.sqrt(((explanations[i] - explanations[j]) ** 2).sum(axis=-1)).mean()
                    d_explanations[i, j] = 1 + np.mean([pearsonr(explanations[i][k], explanations[j][k])[0] for
                                                    k in range(len(explanations[i]))])
                    check_ij[i, j] = 1
                    check_ij[j, i] = 1
                    d_explanations[j, i] = d_explanations[i, j]
    d_explanations = squareform(d_explanations)
    return d_explanations.mean(), d_explanations.std()


def calculate_locality(explanations, explanation_type, select_k=2):
    if explanation_type == 'selection':
        locality = [entropy(np.unique(explanations[i], return_counts=True, axis=0)[1] / len(explanations[i]))
                    for i in range(len(explanations))]
    elif explanation_type == 'attribution':
        discretized_explanations = []
        for i in range(len(explanations)):
            on_indices_val = np.argsort(np.abs(explanations[i]), axis=-1)[:, -select_k:]
            discretize = np.zeros_like(explanations[i])
            for i in range(len(discretize)):
                discretize[i, on_indices_val[i]] = 1
            discretized_explanations.append(discretize)
        locality = [entropy(np.unique(discretized_explanations[i],
                                      return_counts=True, axis=0)[1] / len(discretized_explanations[i]))
                    for i in range(len(discretized_explanations))]
        # locality = [continuous.get_h(explanations[i], k=100) for i in range(len(explanations))]
    else:
        print('Explainer type not implemented!')
    locality = np.array(locality)
    return locality


def calculate_counts(explanations, tol=2):

    unique_exp, counts = np.unique(explanations, return_counts=True, axis=0)
    ind_max, count_max = np.argmax(counts), np.max(counts)
    new_counts = np.zeros_like(counts)
    while count_max >= int(0.1 * len(explanations)):
        new_counts[ind_max] = count_max
        counts[ind_max] = 0
        for i, c in enumerate(counts):
            if hamming(unique_exp[ind_max], unique_exp[i])*explanations.shape[1] <= tol:
                new_counts[ind_max] += c
                counts[i] = 0
        ind_max, count_max = np.argmax(counts), np.max(counts)

    return new_counts


def calculate_locality_v2(explanations, explanation_type, select_k=2, tol=2):

    if explanation_type == 'selection':
        cleaned_counts = [calculate_counts(explanations[i], tol=tol) for i in range(len(explanations))]
        locality = [entropy(cleaned_counts[i] / np.sum(cleaned_counts[i])) for i in range(len(cleaned_counts))]
    elif explanation_type == 'attribution':
        discretized_explanations = []
        for i in range(len(explanations)):
            on_indices_val = np.argsort(np.abs(explanations[i]), axis=-1)[:, -select_k:]
            discretize = np.zeros_like(explanations[i])
            for i in range(len(discretize)):
                discretize[i, on_indices_val[i]] = 1
            discretized_explanations.append(discretize)
        cleaned_counts = [calculate_counts(discretized_explanations[i], tol=tol) for i in range(len(discretized_explanations))]
        locality = [entropy(cleaned_counts[i] / np.sum(cleaned_counts[i])) for i in range(len(cleaned_counts))]
        # locality = [entropy(np.unique(discretized_explanations[i],
        #                               return_counts=True, axis=0)[1] / len(discretized_explanations[i]))
        #             for i in range(len(discretized_explanations))]
    else:
        print('Explainer type not implemented!')
    locality = np.array(locality)
    return locality

