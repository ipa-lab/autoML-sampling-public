#!/usr/bin/env python3
# this file was copied from: https://github.com/josepablocam/ams/tree/master/experiments and adapted for openml data fetching
import warnings

# ignore sklearn future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# we do not use NN
warnings.filterwarnings(action='ignore',
                        message="Warning: optional dependency*")
import numbers
from argparse import ArgumentParser
import copy
from datetime import datetime
import os
import dill as pickle
import random
import traceback

import json
import pandas as pd
import pmlb
import tpot
import sklearn.base
import sklearn.pipeline
import sklearn.metrics
from sklearn.model_selection import StratifiedKFold
import sys

if sys.argv[1] == "--IDEdebugging":
    from experiments import download_datasets as dd
    from experiments import mp_utils
else:
    import download_datasets as dd
    import mp_utils

sys.path.append("../")
sys.path.append(".")
from extract_pkl_csv import to_csv_and_print, get_valid_file_path, find_best_result_sampling_ratio
from utils import *

MAX_TIME_MINS_PER_PIPELINE = 1


# Occasionally TPOT fails
# so rather than waste everything, we just mark
# that iteration of CV as a failure and move on
class FailedOptim(object):
    def __init__(
            self,
            error,
            error_msg,
            X=None,
            y=None,
            search=None,
            default_prob=0.1,
            default_label=0,
    ):
        self.error = error
        self.error_msg = error_msg
        self.fitted_pipeline_ = None
        self.evaluated_individuals_ = None
        self.pareto_front_fitted_pipelines_ = None
        # save a copy of the data that raised error
        # for debugging
        self.X = X
        self.y = y
        self.search = search

        self.default_prob = default_prob
        if y is not None:
            if isinstance(y, pd.DataFrame):
                y = y[y.columns[0]]
            # set first value as default label, in case types are different
            # than integer
            default_label = y[0]
        self.default_label = default_label

    def predict_proba(self, X):
        return np.repeat(self.default_prob, X.shape[0])

    def decision_function(self, X):
        return np.repeat(self.default_prob, X.shape[0])

    def predict(self, X):
        return np.repeat(self.default_label, X.shape[0])

    def _check_dataset(self, X, y):
        return X, y


class RobustSearch(sklearn.base.BaseEstimator):
    def __init__(self, search_model, noise=None):
        mp_utils.init_mp()
        self.search_model = search_model
        self.fitted = False
        self.train = None
        self.test = None
        self.train_full = None

    def fit(self, X, y):
        try:
            if not self.fitted:
                self.fitted = True  # need to set it before, else a second refit call on a FailedOptim calls a regular fit()
                self.search_model.fit(X, y)
                self.search_model.log_file = None
            elif self.fitted_pipeline_ is not None:
                X, y = self._check_dataset(X, y)  # impute missing values like TPOTClassifier
                self.fitted_pipeline_.fit(X, y)
        except (Exception, RuntimeError, TimeoutError, KeyboardInterrupt) as err:
            error_msg = ("Refitting: " if self.fitted else "") + traceback.format_exc()
            print("RobustSearch failed during {}fitting".format("re" if self.fitted else ""))
            print(error_msg)
            self.search_model.log_file = None
            self.failed_model = self.search_model
            self.search_model = FailedOptim(
                err,
                error_msg,
                X=X,
                y=y,
                search=self.search_model,
            )

    def set_train_test(self, train, test, train_full):
        self.train = train
        self.test = test
        self.train_full = train_full

    def __getattr__(self, attr_name):
        return getattr(self.search_model, attr_name)


def get_robust_tpot(
        config_dict=None,
        max_time_mins=5,
        scoring="f1_macro",
        cv=5,
        random_state=42,
        n_jobs=-1,
        check_point_folder=None,
        verbosity=3,
        sampling_ratio=1.0,
        testing=False,
):
    clf = RobustSearch(
        search_model=tpot.TPOTClassifier(
            config_dict=config_dict,
            scoring=scoring,
            cv=cv if not testing else 2,  # testing
            n_jobs=n_jobs,
            max_time_mins=max_time_mins if not testing else 1,
            # max on a single timeline...otherwise can blow out
            # and end up with not a single pipeline fit
            max_eval_time_mins=MAX_TIME_MINS_PER_PIPELINE if not testing else 1,
            random_state=random_state,
            verbosity=verbosity,
            disable_update_check=True,
            subsample=sampling_ratio,
            generations=100 if not testing else 1,  # testing
            population_size=100 if not testing else 2  # testing
        )
    )
    return clf




def get_no_hyperparams_config(config_dict):
    # drop hyperparameters from configuration
    return {k: {} for k in config_dict.keys()}


def get_scoring(scoring, n_target_classes, benchmark_scoring):
    if scoring == "balanced_accuracy_score":
        return sklearn.metrics.make_scorer(
            sklearn.metrics.balanced_accuracy_score
        )
    if benchmark_scoring:
        if n_target_classes == 2:
            scoring = "roc_auc"
        elif n_target_classes > 2:
            scoring = "neg_log_loss"  # sklearn.metrics.make_scorer(log_loss, greater_is_better=False,
            # needs_proba=True)#"neg_log_loss"
    print("Number of classes: {0} -> Using scoring function: {1}".format(int(n_target_classes), scoring))
    return scoring


def get_num_pipelines_explored(model):
    if isinstance(model, sklearn.pipeline.Pipeline):
        return 1
    elif isinstance(model, tpot.TPOTClassifier):
        return len(model.evaluated_individuals_)
    elif isinstance(model, FailedOptim):
        return 0
    else:
        raise Exception("Unknown search model")


def limit_poly_features_in_config(config, X, max_cols=50, max_degree=2):
    # Trying to generate degrees of order 4
    # with anything more than a couple of columns
    # quickly blows up
    # copy in case we modify it
    config = copy.deepcopy(config)
    if X.shape[1] < max_cols:
        return config

    params = None

    poly_comp = "sklearn.preprocessing.PolynomialFeatures"
    if isinstance(config, dict):
        params = config.get(poly_comp, None)
    elif isinstance(config, list) and isinstance(config[0], str):
        # its a list configuration without hyperparamers
        return config
    else:
        # its a list configuration, for specified order
        entry = [
            comp_dict for comp_dict in config if poly_comp in comp_dict.keys()
        ]
        params = None if len(entry) == 0 else entry[0][poly_comp]

    if params is None or 'degree' not in params:
        # not relevant, or using default (degree=2), so good to go
        return config
    else:
        # set a max on the degree
        params['degree'] = [d for d in params['degree'] if d <= max_degree]
        return config


def fetch_data(dataset, target, cache_dir, use_pmlb):
    n_target_classes = None
    try:
        if use_pmlb:
            X, y = pmlb.fetch_data(
                dataset,
                return_X_y=True,
                local_cache_dir=cache_dir,
            )
        else:
            X, y, categorical_indicator, features, n_target_classes = dd.get_openml_data(dataset, target)

    except ValueError:
        path = os.path.join(cache_dir, dataset)
        df = pd.read_csv(path)
        y_col = "target"
        X_cols = [c for c in df.columns if c != y_col]
        X = df[X_cols].values
        y = df[y_col].values
    return X, y, n_target_classes


def run_dataset_learning_curve(
        dataset,
        search,
        config=None,
        max_time_mins=5,
        max_depth=4,
        verbosity=3,
        cv=10,
        scoring="f1_macro",
        n_jobs=-1,
        random_state=None,
        target=None,
        sampling_method="random",
        sampling_ratios=None,  # (0, 1]
        use_pmlb=False,
        testing=False,
        benchmark_scoring=True,
        output="output"
):
    if sampling_ratios is None:
        sampling_ratios = [0.1, 0.5, 1]

    X, y, n_target_classes = fetch_data(dataset, target, cache_dir=dd.DEFAULT_LOCAL_CACHE_DIR, use_pmlb=use_pmlb)

    cv_splitter = StratifiedKFold(
        cv,
        random_state=random_state,
        shuffle=True,
    )

    scoring_fun = get_scoring(scoring, n_target_classes, benchmark_scoring)

    config = limit_poly_features_in_config(config, X)

    if search == "tpot":
        # running search with tpot
        model = get_robust_tpot(
            config_dict=config,
            max_time_mins=max_time_mins,
            scoring=scoring_fun,
            verbosity=verbosity,
            n_jobs=n_jobs,
            random_state=random_state,
            sampling_ratio=1.0,  # learning_curve takes over the subsampling
            testing=testing,
        )
    # elif search == "random":
    #     model = get_robust_random(
    #         config_dict=config,
    #         max_depth=max_depth,
    #         max_time_mins=max_time_mins,
    #         scoring=scoring_fun,
    #         random_state=random_state,
    #         n_jobs=1,
    #     )
    # elif search == "predefined-with-hyperparams":
    #    model = get_robust_predefined_random(
    #        config,
    #        max_time_mins=max_time_mins,
    #        scoring=scoring_fun,
    #    )
    else:
        raise TypeError(
            "configuration must be dictionary (automl) or list (simple)"
        )
    start_time = datetime.now()
    results = LearningCurveWithEstimators(
        # https://medium.com/@nesrine.ammar/how-learning-curve-function-from-scikit-learn-works-692d7d566d17
        model=model,
        # we set shuffle to True, since using a low ratio could lead to the same training samples being picked for all the CV splits
        shuffle=True,
        train_sizes=np.array(sampling_ratios),  # * (1 - 1 / cv),
        cv=cv_splitter,
        exploit_incremental_learning=False,
        scoring=scoring_fun,
        # n_jobs=n_jobs,
        random_state=random_state,
        return_times=True,
        return_estimators=True,
        title='{0} Search Learning Curve for {1}'.format(search.upper(), dataset)
    )

    results.fit(X, y)
    end_time = datetime.now()
    exec_time = (end_time - start_time).total_seconds()
    results.show(outpath=get_valid_file_path(output) + ".pdf")

    nrows = len(results.test_scores_[0])

    # TOPT and ours can fail during fitting...
    fitted_pipelines = [e.fitted_pipeline_ for e in results.estimators_.ravel()]
    evaluated_individuals = [e.evaluated_individuals_ for e in results.estimators_.ravel()]
    pareto_front_fitted_pipelines = [e.pareto_front_fitted_pipelines_ for e in results.estimators_.ravel()]             # important -> needs verbosity = 3

    # replace scores with np.nan if produced by a failed optimization
    scores = [
        score
        if not isinstance(estimator.search_model, FailedOptim) else np.nan
        for score, estimator in
        zip(results.test_scores_.ravel(), results.estimators_.ravel())
    ]
    # mean cv training scores after refitting the pipeline on the full training set
    scores_refitted = [
        score
        if not isinstance(estimator.search_model, FailedOptim) else np.nan
        for score, estimator in
        zip(results.test_scores_refitted_.ravel(), results.estimators_.ravel())
    ]
    # training scores
    train_scores = [
        score
        if not isinstance(estimator.search_model, FailedOptim) else np.nan
        for score, estimator in
        zip(results.train_scores_.ravel(), results.estimators_.ravel())
    ]
    # The time for fitting the estimator on the train set for each cv split.
    train_fit_time = [
        fit_time
        if not isinstance(estimator.search_model, FailedOptim) else np.nan
        for fit_time, estimator in
        zip(results.fit_time_.ravel(), results.estimators_.ravel())
    ]
    # The time for scoring the estimator on the test set for each cv split.
    test_score_time = [
        score_time
        if not isinstance(estimator.search_model, FailedOptim) else np.nan
        for score_time, estimator in
        zip(results.score_time_.ravel(), results.estimators_.ravel())
    ]
    # keep track of errors, so we can debug searches
    errors = [
        {k: v for k, v in dict(vars(estimator.search_model),
                               train=estimator.train, test=estimator.test
                               ).items() if k not in ['search', 'X',
                                                      'y']}  # vars(estimator.search_model)#        {'err': estimator.search_model.error, 'search': estimator.search_model.search}    # using estimator.search model also saves data X, y (only works when pickling and leads to big files)
        if isinstance(estimator.search_model, FailedOptim) else None
        for estimator in results.estimators_.ravel()
    ]
    # pipelines explored
    pipelines_explored = [
        get_num_pipelines_explored(estimator.search_model)
        for estimator in results.estimators_.ravel()
    ]
    estimators = [
        estimator.search_model
        # if  isinstance(estimator.search_model, FailedOptim) else None
        for estimator in
        results.estimators_.ravel()
    ]

    cv_splits = cv_splitter if isinstance(cv_splitter, numbers.Integral) else cv_splitter.n_splits
    results_info = {
        "score": scores,
        "score_refitted": scores_refitted,
        "train_score": train_scores,
        "cv_iter": list(range(cv_splits)) * len(results.train_sizes_),
        "dataset": dataset,
        "sampling_method": sampling_method,
        "sampling_ratio": [i for i in results.sampling_ratio_[results.train_sizes_indices_] for _ in
                           range(cv_splits)],
        "train_sizes": [i for i in results.train_sizes_ for _ in
                        range(cv_splits)],
        "scoring": scoring_fun,
        "search": [search] * nrows * len(results.train_sizes_),
        "config_dict": [config] * nrows * len(results.train_sizes_),
        "max_time_mins": max_time_mins,
        "estimator": estimators,
        "fitted_pipeline": fitted_pipelines,
        "pareto_front_fitted_pipelines": pareto_front_fitted_pipelines,
        "evaluated_individuals": evaluated_individuals,
        "pipelines_explored": pipelines_explored,
        "train_fit_time": train_fit_time,
        "test_score_time": test_score_time,
        "total_exec_time_secs": exec_time,  # / nrows,
        "errors": errors,
    }

    # results_df = pd.DataFrame(results_info)
    # to_csv_and_print(results_df, output)
    return results_info


def to_df_and_save(acc, name, output):
    acc_df = pd.concat(acc, axis=0) if isinstance(acc, list) else pd.DataFrame(acc)
    if name is not None:
        acc_df["name"] = name
    else:
        acc_df["name"] = "unk"

    if output is not None:
        output = output[:3] + output[3:].replace(':', '_')
        try:
            acc_df.drop("estimator", 1, errors="ignore", inplace=True) # python3.8 error: pickle cannot pickle '_io.TextIOWrapper' object
            to_csv_and_print(acc_df, output)

            split = output.split("/")
            pkl_output = "/".join(split[:-1]) + "/pkl/" + split[-1]
            dir_path = os.path.dirname(pkl_output)
            if len(dir_path) > 0:
                os.makedirs(dir_path, exist_ok=True)
            acc_df.to_pickle(pkl_output + ".pkl")
        except Exception as err:
            print("Error: Could not pickle df (possibly due to multiprocessing or python3.8 reasons).", err)

    return acc_df


def load_config(poss_config):
    if isinstance(poss_config, str) and poss_config == "TPOT":
        return copy.deepcopy(tpot.config.classifier_config_dict)
    try:
        config = json.loads(poss_config)
        return config
    except json.JSONDecodeError:
        with open(poss_config, "r") as fin:
            return json.load(fin)


def get_args():
    parser = ArgumentParser(description="Run experiments")
    parser.add_argument(
        "--dataset",
        type=str,
        # nargs="+",
        help="Name of dataset to run",
    )
    # needed for RandomSearch (optional for TPOT)
    parser.add_argument(
        "--config",
        type=str,
        help="String dictionary for config_dict or path to file",
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search strategy",
        choices=[
            "tpot",
            "random",
            # "simple",
            # "predefined-with-hyperparams",
        ]
    )
    parser.add_argument(
        "--pmlb",
        type=int,
        help="Using PMLB or openML datasets",
        default=0
    )
    parser.add_argument(
        "--target",
        type=str,
        help="Target attribute")
    parser.add_argument(
        "--sampling_method",
        type=str,
        help="Sampling Method",
        choices=[
            "stratify",
            "random",  # <subsample> parameter of TPOT randomly picks e.g. 0.5 of the training instances
            "cluster-kmeans",
            "oversampling",
            # IMPORTANT: split the data BEFORE oversampling as else it leads to train and test data being related (https://www.reddit.com/r/MachineLearning/comments/erx7d2/r_oversampling_done_wrong_leads_to_overly/)
        ]
    )
    parser.add_argument(
        "--sampling_ratio",
        type=float,
        nargs="+",
        help="Ratio of instances we train on",
    )
    parser.add_argument(
        "--cv",
        type=int,
        help="Number of CV iters",
        default=10,
    )
    parser.add_argument(
        "--scoring",
        type=str,
        help="Scoring function",
        default="f1_macro",
    )
    parser.add_argument(
        "--max_time_mins",
        type=int,
        help="Time budget for each outer cv iteration",
        default=5,
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        help="Number of cores to use",
        default=-1,
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        help="Max search depth for random search",
        default=4,
    )
    parser.add_argument(
        "--components_only",
        action="store_true",
        help="Drop hyperparameters from configuration",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        help="Seed for RNG",
        default=42,
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name for experiment",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for results",
    )
    parser.add_argument(
        "--benchmark_scoring",
        type=int,
        help="Use the same scoring as the AutoML benchmark (binary: AUROC, multiclass: log loss)",
        default=1,
    )
    parser.add_argument(
        "--test",
        type=int,
        help="Runs experiment on only a few instances on small datasets for local testing",
        default=0,
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        help="Determines how much output gets printed during experiment running",
        default=3,
    )
    parser.add_argument(
        "--rerun_best",
        type=int,
        help="Runs only the best sampling ratios if results exist in /result_successful if set to 1. If set to 2 reruns best and full (1.0) sampling ratio.",
        default=0,
    )
    parser.add_argument(
        "--rerun_score_col",
        type=str,
        help="If rerun_best > 0: Which column (score_refitted, cv_iter_score, ...) to use in order to pick the highest scoring sampling ratio.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="Input path for rerunning best performing sampling ratios",
    )


    return parser.parse_args()


def main():
    args = get_args()
    config = None
    if args.config is not None:
        config = load_config(
            args.config)  # if args.config = "TPOT" we can use Random Search on the default TPOT configurations

    if config is not None and args.components_only:
        print("Dropping hyper-parameters from configuration")
        config = get_no_hyperparams_config(config)

    acc = []
    if args.name is not None:
        print("Running run_experiment.py, name={}".format(args.name))

    if args.random_state:
        # adding more set seeds....something deep down
        # in tpot/sklearn not actually taking the random seed otherwise
        np.random.seed(args.random_state)
        random.seed(args.random_state)

    if args.output is not None:
        dir_path = os.path.dirname(args.output)
        if len(dir_path) > 0:
            os.makedirs(dir_path, exist_ok=True)

    args.output += get_valid_file_path()

    if args.rerun_best > 0:
        args.sampling_ratio = find_best_result_sampling_ratio(args.dataset, args.sampling_ratio, args.rerun_best,
                                                              args.input_path,
                                                              args.rerun_score_col)

    acc = run_dataset_learning_curve(
        args.dataset,
        search=args.search,
        config=config,
        max_depth=args.max_depth,
        max_time_mins=args.max_time_mins,
        cv=args.cv,
        scoring=args.scoring,
        n_jobs=args.n_jobs,
        random_state=args.random_state,
        verbosity=args.verbosity,
        target=args.target,
        sampling_method=args.sampling_method,
        sampling_ratios=args.sampling_ratio,
        use_pmlb=args.pmlb,
        testing=args.test,
        benchmark_scoring=args.benchmark_scoring,
        output=args.output
    )
    to_df_and_save(acc, args.name, args.output)


if __name__ == "__main__":
    try:
        start_time = datetime.now()
        np.set_printoptions(threshold=np.inf)  # stores the whole np array in csv
        pd.set_option("display.max_colwidth",
                      10000)  # sets the max width of pandas to allow df.to_csv() storing with full columns
        main()
        end_time = datetime.now()
        exec_time = (end_time - start_time).total_seconds()
        print("OVERALL_EXEC_TIME", exec_time)
    except Exception as err:
        print("Error:", err)
        args = get_args()
        err_path = args.output + "-error.pkl"
        with open(err_path, "wb") as fout:
            pickle.dump(err, fout)

        detailed_msg = traceback.format_exc()
        tb_path = args.output + "-tb.txt"
        with open(tb_path, "w") as fout:
            fout.write(detailed_msg)
            fout.write("\n")

        failed_args_path = args.output + "-args.pkl"
        with open(failed_args_path, "wb") as fout:
            pickle.dump(args, fout)

        if args.test:
            import pdb

            pdb.post_mortem()
        print(detailed_msg)
        sys.exit(1)
