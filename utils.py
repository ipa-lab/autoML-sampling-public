import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import is_classifier, clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv
from sklearn.model_selection._validation import _fit_and_score
from sklearn.utils import indexable, check_random_state
from yellowbrick.model_selection import LearningCurve
from yellowbrick.style import resolve_colors

DEFAULT_TRAIN_SIZES = np.linspace(0.1, 1.0, 5)


def insert_str(string, str_to_insert, index):
    return string[:index] + str_to_insert + string[index:]


def get_readable_scorer_name(scorer):
    scorer_str = scorer
    if not isinstance(scorer, str):
        scorer_str = scorer._score_func.__name__
        if scorer._sign < 0:
            scorer_str = "Neg " + scorer_str
    return scorer_str.title().replace("_", ' ')

    # extended to also return the indices of the train_sizes_abs
def _translate_train_sizes(train_sizes, n_max_training_samples):
    """Determine absolute sizes of training subsets and validate 'train_sizes'.

    Examples:
        _translate_train_sizes([0.5, 1.0], 10) -> [5, 10]
        _translate_train_sizes([5, 10], 10) -> [5, 10]

    Parameters
    ----------
    train_sizes : array-like of shape (n_ticks,)
        Numbers of training examples that will be used to generate the
        learning curve. If the dtype is float, it is regarded as a
        fraction of 'n_max_training_samples', i.e. it has to be within (0, 1].

    n_max_training_samples : int
        Maximum number of training samples (upper bound of 'train_sizes').

    Returns
    -------
    train_sizes_abs : array of shape (n_unique_ticks,)
        Numbers of training examples that will be used to generate the
        learning curve. Note that the number of ticks might be less
        than n_ticks because duplicate entries will be removed.
    """
    train_sizes_abs = np.asarray(train_sizes)
    n_ticks = train_sizes_abs.shape[0]
    n_min_required_samples = np.min(train_sizes_abs)
    n_max_required_samples = np.max(train_sizes_abs)
    if np.issubdtype(train_sizes_abs.dtype, np.floating):
        if n_min_required_samples <= 0.0 or n_max_required_samples > 1.0:
            raise ValueError("train_sizes has been interpreted as fractions "
                             "of the maximum number of training samples and "
                             "must be within (0, 1], but is within [%f, %f]."
                             % (n_min_required_samples,
                                n_max_required_samples))
        train_sizes_abs = (train_sizes_abs * n_max_training_samples).astype(dtype=np.int, copy=False)
        train_sizes_abs = np.clip(train_sizes_abs, 1,
                                  n_max_training_samples)
    else:
        if (n_min_required_samples <= 0 or
                n_max_required_samples > n_max_training_samples):
            raise ValueError("train_sizes has been interpreted as absolute "
                             "numbers of training samples and must be within "
                             "(0, %d], but is within [%d, %d]."
                             % (n_max_training_samples,
                                n_min_required_samples,
                                n_max_required_samples))
    train_sizes_abs, indices = np.unique(train_sizes_abs, return_index=True)
    if n_ticks > train_sizes_abs.shape[0]:
        warnings.warn("Removed duplicate entries from 'train_sizes'. Number "
                      "of ticks will be less than the size of "
                      "'train_sizes' %d instead of %d)."
                      % (train_sizes_abs.shape[0], n_ticks), RuntimeWarning)

    return train_sizes_abs, indices


def store_indices_and_fit_and_score(estimator, train, test, X, y, scorer, verbose, error_score,
                                    train_full=None, cv_iter=None, **kwargs):
    estimator.set_train_test(train, test, train_full)
    # fit pipelines with subsampled train set
    ret = _fit_and_score(estimator=estimator, train=train, test=test, X=X, y=y, scorer=scorer, verbose=verbose, error_score=error_score,
                         **kwargs)
    print("TEST____fit_score", ret)
    # refitting on full training set
    # only refit on the respective full cv_iteration -> call _fit_and_score on train_full
    refit_out = np.nan
    fit_estimator = ret[4]
    if not fit_estimator.fitted_pipeline_ is None:
        test_score = _fit_and_score(
            estimator=(fit_estimator), X=X, y=y, scorer=scorer, train=train_full, test=test, verbose=verbose,
            parameters=None, fit_params=None, return_train_score=False,
            error_score=error_score, return_times=False,
            return_estimator=False)
        refit_out = test_score[0]
        print("TEST____refit_score", refit_out)
        '''
        for cv_iter_train, cv_iter_test in cv_iter:
            test_score = _fit_and_score(
                estimator=(fit_estimator), X=X, y=y, scorer=scorer, train=cv_iter_train, test=cv_iter_test, verbose=verbose,
                parameters=None, fit_params=None, return_train_score=False,
                error_score=error_score, return_times=False,
                return_estimator=False)
            refit_out.append(test_score)
    else:
        refit_out = [[np.nan]]
    # RuntimeWarning: Mean of empty slice if could not be evaluated (e.g. subsample_ratio is too low)
    cv_mean_test_score = np.nanmean(refit_out, axis=0, dtype=float)[0]  # mean of cross validation after refitting a pipeline (works since the original pipeline is only evaluated on the test set and nothing more)
    ret.append(cv_mean_test_score)
    '''
    ret.append(refit_out)
    return ret

def create_cv_iter_copy(cv_iter, random_state, shuffle):
    rng = check_random_state(random_state)
    return ((rng.permutation(train), test) for train, test in cv_iter) if shuffle else cv_iter

def learning_curve(estimator, X, y, *, groups=None,
                   train_sizes=np.linspace(0.1, 1.0, 5), cv=None,
                   scoring=None, exploit_incremental_learning=False,
                   n_jobs=None, pre_dispatch="all", verbose=0, shuffle=False,
                   random_state=None, error_score=np.nan, return_times=False, return_estimators=False):


    if exploit_incremental_learning and not hasattr(estimator, "partial_fit"):
        raise ValueError("An estimator must support the partial_fit interface "
                         "to exploit incremental learning")
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    # Store it as list as we will be iterating over the list multiple times
    cv_iter = list(cv.split(X, y, groups))

    scorer = check_scoring(estimator, scoring=scoring)
    # add check so that only one scorer gets provided
    # scorers, _ = _check_multimetric_scoring(estimator, scoring=scoring)
    # scorer = scorers

    n_max_training_samples = len(cv_iter[0][0])
    # Because the lengths of folds can be significantly different, it is
    # not guaranteed that we use all of the available training data when we
    # use the first 'n_max_training_samples' samples.
    train_sizes_abs, indices = _translate_train_sizes(train_sizes,
                                                      n_max_training_samples)
    n_unique_ticks = train_sizes_abs.shape[0]
    if verbose > 0:
        print("[learning_curve] Training set sizes: " + str(train_sizes_abs))

    parallel = Parallel(n_jobs=n_jobs, pre_dispatch=pre_dispatch,
                        verbose=verbose)  # , backend='threading')

    train_test_proportions = []
    for train, test in create_cv_iter_copy(cv_iter, random_state, shuffle):
        for n_train_samples in train_sizes_abs:
            train_test_proportions.append((train[:n_train_samples], test, train))  # takes actual random ones if shuffling

    # impute values once and not for each, to save time
    cloned_estim = clone(estimator)
    cloned_estim._fit_init()
    X, y = cloned_estim._check_dataset(X, y)

    out = parallel(delayed(store_indices_and_fit_and_score)(
        estimator=clone(estimator), X=X, y=y, scorer=scorer, train=train, test=test, verbose=verbose,
        train_full=train_full, cv_iter=create_cv_iter_copy(cv_iter, random_state, shuffle),
        parameters=None, fit_params=None, return_train_score=True,
        error_score=error_score, return_times=return_times,
        return_estimator=return_estimators)
                   for train, test, train_full in train_test_proportions)
    out = np.array(out)
    n_cv_folds = out.shape[0] // n_unique_ticks
    dim = 6 if return_times else 4
    out = out.reshape(n_cv_folds, n_unique_ticks, dim)

    out = np.asarray(out).transpose((2, 1, 0))

    ret = train_sizes_abs, indices, out[0], out[1], np.array(out[5]) #np.array(refitted_test_scores) #
    print("TEST____output", ret)

    if return_times:
        ret = ret + (out[2], out[3])
    ret = ret + (out[4], train_sizes)
    return ret


class LearningCurveWithEstimators(LearningCurve):
    def __init__(
            self,
            model,
            ax=None,
            groups=None,
            train_sizes=DEFAULT_TRAIN_SIZES,
            cv=None,
            scoring=None,
            exploit_incremental_learning=False,
            n_jobs=None,
            pre_dispatch="all",
            shuffle=False,
            random_state=None,
            return_times=True,
            return_estimators=False,
            **kwargs
    ):
        super().__init__(model, ax, groups, train_sizes, cv, scoring, exploit_incremental_learning, n_jobs,
                         pre_dispatch, shuffle, random_state, **kwargs)

        # Extend the metric parameters to be used later with return_times and return_estimators
        self.set_params(
            groups=groups,
            train_sizes=train_sizes,
            cv=cv,
            scoring=scoring,
            exploit_incremental_learning=exploit_incremental_learning,
            n_jobs=n_jobs,
            pre_dispatch=pre_dispatch,
            shuffle=shuffle,
            random_state=random_state,
            return_times=return_times,
            return_estimators=return_estimators
        )
        # extended by return_estimators

    def init_drawing(self):
        # compute the mean and standard deviation of the training data
        self.train_scores_mean_ = np.nan_to_num(np.nanmean(self.train_scores_, axis=1, dtype=float))
        self.train_scores_std_ = np.nan_to_num(np.nanstd(self.train_scores_, axis=1, dtype=float))

        # compute the mean and standard deviation of the test data
        self.test_scores_mean_ = np.nan_to_num(np.nanmean(self.test_scores_, axis=1, dtype=float))
        self.test_scores_std_ = np.nan_to_num(np.nanstd(self.test_scores_, axis=1, dtype=float))

        # compute the mean and standard deviation of the test data after refitting on the full train data
        self.test_scores_refitted_mean_ = np.nan_to_num(np.nanmean(self.test_scores_refitted_, axis=1, dtype=float))
        self.test_scores_refitted_std_ = np.nan_to_num(np.nanstd(self.test_scores_refitted_, axis=1, dtype=float))

        # draw the curves on the current axes
        self.draw()
        return self

    # extended by return_times & return_estimators
    # extended by also saving estimators
    # extended by calculating nammean() and nanstd()
    def fit(self, X, y=None):
        # arguments to pass to sk_learning_curve
        sklc_kwargs = {
            key: self.get_params()[key]
            for key in (
                "groups",
                "train_sizes",
                "cv",
                "scoring",
                "exploit_incremental_learning",
                "n_jobs",
                "pre_dispatch",
                "shuffle",
                "random_state",
                "return_times",
                "return_estimators"
            )
        }

        # compute the learning curve and store the scores on the estimator
        curve = learning_curve(self.estimator, X, y, **sklc_kwargs)
        self.train_sizes_, self.train_sizes_indices_, self.train_scores_, self.test_scores_, self.test_scores_refitted_, self.fit_time_, self.score_time_, self.estimators_, self.sampling_ratio_ = curve  # .values()

        return self.init_drawing()

    def finalize(self, **kwargs):
        super(LearningCurveWithEstimators, self).finalize()
        self.ax.set_ylabel(get_readable_scorer_name(self.get_params()['scoring']))

        '''
        self.ax2 = self.ax.twiny()
        #self.ax2.xaxis.set_major_formatter(mtick.PercentFormatter(100, 0, None))
        train_sizes_draw_indices = []
        for i in reversed(self.train_sizes_indices_):
            if not (i > 0 and (self.train_sizes_indices_[i] - self.train_sizes_indices_[i-1] < 0.05)):
                train_sizes_draw_indices.append(i)  # TODO: fix error


        self.ax2.set_xlim(self.ax.get_xlim())
        self.ax2.set_xticks(self.train_sizes_[train_sizes_draw_indices])
        self.ax2.set_xticklabels(map('{:.1%}'.format, self.sampling_ratio_[train_sizes_draw_indices]))
        self.ax2.set_xlabel("Training Subsample Ratios [%]")
        '''

        return self.ax

    def draw(self, **kwargs):
        """
        Renders the training and test learning curves.
        """
        # Specify the curves to draw and their labels
        labels = ("Training Score", "Cross Validation Score", "Refitted CV Score")
        curves = (
            (self.train_scores_mean_, self.train_scores_std_),
            (self.test_scores_mean_, self.test_scores_std_),
            (self.test_scores_refitted_mean_, self.test_scores_refitted_std_),
        )

        # Get the colors for the train, test and refitted test curves
        colors = resolve_colors(n_colors=3)

        # Plot the fill betweens first so they are behind the curves.
        for idx, (mean, std) in enumerate(curves):
            # Plot one standard deviation above and below the mean
            self.ax.fill_between(
                self.train_sizes_, mean - std, mean + std, alpha=0.25, color=colors[idx]
            )

        # Plot the mean curves so they are in front of the variance fill
        for idx, (mean, _) in enumerate(curves):
            self.ax.plot(
                self.train_sizes_, mean, "o-", color=colors[idx], label=labels[idx]
            )

        return self.ax
