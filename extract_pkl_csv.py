import os
import dill as pickle
import random
import sys
import traceback

from numpy.core.defchararray import isnumeric
from sklearn.impute import SimpleImputer
from sklearn.metrics import check_scoring, get_scorer
from sklearn.model_selection import StratifiedKFold, check_cv
from sklearn.model_selection._validation import _fit_and_score
from sklearn.utils import indexable
from tabulate import tabulate
import numpy as np
import pandas as pd
from datetime import datetime, time
from os import listdir
import openml
from matplotlib import pyplot as plt

from utils import create_cv_iter_copy

from experiments import download_datasets as dd

from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve

def main():
    print(len(sys.argv))
    if (len(sys.argv) < 2):
        print("No file to convert provided.")
        return
    filepath = sys.argv[1]
    if filepath.endswith(".pkl"):
        pkl_obj = pickle.loads(open(filepath, 'rb').read())
        if isinstance(pkl_obj, pd.DataFrame):
            pkl_obj.to_csv('{}_manual.csv'.format(sys.argv[1][:-4]))
    elif filepath.endswith(".csv"):
        pkl_obj = pd.read_csv(filepath)

    if isinstance(pkl_obj, pd.DataFrame):
        pkl_obj = pkl_obj.drop('config_dict', 1)
        print(tabulate(pkl_obj, headers='keys', tablefmt='psql'))
    else:
        print(pkl_obj)


def get_valid_file_path(output=None):
    if output is None:
        output = str(datetime.now().isoformat())[:-7].replace(':', '_')
    return output.replace('.pkl', '')


def to_csv_and_print(obj, output=None):
    output = get_valid_file_path(output) + '.csv'
    obj.to_csv(output)  # , mode='a', header=not os.path.exists(output))

    obj_wo_config = obj.drop('config_dict', 1, errors='ignore')  # pretty print in console without dict
    print(tabulate(obj_wo_config, headers='keys', tablefmt='psql'))


def inplace_change(filename, old_string, old_string2, new_string):
    # Safely read the input filename using 'with'
    with open(filename) as f:
        s = f.read()
        if old_string not in s and old_string2 not in s:
            print('{}, {} not found in {}.'.format(old_string, old_string2, filename))
            return

    # Safely write the changed content, if found in the file
    with open(filename, 'w') as f:
        print('Changing "{old_string}" to "{new_string}" in {filename}'.format(**locals()))
        s = s.replace(old_string, new_string)
        s = s.replace(old_string2, new_string)
        f.write(s)


def results_to_latex(column_to_average, excluded_data=None, excluded_sampling_ratio=None, int_only=False,
                     col_to_subtract=None,
                     as_percentage=False,
                     results_dir="results/",
                     bold=True, datasets_table=False, plot=False, file_name="",
                     testing=False,
                     long_min=60):
    if excluded_sampling_ratio is None:
        excluded_sampling_ratio = []
    if excluded_data is None:
        excluded_data = []

    filepaths = [results_dir + f for f in listdir("./" + results_dir) if f.endswith('.csv') and f[0].isdigit()]
    print(filepaths, len(filepaths))
    df = pd.concat(map(pd.read_csv, filepaths))

    # delete rows for excluded datasets
    df = df[~df.dataset.isin(excluded_data)]

    if int_only:
        df = df.dropna(subset=[
            "score"])  # drop null values when averaging pipelines explored since the amount was saved as 0 instead of NaN

    caption = df['sampling_method'].iloc[0]
    print(caption)
    df.drop(columns=['sampling_method'])
    # use agg(['mean', 'count]) to also show how many folds were evaluated
    df = df.groupby(['dataset', 'sampling_ratio'])[column_to_average].agg(['mean']).reset_index()

    df = df.set_index(['dataset', 'sampling_ratio']).unstack('sampling_ratio')

    output_path = results_dir + "result_tables/" + str(datetime.now().isoformat())[:-7].replace(':',
                                                                                                '_')
    df.rename(columns={'sampling_ratio': "ratio"}, inplace=True)
    df.replace(np.nan, '', regex=True, inplace=True)
    Δ = ""  # "$\Delta$"
    df.columns = [Δ + str(col[1]) for col in df.columns]

    # delete sampling ratios for excluded datasets
    df.drop(excluded_sampling_ratio, axis=1, inplace=True)
    # df = df.sub(df['Δ1.0'], axis=0)
    # first 4 columns are strings?

    # df[["Δ0.0001", "Δ0.001", "Δ0.01", "Δ0.05", "Δ0.1", "Δ0.15", "Δ0.2", "Δ0.3", "Δ0.5"]] = df['Δ1.0'].values[:, None] - df[["Δ0.0001", "Δ0.001", "Δ0.01", "Δ0.05", "Δ0.1", "Δ0.15", "Δ0.2", "Δ0.3", "Δ0.5"]]
    # df.style.apply
    # max_idx = df.idxmax(axis=1, skipna = True)
    # df = df.round(3)
    # ATTENTION: does not print, just saves the results to precision 3

    data_list = openml.datasets.get_datasets(df.index, download_data=False)
    print(data_list[0])
    # id, name, instances, features,  classes (target)

    data = []
    for d in data_list:
        d1 = {"ID": d.dataset_id,
              "Name": d.name}
        d2 = {
            k.replace("NumberOf", "\#"): int(v) for k, v in d.qualities.items() if
            k in ['NumberOfInstances', 'NumberOfClasses', 'NumberOfFeatures']
        }
        data.append({**d1, **d2})

    df_datasets = pd.DataFrame(data)
    df_datasets.set_index("ID", inplace=True)
    df_datasets.sort_values(axis=0, by=str("\#Instances"), inplace=True)
    print(df_datasets)

    if datasets_table:
        df_datasets.to_latex(mk_dir_return(output_path) + "_data.txt", caption=caption, escape=False)

    df = df.reindex(index=df_datasets.index)
    df.reset_index(inplace=True)

    df.loc[:, df.columns != 'ID'] = df.loc[:, df.columns != 'ID'].apply(
        pd.to_numeric)  # makes them numeric but fills empty values with nan

    df = df.set_index("ID")

    float_nums = "{:0.3}".format
    out_f = "_results_succesful" + file_name
    if int_only:
        float_nums = None
        out_f = out_f + "_" + column_to_average
        # hack to bold highest of row
        if column_to_average == "train_fit_time":
            df[df.columns] = df[df.columns] / 60  # show as minutes

        df[df.columns] = df[df.columns].fillna(0.0).astype(int)

    out_f = out_f + ".txt"

    cols_to_round = [col for col in df.columns if col not in ["ID", col_to_subtract]]
    if as_percentage:
        df = df.apply(lambda x: x * 100.0 if x.name in cols_to_round else x)

    df_str = df[df.columns]  # .astype(str)
    # 1. subtract columns
    if col_to_subtract:  # subtract column from all the others
        for i, r in df.iterrows():
            # print(i, r)
            # print(r.max(), r.idxmax())
            for c, value in r.iteritems():
                if c != r.idxmax and c != col_to_subtract:
                    df_str.at[i, c] = round(
                        float(df_str.at[i, c]) - float(df_str.at[i, col_to_subtract]) * 100.0 if as_percentage else 1.0,
                        3)  # r.max()         # subtract 1.0 value of row for other columns
    # 2. round and to string and bold highest
    if not int_only:
        df_str = df_str.round(decimals=3)
    if as_percentage:
        df_str = df_str.round(decimals=1)
        if col_to_subtract:
            df_str[col_to_subtract] = df[col_to_subtract].round(3)
    df_str = df_str.astype(str)

    for i, r in df.iterrows():
        if "{}min".format(long_min) not in results_dir:
            if bold:
                df_str.at[i, r.idxmax] = "\\textbf{" + str(df_str.at[i, r.idxmax()]) + "}"
            # df_str.at[i, r.idxmin] = "\\emph{" + str(r.min()) + "}"

            # print(i, round(df.at[i, "1.0"], 3))

        # else:
        #    df2 = pd.DataFrame(columns=['ID', 'sampling_ratio', 'qty2'])
        #    df2.iloc[i] = [1, r.idxmax, ]
        #    print(i, r.idxmax(), round(r.max(), 3))

        print(  # i, r.idxmax(),
            int(r.max() / 60.0))

    df_float = df
    df = df_str
    df = df.reset_index()

    if col_to_subtract:  # subtract column from all the others
        df.columns = [("$\Delta$" + str(col)) if col not in ["ID", col_to_subtract] else col for col in df.columns]

    print(tabulate(df, headers='keys', tablefmt='psql'))

    output_path = output_path + out_f
    df.to_latex(mk_dir_return(output_path), caption=caption, float_format=float_nums, escape=False, index=False, na_rep=" ")
    # df.to_csv(output_path.replace(".txt", ".csv"))
    # replace Nan and zero values
    inplace_change(output_path, ' 0 &', ' nan &', ' &')

    '''
    % save in file and read into string
    figstr= "\begin{figure}[H]
                     \centering
                     \includegraphics[width=0.95\textwidth]{figures/small_datasets_5min/1468_random_sampling_2020-07-27T15_08_54.pdf}
                 \end{figure}"
    '''

    if plot:
        df_float = df_float.T
        ax = df_float.plot(grid=True,
                           # x=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
                           # xlim=(0, 10),
                           # xticks=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0]
                           )  # color=color)
        ax.set_xlabel("Sampling ratio")
        ax.set_ylabel("Macro-averaged F1 Score")
        ax.legend(bbox_to_anchor=(1, 1), title="Dataset").get_frame().set_linewidth(0.0)
        ax.set_xticks(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.5])  # [0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0])
        ax.set_xticklabels([0.0001, 0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0])
        plt.tight_layout()
        plt.savefig(mk_dir_return("paper_latex/figures/big_datasets_sampling_ratio.pdf"))
        plt.show()


def mk_dir_return(output_path):
    dir_path = os.path.dirname(output_path)
    if len(dir_path) > 0:
        os.makedirs(dir_path, exist_ok=True)
    return output_path


# adapted from TPOTBase to impute missing values
def _check_dataset(features, target, sample_weight=None):
    """Check if a dataset has a valid feature set and labels.

    Parameters
    ----------
    features: array-like {n_samples, n_features}
        Feature matrix
    target: array-like {n_samples} or None
        List of class labels for prediction
    sample_weight: array-like {n_samples} (optional)
        List of weights indicating relative importance
    Returns
    -------
    (features, target)
    """
    # Check sample_weight
    _imputed = False

    def _impute_values(features):
        """Impute missing values in a feature set.

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            A feature matrix

        Returns
        -------
        array-like {n_samples, n_features}
        """
        '''
        if self.verbosity > 1:
            print('Imputing missing values in feature set')

        if self._fitted_imputer is None:
            self.'''
        _fitted_imputer = SimpleImputer(strategy="median")
        _fitted_imputer.fit(features)

        return _fitted_imputer.transform(features)

    '''
    if sample_weight is not None:
        try: sample_weight = np.array(sample_weight).astype('float')
        except ValueError as e:
            raise ValueError('sample_weight could not be converted to float array: %s' % e)
        if np.any(np.isnan(sample_weight)):
            raise ValueError('sample_weight contained NaN values.')
        try: check_consistent_length(sample_weight, target)
        except ValueError as e:
            raise ValueError('sample_weight dimensions did not match target: %s' % e)

    # If features is a sparse matrix, do not apply imputation
    if sparse.issparse(features):
        if self.config_dict in [None, "TPOT light", "TPOT MDR"]:
            raise ValueError(
                'Not all operators in {} supports sparse matrix. '
                'Please use \"TPOT sparse\" for sparse matrix.'.format(self.config_dict)
            )
        elif self.config_dict != "TPOT sparse":
            print(
                'Warning: Since the input matrix is a sparse matrix, please makes sure all the operators in the '
                'customized config dictionary supports sparse matriies.'
            )
    else:
    '''
    if isinstance(features, np.ndarray):
        if np.any(np.isnan(features)):
            _imputed = True
    elif isinstance(features, pd.DataFrame):
        if features.isnull().values.any():
            _imputed = True

    if _imputed:
        features = _impute_values(features)

    try:
        if target is not None:
            from sklearn.utils import check_X_y
            X, y = check_X_y(features, target, accept_sparse=True, dtype=None)
            if _imputed:
                return X, y
            else:
                return features, target
        '''else:
            X = check_array(features, accept_sparse=True, dtype=None)
            if self._imputed:
                return X
            else:
                return features
        '''
    except (AssertionError, ValueError):
        raise ValueError(
            'Error: Input data is not in a valid format. Please confirm '
            'that the input data is scikit-learn compatible. For example, '
            'the features must be a 2-D array and target labels must be a '
            '1-D array.'
        )


def refit_best(results_dir="results_successful/refit_cv_iteration/",
               random_state=42,
               shuffle=True,
               groups=None,
               verbose=3,
               scorer='f1_macro',
               error_score=np.nan, testing=False,
               bold=True, datasets_table=False, plot=False, file_name=""):
    np.random.seed(random_state)
    random.seed(random_state)

    sys.stdout = open(
        'results_successful/refit_cv_iteration/' + get_valid_file_path() + '_pipelines_refitting' + '.txt', 'w')
    # for every dataset
    for f in listdir("./" + results_dir):
        if f.endswith('.csv') and f[0].isdigit():
            df = pd.read_csv(results_dir + f)
            # get dataset
            dataset = int(df['dataset'].iloc[0])
            print("#1 Dataset", dataset, "_______________")
            cv_max = int(df['cv_iter'].max() + 1)
            X, y, categorical_indicator, features, n_target_classes = dd.get_openml_data(dataset)
            X, y = _check_dataset(X, y)  # impute missing values like TPOTClassifier
            # split dataset
            X, y, groups = indexable(X, y, groups)

            cv = StratifiedKFold(
                cv_max,
                random_state=random_state,
                shuffle=True,
            )
            scorer = get_scorer(scoring=scorer)

            cv = check_cv(cv, y, classifier=True)
            # Store it as list as we will be iterating over the list multiple times
            cv_iter = list(cv.split(X, y, groups))

            cv_iter_nr = 0
            # for each cv iteration split
            results_list = [[] for _ in range(int(len(df) / cv_max))]
            cnt = 0
            for train, test in create_cv_iter_copy(cv_iter, random_state, shuffle):
                # for each pipeline on respective cv iteration
                print("#2 Fold", cv_iter_nr)  # , train, test, "_______________")
                cv_iter_rows = df.loc[df['cv_iter'] == cv_iter_nr]

                # For every row: check if experiment is already run -> take that value and add to list, else: run experiment
                for idx, index in enumerate(cv_iter_rows.index):
                    fit_estimator = cv_iter_rows['fitted_pipeline'][index]
                    print("#", "Dataset", dataset, "Fold", cv_iter_nr, "Nr", idx * cv_max + cv_iter_nr)
                    if fit_estimator is not np.nan:
                        print("pipeline =", fit_estimator)
                        # not yet evaluated
                        if np.isnan(cv_iter_rows["cv_iter_score_refitted"][index]):
                            if testing:
                                results = [cnt, 2, 3]
                                results_list[idx].append(results[:2])
                            else:
                                try:
                                    results = _fit_and_score(
                                        estimator=eval(fit_estimator), X=X, y=y, scorer=scorer, train=train, test=test,
                                        verbose=verbose,
                                        parameters=None, fit_params=None, return_train_score=False,
                                        error_score=error_score, return_times=True,
                                        return_estimator=False)
                                    print(results[:2])
                                    results_list[idx].append(results[:2])
                                except Exception as err:
                                    print(err)
                                    results_list[idx].append([None, None])
                        else:
                            print("Skipped: already evaluated")
                            results_list[idx].append([cv_iter_rows["cv_iter_score_refitted"][index],
                                                      cv_iter_rows["cv_iter_refit_time"][index]])
                    else:
                        print("# nan pipeline")
                        results_list[idx].append([None, None])
                    cnt = cnt + 1


                cv_iter_nr = cv_iter_nr + 1
            results_list_sorted = [item for sublist in results_list for item in sublist]
            df_results = pd.DataFrame(results_list_sorted, columns=["cv_iter_score_refitted", "cv_iter_refit_time"])

            # if running on already refitted -> rename old column
            if "cv_iter_refit_time" in df.columns:
                df = df.rename(columns={"cv_iter_score_refitted": "old_cv_iter_score_refitted",
                                        "cv_iter_refit_time": "old_cv_iter_refit_time"})
            df = pd.concat([df, df_results], axis=1, sort=False)
            # drop index column
            df = df.drop(df.columns[0], 1)
            # save everything in another csv aka attach to current df
            to_csv_and_print(df, "results_successful/refit_cv_iteration/" + f[:-4] + "_refit_cv_iteration" + (
                "_test" if testing else ""))

    sys.stdout = sys.__stdout__
    print("Finished")


def find_best_result_sampling_ratio(dataset_id, sampling_ratio,
                                    rerun_best=1,
                                    input_path="results/results_5min/", rerun_score_col='score_refitted'):
    filepaths = [input_path + f for f in os.listdir("./" + input_path[:-1]) if
                 f.endswith('.csv') and f.startswith(dataset_id + "_")]
    if filepaths:
        sampling_ratio = []
        print(filepaths, len(filepaths))
        df = pd.concat(map(pd.read_csv, filepaths))
        df.drop(columns=['sampling_method'])
        # use agg(['mean', 'count]) to also show how many folds were evaluated
        df = df.groupby(['dataset', 'sampling_ratio'])[rerun_score_col].agg(['mean']).reset_index()
        df = df.set_index(['dataset', 'sampling_ratio']).unstack('sampling_ratio')
        if len(df.columns) > 1: df = df.iloc[:, : -1]  # best ratio except for 1.0
        df = df.idxmax(axis=1)
        for row in df:
            sampling_ratio.append(row[1])
            if rerun_best == 2:
                sampling_ratio.append(1.0)
        print("____", dataset_id, sampling_ratio)
    return sampling_ratio


def get_distinct_datasets(files_short, files_long):
    d_list_short = []
    d_list_long = []
    for f in files_short:
        d = f.split("_")[0]
        if d not in d_list_short:
            d_list_short.append(d)
    for f in files_long:
        d = f.split("_")[0]
        if d not in d_list_long:
            d_list_long.append(d)

    return d_list_short, d_list_long


def compare_5min_60min(column_to_average, excluded_data=None, excluded_sampling_ratio=None, results_dir="results/",
                       file_name="_comparison_5min_60min", short_min=5, long_min=60):
    df0 = pd.DataFrame()
    best_ratios = []
    files_short = [f for f in listdir("./" + results_dir + "/results_" + str(short_min) + "min/")
                  if f.endswith('.csv') and f[0].isdigit()]
    files_long = [f for f in listdir("./" + results_dir + "/results_" + str(long_min) + "min/")
                  if f.endswith('.csv') and f[0].isdigit()]
    d_list_short, d_list_long = get_distinct_datasets(files_short, files_long)
    filepaths_short = [results_dir + "/results_" + str(short_min) + "min/" + f for f in
                       files_short if f.split("_")[0] in d_list_long]
    filepaths_long = [results_dir + "/results_" + str(long_min) + "min/" + f for f in
                       files_long if f.split("_")[0] in d_list_short]
    filepaths_pairs = {str(short_min): filepaths_short,
                       str(long_min): filepaths_long}

    for min in [str(short_min), str(long_min)]:
        filepaths = filepaths_pairs[min]
        print(filepaths, len(filepaths))
        df = pd.concat(map(pd.read_csv, filepaths))
        if excluded_sampling_ratio is None:
            excluded_sampling_ratio = []
        if excluded_data is None:
            excluded_data = []

        # delete rows for excluded datasets
        df = df[~df.dataset.isin(excluded_data)]

        df = df.dropna(subset=[
            "score"])  # drop null values when averaging pipelines explored since the amount was saved as 0 instead of NaN

        caption = df['sampling_method'].iloc[0]
        print(caption)
        df.drop(columns=['sampling_method'])
        # for i in df['dataset']:

        # use agg(['mean', 'count]) to also show how many folds were evaluated
        df = df.groupby(['dataset', 'sampling_ratio'])[[column_to_average, "pipelines_explored"]].agg(
            ['mean']).reset_index()

        df = df.set_index(['dataset', 'sampling_ratio']).unstack('sampling_ratio')

        output_path = results_dir + "/results_{}min/result_tables/".format(short_min) + str(datetime.now().isoformat())[:-7].replace(':',
                                                                                                    '_')
        df.rename(columns={'sampling_ratio': "ratio"}, inplace=True)
        df.replace(np.nan, '', regex=True, inplace=True)
        Δ = ""  # "$\Delta$"
        df.columns = [Δ + str(col[2]) + str(col[0]) for col in df.columns]

        if not len(best_ratios):
            best_ratios = [find_best_result_sampling_ratio(str(data), sampling_ratio=[],
                                                           input_path= results_dir + "/results_{}min/".format(short_min),
                                                           rerun_score_col='score_refitted')[0] for data in
                           df.index]  # \
            # if (ratio == "best") else ["1.0"] * len(df.index)

        print(best_ratios)
        c = 0
        l_best_ratio = []
        l_best_score_refitted = []
        l_best_pipelines_explored = []
        l_full_score_refitted = []
        l_full_pipelines_explored = []

        for i, r in df.iterrows():
            print(i, r)
            l_best_ratio.append(best_ratios[c])
            l_best_score_refitted.append(r[str(best_ratios[c]) + "score_refitted"])
            l_best_pipelines_explored.append(r[str(best_ratios[c]) + "pipelines_explored"])
            l_full_score_refitted.append(r["1.0" + "score_refitted"])
            l_full_pipelines_explored.append(r["1.0" + "pipelines_explored"])
            c = c + 1

        data2 = {("best" if min == str(short_min) else "full") + "_ratio": l_best_ratio if min == str(short_min) else "1.0",
                 min + "best_score_refitted": l_best_score_refitted,
                 min + "best_pipelines_explored": l_best_pipelines_explored,
                 min + "full_score_refitted": l_full_score_refitted,
                 min + "full_pipelines_explored": l_full_pipelines_explored
                 }
        df2 = pd.DataFrame(data2)

        df = df.reset_index()
        # df2 = df2.reindex(df.index)
        df2["dataset"] = df["dataset"]
        cols = ["dataset"] + df2.columns[:-1].tolist()
        df2 = df2[cols]

        if min == str(short_min):
            df0 = df2
        else:
            df0 = df0.merge(df2, on=["dataset"])
    #    df2.insert(0, 'dataset', df2.dataset(1))

    print(tabulate(df0, headers='keys', tablefmt='psql'))

    mk_dir_return(output_path)

    # float_nums = "{:0.3}".format
    df0 = df0.round({"{}best_score_refitted".format(short_min): 3, "{}best_pipelines_explored".format(short_min): 0,
                     "{}full_score_refitted".format(short_min): 3,
                     "{}full_pipelines_explored".format(short_min): 0,
                     "{}best_score_refitted".format(long_min): 3, "{}best_pipelines_explored".format(long_min): 0,
                     "{}full_score_refitted".format(long_min): 3,
                     "{}full_pipelines_explored".format(long_min): 0
                     })
    df0["{}best_pipelines_explored".format(short_min)] = df0["{}best_pipelines_explored".format(short_min)].astype(int)
    df0["{}full_pipelines_explored".format(short_min)] = df0["{}full_pipelines_explored".format(short_min)].astype(int)
    df0["{}best_pipelines_explored".format(long_min)] = df0["{}best_pipelines_explored".format(long_min)].astype(int)
    df0["{}full_pipelines_explored".format(long_min)] = df0["{}full_pipelines_explored".format(long_min)].astype(int)
    out_f = "_results_succesful" + file_name
    out_f = out_f + ".txt"
    output_path = output_path + out_f

    # rearrange to best
    df0 = df0[
        ["dataset",
         "best_ratio", "{}best_score_refitted".format(short_min), "{}best_pipelines_explored".format(short_min),
         "{}best_score_refitted".format(long_min), "{}best_pipelines_explored".format(long_min),
         "full_ratio", "{}full_score_refitted".format(short_min), "{}full_pipelines_explored".format(short_min),
         "{}full_score_refitted".format(long_min), "{}full_pipelines_explored".format(long_min)]
    ]
    df = df0

    # reindex according to size
    data_list = openml.datasets.get_datasets(df["dataset"], download_data=False)
    data = []
    for d in data_list:
        d1 = {"ID": d.dataset_id,
              "Name": d.name}
        d2 = {
            k.replace("NumberOf", "\#"): int(v) for k, v in d.qualities.items() if
            k in ['NumberOfInstances', 'NumberOfClasses', 'NumberOfFeatures']
        }
        data.append({**d1, **d2})

    df_datasets = pd.DataFrame(data)
    df_datasets.set_index("ID", inplace=True)
    df_datasets.sort_values(axis=0, by=str("\#Instances"), inplace=True)

    df.set_index("dataset", inplace=True)
    df = df.reindex(index=df_datasets.index)
    df.reset_index(inplace=True)

    print(tabulate(df, headers='keys', tablefmt='psql'))

    df.to_latex(output_path, caption=caption,  # float_format=float_nums,
                escape=False, index=False, na_rep=" ")
    # replace Nan and zero values
    inplace_change(output_path, ' 0 &', ' nan &', ' &')


def analyze_counts_single_run(row, counts=None):
    if counts is None:
        counts = {}
    eval_individuals = row['evaluated_individuals']
    # print(eval_individuals)
    for k, v in eval_individuals.items():
        if v['operator_count'] == 5000: continue
        operators = k.split('(')  # [:-1]  # drop 'input_matrix), ... )
        for op in operators[:v['operator_count']]:
            if (op.startswith('input_matrix')): continue
            if op in counts:
                counts[op] += 1
            else:
                counts.update({op: 1})
    return counts, row['pipelines_explored']


def analyze_counts_all(orig_results_dir="results/results_5min/pkl/",
                       read_existing=False,
                       short_min=5):
    results_dir = orig_results_dir + ('pipeline_analysis/' if read_existing else '')
    filepaths = [results_dir + f for f in listdir("./" + results_dir) if f.endswith('.csv' if read_existing else '.pkl')
                 and f[0].isdigit()
                 ]
    all_df = []
    avg_operators = []
    covered_datasets = []
    total_pl_explored = 0
    total_operators = 0
    for f in filepaths:
        print("_", f)
        if read_existing:
           df = pd.read_csv(f, index_col=0)
        else:
            df, data, pipelines_explored, operators = analyze_counts_dataset(f, short_min)                 # skips 60 min experiments and half run experiments
            if df is None: continue # skip 60 min experiments
            total_pl_explored += pipelines_explored
            total_operators += operators
            output_path = results_dir + "pipeline_analysis/" + data + '_operator_frequencies_' + str(
                datetime.now().isoformat())[:-7].replace(':',
                                                         '_') + '.csv'

            covered_datasets.append(int(data))
            df.to_csv(mk_dir_return(output_path) )

        avg_op = df.loc["total_operators", ][2:].div(df.loc["total_pipelines", ][2:])
        avg_operators.append(avg_op)     # total operators / total pipelines for each dataset
        all_df.append(df.drop(['total_operators', 'total_pipelines'], 0))

    if not read_existing:
        print('Total Pipelines', total_pl_explored)
        print('Total Operators', total_operators)
    df_avg_operators = pd.DataFrame(avg_operators)
    print('Average operators per pipeline:', [("{:0.2f}".format(a), "{:0.2f}".format(s)) for a, s, in
                                              zip(np.nanmean(df_avg_operators, 0), np.std(df_avg_operators, 0))])

    from functools import reduce
    #df_sum = reduce(lambda x, y: x.add(y, fill_value=0), all_df)
    df_sum = reduce(lambda x, y: x.astype(float).add(y.astype(float), fill_value=0), all_df)

    for col in df_sum.columns:
        if col.startswith('sampling_ratio'):
            df_sum[col] = df_sum[col] / df_sum[col].sum()
    df_sum['max_time_mins'] = all_df[0]['max_time_mins']
    df_sum['dataset'] = ', '.join([ str(d['dataset'][0]) for d in all_df])
    df_sum.to_csv(orig_results_dir + "pipeline_analysis/all_operator_frequencies_" + str(
            datetime.now().isoformat())[:-7].replace(':', '_') + '.csv')


    plot_cols = [c for c in df_sum.columns if c.startswith('sampling_ratio')]
    df_plot = df_sum[plot_cols]
    df_plot.columns = [c[14:] for c in plot_cols]
    df_plot = df_plot.transpose()
    # Get 5 most occuring operators for each sampling ratio
    nlargest = 5
    order = np.argsort(-df_plot.values, axis=1)[:, :nlargest]       # n most occuring operators for each sampling ratio
    unique_cols = pd.unique(order.ravel())
    print(unique_cols)
    print(df_plot.columns[unique_cols])
    df_plot_nlargest = df_plot[df_plot.columns[unique_cols]]

    plot_frequencies(df_plot_nlargest, 'line', False, True)
    #plot_frequencies(df_plot_nlargest, 'bar', True, True)

    print("Finished plotting")

# https://stackoverflow.com/a/43439132/14702949
def plot_frequencies(df_plot_nlargest, kind='bar', stacked=True, legend=True, font_size="medium", n_col=3, bbox=(0,1.025,1,0.25), loc="center",):
    ax = df_plot_nlargest.plot(kind=kind, stacked=stacked, style=['-', '--', '-.', ':', '-', '--', '-.', ':'])
    plt.xlabel('Sampling Ratio')
    plt.ylabel('Relative Frequency')
    plt.xticks(rotation=0)
    if legend:
        plt.legend(
                   fontsize=font_size,
                   ncol = n_col,
                   bbox_to_anchor=bbox,
                   loc = loc,
                   )
    if kind == 'line':
        ax.set_xticks(np.arange(len(df_plot_nlargest)))
        ax.set_xticklabels(df_plot_nlargest.index)
    plt.tight_layout()
    plt.savefig(mk_dir_return("paper_latex/figures/pa_operator_frequencies_" + kind + ".pdf"))
    plt.show()

def analyze_counts_dataset(filepath, short_min=5):
    if (not filepath):
        print("No file to convert provided.")
        return None, None, None, None
    if filepath.endswith(".pkl"):
        pkl_obj = pickle.loads(open(filepath, 'rb').read())

    # print(tabulate(pkl_obj, headers='keys', tablefmt='psql'))
    max_time_mins = pkl_obj.iloc[0, 11]
    n_runs = np.shape(pkl_obj)[0]
    if str(max_time_mins) != str(short_min) or n_runs < 25 : return None, None, None, None # skip 60 min experiments and half run experiments
    dataset = pkl_obj.iloc[0, 4]
    print(max_time_mins, n_runs, dataset)
    data_results = {'dataset': dataset,
                    'max_time_mins': max_time_mins}
    counts = {}
    sr = -1
    total_pl_explored = 0
    sr_pl_explored = 0
    total_operators = 0
    for idx, row in pkl_obj.iterrows():
        if row['fitted_pipeline']:
            new_sr = row['sampling_ratio']
            if new_sr != sr and sr != -1:       # ensures that it works even when some folds failed
                total = sum(counts.values(), 0.0)
                total_operators += total
                abs_counts = {k: v / total for k, v in counts.items()}

                data_results.update({'sampling_ratio' + str(sr): {
                    **abs_counts,
                    'total_operators': total,
                    'total_pipelines': sr_pl_explored,
                }})
                counts={}
                total_pl_explored += sr_pl_explored
                sr_pl_explored = 0
            counts, pipelines_explored = analyze_counts_single_run(row, counts)
            sr_pl_explored += pipelines_explored
            sr = new_sr

    # Treat 1.0 seperately
    total = sum(counts.values(), 0.0)
    total_operators += total
    total_pl_explored += sr_pl_explored
    abs_counts = {k: v / total for k, v in counts.items()}
    data_results.update({'sampling_ratio' + str(sr): {
        **abs_counts,
        'total_operators': total,
        'total_pipelines': sr_pl_explored,
    }})

    # print(data_results)
    df = pd.DataFrame(data_results)
    pkl_obj = None
    return df, dataset, total_pl_explored, total_operators


if __name__ == "__main__":
    print(sys.argv)
    read_existing = str(sys.argv[2]) == "1"
    short_min = sys.argv[3]
    long_min = sys.argv[4]
    kwargs = {
        "excluded_data": []
             ,
        # "excluded_sampling_ratio": ["0.0001", "0.001"],
        # "col_to_subtract": "1.0",
        # "as_percentage": True,
        "results_dir": sys.argv[1] + "/results_{}min/".format(short_min),
        "bold": True,
        "testing": False,
        "long_min": long_min,
    }

    # dataset overview and all scores (no datasets excluded)
    results_to_latex('score_refitted',
                     plot=False, datasets_table=True,
                     **kwargs)

    # amount of pipelines explored
    results_to_latex('pipelines_explored', int_only=True, **kwargs)

    # train fit time
    compare_5min_60min("score_refitted", excluded_data=sys.argv[5:] if len(sys.argv) > 5 else []
                       , results_dir=sys.argv[1] + "/",
                       short_min=short_min,
                       long_min=long_min)

    # difference as percent to full ratio for only the big datasets
    kwargs = {**kwargs, **{
        "excluded_sampling_ratio": ["0.0001", "0.001"],
        "col_to_subtract": "1.0",
        "as_percentage": True,
    }}
    results_to_latex('score_refitted',
                     file_name="_diff",
                     **kwargs)

    analyze_counts_all(sys.argv[1] + '/results_{}min/pkl/'.format(short_min), read_existing=read_existing, short_min=short_min)
