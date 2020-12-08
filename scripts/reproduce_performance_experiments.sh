#!/usr/bin/env bash
# this file was adapted from: https://github.com/josepablocam/ams/tree/master/experiments

source scripts/folder_setup.sh

mkdir -p ${RESULTS}
# 0: run specified sampling_ratios
# 1: rerun only best
# 2: rerun best and full (1.0)
RERUN_BEST=0
MAX_TIME_MINS=5
TEST_MODE=0
BENCHMARK_SCORING=0
FIND_BEST_RATIO=5
while getopts ":t:b:m:r:f:" opt; do
    case $opt in
    t) TEST_MODE="$OPTARG"
        echo 'Starting TEST_MODE mode for local';;
    b) BENCHMARK_SCORING="$OPTARG";;
    m) MAX_TIME_MINS="$OPTARG";;
    r) RERUN_BEST="$OPTARG";;
    f) FIND_BEST_RATIO="$OPTARG";;
    *) echo 'Error in command line parsing' >&2
        exit 1
    esac
done
shift "$((OPTIND - 1))"
# If rerun_best > 0: Which column (score_refitted, cv_iter_score, ...) to use in order to pick the highest scoring sampling ratio.
RERUN_SCORE_COL="score_refitted"
INPUT_PATH="results/results_${FIND_BEST_RATIO}min/"
echo ${TEST_MODE}
echo ${BENCHMARK_SCORING}
if [ "$TEST_MODE" -eq 1 ]; then
    TPOT_DATASETS[0]="1468"
    TPOT_DATASETS[1]="1464"
    TPOT_DATASETS[2]="12"
    SAMPLING_RATIOS="0.0001 0.1 0.3 0.5 1.0"
else
    SAMPLING_RATIOS="0.0001 0.001 0.01 0.05 0.1 0.15 0.2 0.3 0.5 1.0"
    TPOT_DATASETS[3]="1468"   # cnae-9                     1080 documents of free text business descriptions of Brazilian companies
    TPOT_DATASETS[4]="12"     # mfeat-factors              2000 One of a set of 6 datasets describing features of handwritten numerals (0 - 9) extracted from a collection of Dutch utility maps
    TPOT_DATASETS[5]="3"      # kr-vs-kp                   3196 Chess End-Game -- King+Rook versus King+Pawn
    TPOT_DATASETS[6]="1489"   # phoneme                    5404 distinguish between nasal (class 0) and oral (class 1) sounds
    TPOT_DATASETS[7]="40668"  # connect-4                 67557 Contains all legal 8-ply positions in the game of connect-4 in which neither player has won yet
    TPOT_DATASETS[8]="41138"  # APSFailure                76000 APS Failure and Operational Data for Scania Trucks
    TPOT_DATASETS[9]="41168"  # jannis                    83733 part of the autoML chalearn challenge
    TPOT_DATASETS[10]="23517" # numerai28.6               96320 Encrypted Stock Market Training Data from Numer.ai
    TPOT_DATASETS[11]="23512" # higgs                     98050 Higgs Boson detection data
    TPOT_DATASETS[12]="41150" # MiniBooNE                130064 Distinguish electron neutrinos (signal) from muon neutrinos (background)
    TPOT_DATASETS[13]="1483"  # ldpa                     164860 Localization Data for Person Activity Data Set
    TPOT_DATASETS[14]="1503"  # spoken-arabic-digit      263256 Time series of mel-frequency cepstrum coefficients (MFCCs) corresponding to spoken Arabic digits
    TPOT_DATASETS[15]="1169"  # airlines                 539383 Predict whether a given flight will be delayed
    TPOT_DATASETS[16]="1596"  # covertype                581012 Predicting forest cover type from cartographic variables only
    TPOT_DATASETS[17]="42468" # hls4ml_lhc_jets_hlf      830000 Identify jets of particles from the LHC, created for the study of ultra low latency inference with hls4ml.
    TPOT_DATASETS[18]="354"   # poker                   1025010 Preprocesssed poker dataset (see also 1569 or 1567)
fi

# evaluation params
CV=5
SEED=42
N_JOBS=-1
SCORING_FUN="f1_macro"

#
SAMPLING_METHOD[0]="stratify",
SAMPLING_METHOD[1]="random",
SAMPLING_METHOD[2]="cluster-kmeans"
SAMPLING_METHOD[3]="oversampling"

#tsp params
N_LOOPS=$((24*7))
TIME_SLEEP=3600
TIME_MAX_FREEZE_MINS=$((MAX_TIME_MINS*20))
N_RETRY=5
# number of datasets to evaluate at the same time (higher means earlier finishing, but may evaluate less pipelines if cores are under full load)
N_PROC=2

# run using task-spooler
# https://vicerveza.homeunix.net/~viric/soft/ts/
tsp -S ${N_PROC}

for dataset in "${!TPOT_DATASETS[@]}"; do
    for search in tpot; do # random
        output_folder="${RESULTS}/results_${MAX_TIME_MINS}min" #${search}/${TPOT_DATASETS[dataset]}"
        echo "Running experiments for folder: ${output_folder}"
        # in case doesn't exist
        mkdir -p ${output_folder}

        # we label our tasks with -L and are thus able to rerun them
        # the label must be an int since we use arr_cnt[<label>] to increment the respective retry count
        tsp -L ${dataset} python experiments/run_experiment.py \
            --search ${search} \
            --dataset ${TPOT_DATASETS[dataset]} \
            --cv ${CV} \
            --max_time_mins ${MAX_TIME_MINS} \
            --random_state ${SEED} \
            --name "random_sampling" \
            --sampling_method "random" \
            --sampling_ratio ${SAMPLING_RATIOS} \
            --scoring ${SCORING_FUN} \
            --n_jobs ${N_JOBS} \
            --output "${output_folder}/${TPOT_DATASETS[dataset]}_random_sampling_" \
            --config "TPOT" \
            --benchmark_scoring ${BENCHMARK_SCORING} \
            --test ${TEST_MODE} \
            --rerun_best ${RERUN_BEST} \
            --rerun_score_col ${RERUN_SCORE_COL} \
            --input_path ${INPUT_PATH}

        # same with stratified_sampling
    done
done

tsp
n_tsp_lines=999         # for first iteration
sec_checkpoint=$(date +%s)
# run for 3 days every 60 min
i=0
while ((i <= N_LOOPS && n_tsp_lines > 1)); do                                                         # if there are tasks and the time is not over
    sleep $((MAX_TIME_MINS*60))
    tsp_output=$(tsp -l)
    changed=false
    while read line; do                                                                                 # for each line of tsp table
        arr_line=(${line//;/ })
        if [[ ${arr_line[0]} = "ID" ]]; then
            continue
        fi
        err_nr=${arr_line[3]}
        tsp_output_file=${arr_line[2]}
        tsp_lbl=$(grep -Po '\[\K[^][]*(?=])' <<<${line})
        tsp_py_cmd=${line#*]}

        # Try to unfreeze every MAX_TIME_MINS
        if [[ "${arr_line[1]}" = "running" ]]; then
            mins_since_mod=$(( ($(date +%s) - $(date +%s -r ${tsp_output_file})) / 60 ))
            if [[ ${mins_since_mod} -gt ${TIME_MAX_FREEZE_MINS} ]]; then                                           # if the output output has not changed for TIME_MAX_FREEZE_MINS
                echo "~~~~NO OUTPUT FOR" ${mins_since_mod} "MINUTES. SENDING CTRL + C SIGNAL" "["${tsp_lbl}"]"$tsp_py_cmd
                kill -2 $(tsp -p ${arr_line[0]})                                                                   # send CTRL + C aka stop current generation/TPOT run
            fi
            if [[ ${mins_since_mod} -gt $((TIME_MAX_FREEZE_MINS*4)) ]]; then
                echo "~~~~NO OUTPUT FOR" ${mins_since_mod} "MINUTES. SENDING TERMINATE SIGNAL" "["${tsp_lbl}"]"$tsp_py_cmd
                kill -15 $(tsp -p ${arr_line[0]})                                                                  # send TERMINATE aka soft stop the run for this dataset
            fi
            if [[ ${mins_since_mod} -gt $((TIME_MAX_FREEZE_MINS*8)) ]]; then
                echo "~~~~NO OUTPUT FOR" ${mins_since_mod} "MINUTES. SENDING KILL SIGNAL" "["${tsp_lbl}"]"$tsp_py_cmd
                kill -9 $(tsp -p ${arr_line[0]})                                                                   # send KILL aka hard stop the run for this dataset
            fi
        fi

        # Re-run cancelled tasks every TIME_SLEEP
        if [[ $(($(date +%s) - ${sec_checkpoint})) -gt ${TIME_SLEEP} || ${changed} = true ]]; then
            if [[ ${err_nr} =~ ^[+-]?[0-9]+$ ]]; then                                                     # if task is finished
                if [[ ${err_nr} -ne 0 ]]; then
                    arr_cnt[${tsp_lbl}]=$((arr_cnt[${tsp_lbl}] + 1))
                    if [ ${arr_cnt[${tsp_lbl}]} -le ${N_RETRY} ]; then
                        echo "____Retry:" ${arr_cnt[${tsp_lbl}]} "[${tsp_lbl}]"$tsp_py_cmd
                        tsp -t ${arr_line[0]}                                                            # output last 10 lines (see /tmp/ for more)
                        tsp -L ${tsp_lbl} ${tsp_py_cmd}                                                  # re-run failed
                    else
                        echo "ERROR: REACHED RETRY LIMIT FOR" $tsp_py_cmd
                    fi
                fi
                tsp -r ${arr_line[0]}                                                                   # remove finished and failed tasks
            fi
            if [ ${changed} = false ]; then
                printf "%*s$(date +"%Y-%m-%dT%H:%M:%S")\n" "${COLUMNS:-$(($(tput cols) - 19))}" '' | tr ' ' -
                echo "$tsp_output"
                changed=true
                i=$(( i+1 ))
                sec_checkpoint=$(date +%s)
            fi
        fi
    done <<<"$tsp_output"
    n_tsp_lines=$(wc -l < <(tsp -l))
done
