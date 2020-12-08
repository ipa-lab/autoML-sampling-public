# Reproducing Experiments

We provide scripts to reproduce our experiments. 
As the experiments require immense computational resources taking on the order of multiple weeks to run,
we also publish our experimental dataset under the following DOI:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4292740.svg)](https://doi.org/10.5281/zenodo.4292740)

To analyze the results of our experimental dataset, simply place the extracted files (keeping the folder structure) in `results/`. 

### Setup 
All the following steps are to be run from the root directory. 
First we install the necessary packages and resources.
```bash
$ bash scripts/setup.sh
```

To analyze the results without rerunning the experiments (and instead using our experimental data that we provide above),
simply run
```bash
$ python extract_pkl_csv.py results 0 5 60 
```
which will create various tables, figures, etc. (see below for more information).
 
### 5 Minute Experiments
To reproduce the experiments we start by running the 5 minute experiments on 10 sampling ratios. 
Executing the following command will create the respective csv,
pkl and pdf files for the 5 minute experiments in `results/results_5min` and `results/results_5min/pkl`.
```bash
$ bash scripts/reproduce_performance_experiments.sh -m 5 -r 0
```

Failed experiments will be reattempted automatically for a few times.
Before proceeding ensure that all experiments have finished successfully.
To rerun failed experiments manually, simply rerun the previous command after 
removing the successfully evaluated datasets (e.g. by commenting out: `# TPOT_DATASETS[5]="3"`).

### 60 Minute Experiments
Based on the results of the 5 minute experiments we can then run the 60 
minute experiments for the optimal found sampling ratio and the full dataset.
```bash
$ bash scripts/reproduce_performance_experiments.sh -m 60 -r 2 -f 5
```

### Figures, Tables, Data Analysis
To extract the tables and figures of our data we can execute the following command.
The argument `0` indicates that we read in the pkl files. 
```bash
$ python extract_pkl_csv.py results 0 5 60
```
As this needs a lot of RAM (pkl files of up to 19GB), we also provide a way 
to read in the already extracted csv results (`results_paper/results_5min/pkl/pipeline_analysis`) 
by using argument`1` instead of `0`.
```bash
$ python extract_pkl_csv.py results_paper 1 5 60
```
The first extract command creates various files in the following directories:

* Tables `results/results_5min/tables`
* Figures `paper_latex/figures`
* Pipeline Analyis (only when using argument `0`) `results/results_5min/pkl/pipeline_analysis`

