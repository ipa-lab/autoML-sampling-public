#!/usr/bin/env bash

conda env export --no-builds --name autoML-sampling-env | grep -v -E "(^prefix: )|(icc_rt)|(vc)|(vs2015_runtime)|( m2-)|( msys2-conda-epoch)|(win_inet_pton)|(wincertstore)" > environment.yml