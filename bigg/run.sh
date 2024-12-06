# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
set -e
set -x

rm -rf lib bin include share

virtualenv -p python3 .
source ./bin/activate

pip install -r ./requirements.txt

pip uninstall numpy
pip install "numpy<2.0.0"

pip install -e .

# Install PyTorch 1.9.0 which has better compatibility
pip install torch==1.9.0 torchvision==0.10.0
pip install torch-scatter==2.0.8

# Build torch-scatter from source as fallback
pip install --no-binary torch-scatter torch-scatter

python -m bigg.unit_test.lib_test
