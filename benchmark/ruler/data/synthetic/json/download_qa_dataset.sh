# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

wget --https-only --tries=3 --timeout=30 \
    https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O squad.json

# The original CMU host (curtis.ml.cmu.edu) is no longer reliably reachable.
# This mirror contains the same HotpotQA v1 distractor development JSON.
wget --https-only --tries=3 --timeout=30 \
    'https://huggingface.co/datasets/namlh2004/hotpotqa/resolve/main/hotpot_dev_distractor_v1.json?download=true' \
    -O hotpotqa.json
