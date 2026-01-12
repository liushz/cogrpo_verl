# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

# setup.py is the fallback installation script when pyproject.toml does not work
from pathlib import Path

from setuptools import find_packages, setup

__version__ = "0.0.1"

install_requires = [
    "accelerate",
    "codetiming",
    "datasets",
    "dill",
    "hydra-core",
    "numpy",
    "pandas",
    "peft",
    "pyarrow>=19.0.0",
    "pybind11",
    "pylatexenc",
    "ray[default]>=2.41.0",
    "torchdata",
    "tensordict<=0.6.2",
    "transformers",
    "wandb",
    "packaging>=20.0",
    "vllm==0.10.1.1",
    # "sglang[srt,openai]==0.4.6.post5",
    "torch-memory-saver>=0.0.5",
    "latex2sympy2_extended==1.10.1",
    "liger-kernel==0.5.10",
    "flashinfer-python==0.2.9",
    "word2number==1.1",
    "flash_attn==2.8.2"
]

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="repro",
    version=__version__,
    package_dir={"": "."},
    packages=find_packages(where="."),
    url="https://github.com/open-compass/RePro",
    license="Apache 2.0",
    author="Junnan Liu",
    author_email="to.liujn@outlook.com",
    description="RePro",
    install_requires=install_requires,
    package_data={
        "verl": ["trainer/config/*.yaml"],
    },
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)