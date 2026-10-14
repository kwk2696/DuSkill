import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, "README.md")
if os.path.exists(readme_path):
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()
    long_description_content_type = "text/markdown"
else:
    long_description = ""
    long_description_content_type = "text/plain"

setup(
    name="duskill",
    version="1.0.0",
    description="DuSkill: Robust Policy Learning via Offline Skill Diffusion (AAAI 2024)",
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    packages=[package for package in find_packages() if package.startswith("duskill")],
    python_requires=">=3.8",
    # NOTE: `sb3-jax-haiku` (https://github.com/kwk2696/sb3-jax-haiku) and the
    # vendored Meta-World fork in `duskill/envs/metaworld` are not on PyPI and
    # must be installed from source. See requirements.txt.
    install_requires=[
        "jax",
        "jaxlib",
        "dm-haiku",
        "optax",
        "numpy",
        "scipy",
        "gym<0.26",
        "stable-baselines3<2.0",
        "transformers",
        "wandb",
        "opencv-python",
        "tqdm",
        "PyYAML",
    ],
)
