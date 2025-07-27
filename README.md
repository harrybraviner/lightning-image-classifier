Lightning Image Classifer
=========================

An image classifier using the PyTorch Lightning framework.

This repo doesn't contain anything groundbreaking, but it should serve as a good, clean example of organizing such a project.

Current status: Rough draft.

# Setup

Install `uv` using
```
pip install uv
```
or using any other method listed at [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv).

Once `uv` is installed, run the following from within the cloned repo directory to install the project dependencies:
```
uv sync
```
You may have to modify the `pyproject.toml` file to install a non-CUDA version of torch if you are running this command on a machine that does not have a GPU.

