Prostate MRI Deep Learning Benchmark
This repository contains code and experiment scripts for benchmarking deep learning models on prostate MRI data for tasks such as lesion detection, segmentation, or classification. The project is implemented in Python with experiment orchestration via simple command files and a batch runner.

Repository structure
src/models/ – Model architectures and related utilities for prostate MRI deep learning experiments.

experiments/ – Configuration files, experiment definitions, and logs for different runs.

requirements.txt – Python package dependencies required to run the project.

commands.txt – List of CLI commands that define the full experimental pipeline (training, evaluation, etc.).

run_all_commands.bat – Convenience script (Windows) to execute all commands from commands.txt sequentially.

# On Windows
run_all_commands.bat
Alternatively, you can execute individual experiments by running the corresponding lines from commands.txt directly in your terminal.

Data
This benchmark assumes access to prostate MRI datasets that are not included in the repository. Please configure your local paths and dataset structure according to the instructions in the experiment configuration files (e.g., under experiments/) before running any commands.

Contributing
Issues and pull requests are welcome for improving model implementations, adding new baselines, or extending experiment coverage. Before submitting changes, please ensure that your code is formatted consistently and that all relevant experiments still run successfully.

License
Add your chosen license here (for example, MIT, Apache 2.0, or a custom academic license), and include the corresponding LICENSE file in the repository.
