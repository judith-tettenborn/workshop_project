# Workshop reproducible code: example code from my controlled release experiments

This project processes methane measurement timeseries obtained by mobile measurements in controlled release experiments using python. It identifies transects of methane plumes and after a manual quality check it calculates maximum enhancement and integrated spatial peak area per transect. Then it derives the statistical model to estimate emission rates from spatial integrated peak areas. Further, statistical analysis and data visualisations are carried out.

OS: tested on Windows

## Prerequisits
- python (3.10.13)
- matplotlib (3.8.0)
- numpy (1.25.0)
- pandas (1.5.2)
- tilemapbase (0.4.7)
- scipy (1.10.1)
- geopy (2.3.0)

see requirements.txt and environment.yml files

## Installation
Instructions for setting up the environment (e.g., required dependencies, how to install).

It is recommended to set up a virtual environment (find more information [here](https://docs.python.org/3/library/venv.html) or [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) when using anaconda)

Clone the repository.
```sh
git clone https://github.com/judith-tettenborn/workshop_project.git
cd workshop_project
```
Install the necessary packages, either using
conda (recommended)
```sh
conda env create -f environment.yml
```
or pip
```sh
pip install -r requirements.txt
```

## Usage
How to run the scripts or use the model.
Example commands or workflow.

Navigate to the project 'workshop_project' and run the script from there in order to get the right paths to data and output folders.

## Data
Describe any datasets used (if applicable)

## Project Structure

The project structure distinguishes three kinds of folders:
- read-only (RO): not edited by either code or researcher
- human-writeable (HW): edited by the researcher only.
- project-generated (PG): folders generated when running the code; these folders can be deleted or emptied and will be completely reconstituted as the project is run.


```
.
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
├── data               <- All project data, ignored by git
│   ├── raw            <- Methane timeseries and gps data per experiment (RO)
│   ├── processed      <- The processed data, to use in Script 2 to create final data (PG/HW)
│   ├── final          <- The final data containing the extracted and quality-checked methane enhancements and associated chracteristics (PG)
├── docs               <- Documentation notebook for users (HW)
│   ├── manuscript     <- Manuscript source, e.g., LaTeX, Markdown, etc. (HW)
│   └── reports        <- Other project reports and notebooks (e.g. Jupyter, .Rmd) (HW)
├── results
│   ├── figures        <- Figures for the manuscript or reports (PG)
│   └── statistics     <- Other output for the manuscript (PG)
├── src                <- Source code for this project (HW)
│   ├── helper_functions
│   ├── peak_analysis
│   ├── plotting
│   └── preprocessing

```

## Add a citation file
Create a citation file for your repository using [cffinit](https://citation-file-format.github.io/cff-initializer-javascript/#/)
TODO

## License

This project is licensed under the terms of the [MIT License](/LICENSE).
TODO
