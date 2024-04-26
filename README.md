# Stochastic Games for User Journeys
This is the repository for the FM2024 submission ``Stochastic Games for User Journeys'' by Kobialka, Pferscher, Bergersen, Johsen, and Tapia Tarifa.

## Outline
.
├── data : Folder for event logs
├── greps : Output folder for the GrepS case study, contains Figures 3, 4 and 5
├── journepy : Journey analysis library with parsing modules
├── mc_outputs : Stores the model checking outputs for reproducability
├── out : General output folder, contains Figures 6 and 7
├── queries : Used model checking queries in PRISM-games format


## Requirements
All experiments were tested with Python 3.10.6. To install all required libraries run
```
pip install -r requirements.txt
```

## First steps:

1. Download data sources into `data`
- [GrepS](https://zenodo.org/records/6962413/files/data.csv?download=1)
- [BPIC'17](https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884)

2. Install PRISM-games 3.2.1
- [PRISM-games](https://www.prismmodelchecker.org/games/download.php)

3. Update the links in the second cell of `io_alergia_greps.ipynb` and `io_alergia_bpic17.ipynb` to your local installs.
