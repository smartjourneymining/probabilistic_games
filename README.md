# Stochastic Games for User Journeys
This is the repository for the FM2024 submission ``Stochastic Games for User Journeys'' by Kobialka, Pferscher, Bergersen, Johnsen, and Tapia Tarifa.

## Outline
.  
├── data : Folder for event logs  
├── greps : Output folder for the GrepS case study, contains Figures 3, 4 and 5  
├── journepy : Journey analysis library with parsing modules  
├── mc_outputs : Stores the model checking outputs for reproducability  
├── out : General output folder, contains Figures 6 and 7  
├── queries : Used model checking queries in PRISM-games format  


## Requirements
All experiments were tested with Python 3.10.12. To install all required libraries run
```
pip install -r requirements.txt
```
For visualizations are additional libraries used:
- [plotly](https://plotly.com/python/getting-started/) for Sankey diagrams
- [pygraphviz](https://pygraphviz.github.io/documentation/stable/install.html) for plotting graphs

## Reproduce Case Studies:

1. Download data sources into `data`
    - [GrepS](https://zenodo.org/records/6962413/files/data.csv?download=1)
    - [BPIC'17](https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884)

2. Install PRISM-games 3.2.1
    - [PRISM-games](https://www.prismmodelchecker.org/games/download.php)

3. Update the links in the second cell of `io_alergia_greps.ipynb` and `io_alergia_bpic17.ipynb` to your local installs.

4. Run both notebooks to reproduce all experiments and plots.
   Due to memory consumption in BPIC'17 needs the Java maximal memory be increaesd to 8GB with `-javamaxmem 8g`. 
    These cells are currently commented out in the notebook.

## Complementary information
We present complementary information omitted in the paper due to the page limit.

### Greps 

The full stochastic user journey game for [GrepS](/greps/greps-example_environment_actions.png) with touchpoint names, transition probabilities and transition names:
![Full GrepS SUJG](/greps/greps-example_environment_actions.png)

### BPIC'17

The gas-by-step exepriment for BPIC'17:
![Stepwise gas bounds](/out/bpic_17_steps.png)

The bounded success probability for BPIC'17, BPIC'17-1 is in solid lines and BPIC'17-2 in dashed lines:
![Bounded success probability](/out/bpic_bounded.png)

## Docker Instructions
To build your own Docker image from scratch, run 
```
docker build -t probabilistic_games .
```
The docker image requires to download the datasets and [Prism-games](https://www.prismmodelchecker.org/dl/prism-games-3.2.1-src.tar.gz).
The datasets need to be contained in the `data` folder and Prism-games in the project folder.
For further details and instructions for running the Docker image, please see [Docker_README.md](Docker_README.md).

## License
Our code is licensed under the GNU General Public License v2, [GPLv2](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html).
The BPIC'17 dataset is licensed under the [4TU General Terms of Use](https://data.4tu.nl/articles/_/12721292/1) license.
