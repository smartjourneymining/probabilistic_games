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
