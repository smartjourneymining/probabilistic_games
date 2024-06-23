# Stochastic Games for User Journeys (Artifact)
This is the artifact for the FM2024 publication "Stochastic Games for User Journeys" by Kobialka, Pferscher, Bergersen, Johnsen, and Tapia Tarifa. We provide the artifact as a Docker container that includes all required data, tools and source code. 

## Installing Docker
Docker is a virtualization software, distributed [here](https://docs.docker.com/engine/install/).
Please follow the installation instructions for your machine.

## Load Docker container
Ensure that docker is running by executing ```docker --version ``` in the terminal. 
To load the docker container, execute the following command:
```
docker load --input stochastic-user-journey-games.tar  
```
You can check with ```docker images``` if the docker image is loaded.

## Run Docker container
To run the container replace ```path/to/your/output/repository``` by any directory on your local machine to which the generated plots, figures and tables should be exported. 

```
docker run -v 'path/to/your/output/repository:/home/out' --platform linux/amd64 stochastic-user-journey-games:latest
```

To run the fast execution, taking 90 min, append the short execution script to your call.

```
--short_execution="True"
```

## Reproducability
The artefact contains all necessary data and scripts to reproduce the findings in our paper.
Specifically, the artefact reproduces:
- Table 2
- Figures 3, 4, 5, 6 and 7.

Under the short execution, Figure 4c and bpic_bounded.png differ from the presented ones due to increased step-sizes and reduced model-sizes.

Additional plots shown on the github page are also included.

After running the image, the linked `out` folder contains an new `bpic17` and `greps` folder.
The png's for Figures 3, 4 and 5 are contained in `out/greps`, for Figures 6 and 7 in `out/bpic17.
A markdownfile for Table 2 is generated in `out`.

*Important:* Please note that the produced png's' for Fig. 5 and 7 are not equal to those in the paper.
The used Sankey library allows adjusting the nodes in the plot before saving the image to improve readability.
Therefore, *.html files are additionally generated which allow adjusting the nodes to the layout seen in the paper.

## Smoke-free test
To ensure the successful installation and correct execution, please run the image under the short execution setting and confirm that no errors where thrown.