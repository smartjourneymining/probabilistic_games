# Stochastic Games for User Journeys (Artifact)
This is the artifact for the FM2024 publication "Stochastic Games for User Journeys" by Kobialka, Pferscher, Bergersen, Johnsen, and Tapia Tarifa. We provide the artifact as a Docker container that includes all required data, tools and source code. 

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
