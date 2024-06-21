FROM --platform=linux/amd64  debian:bookworm

RUN apt-get update && apt-get install -y python3 python3-pip

# to compile PRISM Binary
RUN apt-get install -y gcc make

# PRISM requires JAVA
ENV JAVA_HOME=/opt/java/openjdk
COPY --from=eclipse-temurin:21 $JAVA_HOME $JAVA_HOME
ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /home

# directories that contains all generated automata, plots and figures
RUN mkdir /home/results

# directories for PRISM queries
RUN mkdir /home/queries

# directories used in tool and output
RUN mkdir /home/generated
RUN mkdir /home/out
RUN mkdir /home/out/greps
RUN mkdir /home/out/bpic

# copy binary of PRISM
COPY prism-games-3.2.1-src.tar.gz /home/

# install PRISM
RUN tar xfz /home/prism-games-3.2.1-src.tar.gz && cd /home/prism-games-3.2.1-src/prism && make && cd /home
RUN chmod +x /home/prism-games-3.2.1-src/prism

# install required python libraries
COPY requirements.txt /home/
RUN pip3 install --no-cache-dir -r requirements.txt --break-system-packages
RUN apt-get install -y graphviz graphviz-dev
RUN pip3 install pygraphviz --break-system-packages

# copy project files
COPY queries /home/queries
COPY journepy /home/journepy
COPY data /home/data
COPY .project-root /home/.project-root
COPY probabilistic_game_utils.py /home/probabilistic_game_utils.py

# copy python scripts that execute the experiments
COPY io_alergia_greps.py /home/
# COPY io_alergia_bpic.py /home/
COPY run.py /home/



CMD ["python3", "./run.py"]
