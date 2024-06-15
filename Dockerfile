FROM debian:11

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

# install required python libraries
COPY requirements.txt /home/
RUN pip3 install --no-cache-dir -r requirements.txt

# copy binary of PRISM
COPY prism-games-3.2.1-src.tar.gz /home/

# install PRISM
RUN tar xfz /home/prism-games-3.2.1-src.tar.gz && cd /home/prism-games-3.2.1-src/prism && make
# PRISM PATH: "/home/prism-games-3.2.1-src/prism/bin/prism"
RUN cd /home

# copy files for simple PRISM test
COPY queries /home/queries
COPY journepy /home/journepy

# TODO: CHANGE script name!

# copy python scripts that execute the experiments
COPY scriptname.py /home/

CMD ["python3", "./scriptname.py"]