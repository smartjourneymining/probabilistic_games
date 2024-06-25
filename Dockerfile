# syntax=docker/dockerfile:1.4
FROM --platform=linux/amd64  debian:bookworm

WORKDIR /home
COPY requirements.txt .project-root ./
COPY data data/

RUN <<EOF
    apt-get update
    apt-get install -y curl
    apt-get install -y python3 python3-pip
    # to compile PRISM Binary
    apt-get install -y gcc make
    # PRISM needs some Java >= 8
    apt-get install -y openjdk-17-jdk-headless
    # directories for PRISM queries
    mkdir -p /home/queries
    # directories used in tool and output
    mkdir -p /home/generated
    mkdir -p /home/out
    # install PRISM
    curl https://www.prismmodelchecker.org/dl/prism-games-3.2.1-src.tar.gz | tar zxvf -
    (cd prism-games-3.2.1-src/prism && make)
    apt-get install -y graphviz graphviz-dev
    pip3 install pygraphviz --break-system-packages
    # install required python libraries
    pip3 install --no-cache-dir -r requirements.txt --break-system-packages
    # download case studies data
    curl "https://zenodo.org/records/6962413/files/data.csv?download=1" -o data/data.csv
    curl "https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884" -o "data/BPI Challenge 2017.xes"
EOF

COPY probabilistic_game_utils.py \
    io_alergia_greps.py \
    io_alergia_bpic.py \
    run.py \
    ./

# copy project subdirectories
COPY queries /home/queries
COPY journepy /home/journepy

ENTRYPOINT ["python3", "-u", "./run.py"]
