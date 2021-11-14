FROM python:3.8-slim

RUN apt-get update && apt install -y gdal-bin openssl wget && \
    cd /usr/local && \
    wget -O minizinc-2.5.5-linux64.tar.gz https://github.com/MiniZinc/MiniZincIDE/releases/download/2.5.5/MiniZincIDE-2.5.5-bundle-linux-x86_64.tgz && \
    tar xfz minizinc-2.5.5-linux64.tar.gz && \
    cp MiniZincIDE-2.5.5-bundle-linux-x86_64/bin/* bin/ && \
    cp -ra MiniZincIDE-2.5.5-bundle-linux-x86_64/share/* share/ && \
    rm -rf minizinc-2.5.5-linux64.tar.gz

WORKDIR /opt/algorithm/

COPY . .

RUN pip install -r requirements.txt

EXPOSE 5000

ENV FLASK_APP=main.py 

ENTRYPOINT flask run --host=0.0.0.0

#For debug
#RUN apt-get install -y iputils-ping
#CMD ping 8.8.8.8