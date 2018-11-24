FROM python:3.6-slim-jessie as khepera-engine

# install build tools and git
RUN apt-get update -yqq \
    && apt-get install -yqq --no-install-recommends \
    	git \
    	g++ \
    	make

WORKDIR /tmp/

# build Khepera engine
RUN git clone git://github.com/Ewande/khepera.git --depth 1 -q
RUN cd khepera && make

# generate kheppy version based on git
COPY . .
RUN python setup.py sdist

FROM python:3.6-slim-jessie

WORKDIR /opt/

COPY --from=khepera-engine /tmp/khepera/SimulationServer.so .

COPY . .

COPY --from=khepera-engine /tmp/kheppy.egg-info/PKG-INFO .

RUN pip install .

ENV KHEPERA_LIB /opt/SimulationServer.so
