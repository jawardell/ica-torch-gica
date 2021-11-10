FROM ubuntu:16.04
ENV MCRROOT=/usr/local/MATLAB/MATLAB_Runtime/v91
ENV MCR_CACHE_ROOT=/tmp
#RUN printf "deb http://archive.debian.org/debian/ jessie main\ndeb-src http://archive.debian.org/debian/ jessie main\ndeb http://security.debian.org jessie/updates main\ndeb-src http://security.debian.org jessie/updates main" > /etc/apt/sources.list

RUN apt-get update && apt-get install -y software-properties-common
#RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y \
    zip unzip wget \
    libjasper-runtime libx11-dev libxcomposite-dev \
    libxcursor-dev libxdamage-dev libxext-dev \
    libxfixes-dev libxft-dev libxi-dev \
    libxrandr-dev libxt-dev libxtst-dev \
    libxxf86vm-dev libasound2-dev libatk1.0-dev \
    libcairo2-dev gconf2 \
    libsndfile1-dev libxcb1-dev libxslt-dev \
    curl \
    libgtk-3-dev 

#RUN cd /usr/local && wget http://ftp.mozilla.org/pub/firefox/releases/69.0/linux-x86_64/en-US/firefox-69.0.tar.bz2 && tar xvjf firefox-69.0.tar.bz2 && ln -s /usr/local/firefox/firefox /usr/bin/firefox
#RUN export BROWSER=/usr/bin/firefox
#RUN /usr/bin/firefox -headless --setDefaultBrowser &
RUN mkdir /tmp/mcr_installer && \
    cd /tmp/mcr_installer && \
    wget http://ssd.mathworks.com/supportfiles/downloads/R2016b/deployment_files/R2016b/installers/glnxa64/MCR_R2016b_glnxa64_installer.zip && \
    unzip MCR_R2016b_glnxa64_installer.zip && \
    ./install -mode silent -agreeToLicense yes && \
    rm -Rf /tmp/mcr_installer

# FROM coinstac/coinstac-base-python-stream

# Set the working directory
#COPY groupicatv4.0b /computation/groupicatv4.0b
#RUN adduser --disabled-password --gecos '' docker
#RUN adduser docker sudo
#RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
#USER docker
#RUN cd /computation/groupicatv4.0b/GroupICATv4.0b_standalone_aug_8_2019 && unzip MCRInstaller.zip -d /tmp/MCRInstaller &&  cd /tmp/MCRInstaller  && sudo ./install -mode silent -agreeToLicense yes
#RUN bash download_mcr.sh && unzip MCRInstaller.zip -d /tmp/MCRInstaller && rm MCRInstaller.zip &&  cd /tmp/MCRInstaller  && sudo ./install -mode silent -agreeToLicense yes 

# Copy the current directory contents into the container
WORKDIR /app
COPY requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt
RUN pip install awscli s3utils
COPY ./groupicatv4.0b/icatb/nipype-0.10.0/nipype/interfaces/gift /usr/local/lib/python3.6/site-packages/nipype/interfaces/gift
RUN chmod -R a+wrx /app
#RUN chmod -R a+wrx /usr/local/MATLAB/MATLAB_Runtime/v91


COPY . /app
#RUN (timeout 300 bash /app/groupicatv4.0b/GroupICATv4.0b_standalone/run_groupica.sh /usr/local/MATLAB/MATLAB_Runtime/v91; exit 0)




