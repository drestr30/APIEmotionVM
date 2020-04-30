#!/bin/bash
# Author: Justice Amoh
# Description: Bootstrap script for Vagrant Machine. The VM image is configured
# for the course ENGS108, Applied Machine Learning, Dartmouth College
# Adapted from: bootstrap.sh by Holberton School

# Anaconda
apt-get update -q
su - vagrant

echo download Anaconda
miniconda=Miniconda3-latest-Linux-x86_64.sh
cd /vagrant
if [[ ! -f $miniconda ]]; then
    wget --quiet http://repo.continuum.io/miniconda/$miniconda
fi

echo installing Anaconda
chmod +x $miniconda
./$miniconda -b -p /home/vagrant/anaconda

echo 'export PATH="/home/vagrant/anaconda/bin:$PATH"' >> /home/vagrant/.bashrc
source /home/vagrant/.bashrc

/home/vagrant/anaconda/bin/conda install python=3.7.0

# Tensorflow
/home/vagrant/anaconda/bin/pip install tensorflow==1.15.0
 
# Keras
/home/vagrant/anaconda/bin/pip install keras
/home/vagrant/anaconda/bin/pip install opencv-python-headless

# Conda Installations
/home/vagrant/anaconda/bin/pip install Flask

echo 'All set!'



