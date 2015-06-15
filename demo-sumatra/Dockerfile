FROM ubuntu:14.04

##############
# Setup Area #
##############


# Update packages
RUN apt-get update --fix-missing -y 

# Install Python Setuptools
RUN apt-get install -y python-setuptools

# Install pip
RUN easy_install pip

# Install git
RUN apt-get install -y git

# Install GitPython
RUN pip install GitPython==0.3.6 Django==1.6 django-tagging httplib2 jinja2 docutils parameters hgapi pyyaml

# Install sumatra
RUN pip install sumatra

# Including files.
COPY . /src

# Cleaning up
RUN cd /src; rm -rf .git/; rm -rf .smt/

# Setup Sumatra
RUN git config --global user.email "scientist_email@domain.com"
RUN git config --global user.name "scientist_name"
RUN cd /src; git init
RUN cd /src; git add --all
RUN cd /src; git commit -m "init."
RUN cd /src; smt init demo-sumatra

# Expose Smtweb port
EXPOSE 5000

###################
# Researcher Area #
###################

# Install core dependencies
RUN echo "Y" | apt-get install python-numpy

# Install python dependencies
# Note: Simulation dependencies have to be put in the requirements.txt file.
RUN cd /src; pip install -r requirements.txt

# Usage
# To Build: ./manage.sh --build --simulation demo-sumatra
# To Run without smtweb: ./manage.sh --run-core --simulation demo-sumatra
# To Run with smtweb./manage.sh --run-web --simulation demo-sumatra
# To force Stop: ./manage.sh --stop --simulation demo-sumatra