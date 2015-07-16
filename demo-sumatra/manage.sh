#!/bin/bash

if [ $1 == "--build" ]; then
	manage_command="build"
elif [ $1 == "--run-core" ]; then
	manage_command="run-core"
elif [ $1 == "--run-web" ]; then
	manage_command="run-web"
elif [ $1 == "--stop" ]; then
	manage_command="stop"
else
	echo "Enter command (build|run|stop): "
	read manage_command
fi

if [ $2 == "--simulation" ]; then
	if [ $# == 3 ]; then
		simulation_name=$3
	else
		echo "Enter simulation's name: "
		read simulation_name
	fi
else
	echo "Enter simulation's name: "
	read simulation_name
fi

# DDSM Api token
token=`cat .ddsm_token`
echo "The DDSM API token is: $token."
echo "It is not used yet but will be very soon! :-)"

if [ $manage_command == "build" ]; then
	echo "Building the container $simulation_name..."
	rm -rf *docker_image.tar
	docker build -t $simulation_name .
	image_name=`date +"%Y%m%d%H%M%S"`
	docker save $simulation_name > "$image_name"-docker_image.tar
	echo "Container $simulation_name built."
elif [ $manage_command == "run-core" ]; then
	echo "Running the container $simulation_name..."
	echo "Enter the executable command: "
	read execute_cmd
	# Warning:: Path to data should not have spaces. Be careful with muli-words directories names.
	data_path=$(dirname $(readlink -f "../demo-data"))
	echo "$data_path"
	#--executable=python --main=main.py default.param
	#Add the input data volume or files here to provide them to 
	# Add API token to smt in a certain way. We will have to smt configure. It is better to leave it outside here. Or the same key will be used
	# which does not help for sharing. It should be in a form of an http store link. ddsm://domain:token.
	if [ -f *docker_image.tar ]; then
	   image_name=`ls | grep docker_image | awk -F '[: ]+' '{print $1}'`
	   echo $image_name
	   echo "New container image detected."
	   docker run -i -t -v "$data_path"/demo-data/default.param:/src/default.param $simulation_name /bin/bash -c "cd /src; smt run $execute_cmd"
	   rm -rf *docker_image.tar
	else
	   docker run -i -t -v "$data_path"/demo-data/default.param:/src/default.param $simulation_name /bin/bash -c "cd /src; smt run $execute_cmd"
	fi	
	echo "Container $simulation_name killed."
elif [ $manage_command == "run-web" ]; then
	echo "Running the container $simulation_name..."
	echo "Enter the executable command: "
	read execute_cmd
	# Warning:: Path to data should not have spaces. Be careful with muli-words directories names.
	data_path=$(dirname $(readlink -f "../demo-data"))
	echo "$data_path"
	#--executable=python --main=main.py default.param
	#Add the input data volume or files here to provide them to 
	# Add API token to smt in a certain way. We will have to smt configure. It is better to leave it outside here. Or the same key will be used
	# which does not help for sharing. It should be in a form of an http store link. ddsm://domain:token.
	if [ -f *docker_image.tar ]; then
	   echo "New container image detected."
	   docker run -i -t -p 127.0.0.1:5000:5000 -v "$data_path"/demo-data/default.param:/src/default.param $simulation_name /bin/bash -c "cd /src; smt run $execute_cmd; smtweb -p 5000 --allips"
	   rm -rf *docker_image.tar
	else
	   docker run -i -t -p 127.0.0.1:5000:5000 -v "$data_path"/demo-data/default.param:/src/default.param $simulation_name /bin/bash -c "cd /src; smt run $execute_cmd; smtweb -p 5000 --allips"
	fi	
	echo "Container $simulation_name killed."
elif [ $manage_command == "stop" ]; then
	echo "Stopping the container $simulation_name..."
	docker stop $(docker ps | grep python-simul | awk -F '[: ]+' '{print $1}')
    # docker rm $(docker ps | grep python-simul | awk -F '[: ]+' '{print $1}')
    echo "Container $simulation_name stoped."
else
	echo -e "\033[31m Usage: You can only do build, run-core, run-web or stop."
	echo -e "\033[0m "
fi

