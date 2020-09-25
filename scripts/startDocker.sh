#!/bin/bash

DOCKER_IMAGE=nvcr.io/nvidia/clara-train-sdk:v3.0

DOCKER_Run_Name=brats_seg

jnotebookPort=19010
GPU_IDs='7'
AIAA_PORT=19011

#################################### check if parameters are empty

#################################### check if name is used then exit
docker ps -a|grep ${DOCKER_Run_Name}
dockerNameExist=$?
if ((${dockerNameExist}==0)) ;then
  echo --- dockerName ${DOCKER_Run_Name} already exist
  echo ----------- attaching into the docker
  docker exec -it ${DOCKER_Run_Name} /bin/bash
  exit
fi

echo -----------------------------------
echo starting docker for ${DOCKER_IMAGE} using GPUS ${GPU_IDs}
echo -----------------------------------

extraFlag="-it "
cmd2run="/bin/bash"

extraFlag=${extraFlag}" -p "${jnotebookPort}":19010 -p "${AIAA_PORT}":19011"
echo starting please run "./installDashBoardInDocker.sh" to install the lab extensions then start the jupeter lab
echo once completed use web browser with token given yourip:${jnotebookPort} to access it

DATA_PATH=/home/liuyuan/shu_codes/datasets/brats
docker run --rm ${extraFlag} \
  --name=${DOCKER_Run_Name} \
  -e NVIDIA_VISIBLE_DEVICES=${GPU_IDs} \
  -v ${PWD}/../:/workspace/ \
  -v ${DATA_PATH}:/workspace/data \
  -w /workspace/scripts \
  --runtime=nvidia \
  --shm-size=20g --ulimit memlock=-1 --ulimit stack=67108864 \
  ${DOCKER_IMAGE} \
  ${cmd2run}

