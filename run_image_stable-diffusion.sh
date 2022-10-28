#!/bin/bash
SCRIPT_DIR=$(cd $(dirname $0); pwd)
DOCKSHARE=${SCRIPT_DIR}

DOCKIMG="stable-diffusion:latest"
INITDIR="/home/dockshare"

USER=$(id -u)
GROUP=$(id -g)

docker run -it --runtime=nvidia \
            -w ${INITDIR} --rm --dns 137.153.66.28 \
            -e DISPLAY=$DISPLAY    \
            -v /tmp/.X11-unix:/tmp/.X11-unix \
            -v /var/lib/dbus:/var/lib/dbus \
            -v /var/run/dbus:/var/run/dbus \
            --device=/dev/dri:/dev/dri \
            -v /etc/localtime:/etc/localtime:ro \
            -v ${DOCKSHARE}:${INITDIR} ${DOCKIMG} bash

