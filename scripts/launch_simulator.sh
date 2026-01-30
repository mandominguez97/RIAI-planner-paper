#!/bin/bash
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

trap handle_exit EXIT
function handle_exit() {
    echo "Exiting simulation."
    echo "${PIDS[@]}"
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid"
        fi
    done 
    pkill -9 MicroXRCEAgent
    exit 0
}

VERBOSE=false
run_cmd() {
    local cmd="$1" 
    if $VERBOSE; then
        echo "Executing: $cmd"
        (eval "$cmd" &)
        PIDS+=("$!")
    else
        (eval "$cmd > /dev/null 2>&1 &")
        PIDS+=("$!") 
    fi
}

echo "Setting up the simulation environment..."
num_vehicles=${1:-1}
PX4_FOLDER="$(pwd)/../PX4-Autopilot"

PIDS=()
export PX4_GZ_WORLD="riai_planner_paper_world"
export GZ_SIM_RESOURCE_PATH="$(pwd)/../gz_assets/models/:$(pwd)/../gz_assets/worlds/"
echo "starting gz server..."
run_cmd "gz sim -r $PX4_GZ_WORLD.sdf"
sleep 10

y_0="-81"
for((vehicle=1; vehicle<=num_vehicles; vehicle++)); do

    export PX4_SIM_MODEL="x500_vision"
    export PX4_SYS_AUTOSTART=4005

    camera_topic="/world/${PX4_GZ_WORLD}/model/${PX4_SIM_MODEL}_${vehicle}/link/mono_cam/base_link/sensor/camera/image"
    camera_topics="$camera_topics $camera_topic"

    y_n=$((y_0 - (vehicle-1) * 4))
    export PX4_UXRCE_DDS_NS="px4_${vehicle}"
    export PX4_GZ_MODEL_POSE="90,${y_n},3,0,0,0"
    
    export PX4_PARAM_EKF2_EV_CTRL=11
    export PX4_PARAM_EKF2_HGT_REF=3
    export PX4_PARAM_EKF2_GPS_CTRL=0
    export PX4_PARAM_EKF2_BARO_CTRL=0
    export PX4_PARAM_EKF2_RNG_CTRL=0

    run_cmd "${PX4_FOLDER}/build/px4_sitl_default/bin/px4 -i $vehicle"
    sleep 5
done

run_cmd "ros2 run ros_gz_image image_bridge $camera_topics"
sleep 5

run_cmd "MicroXRCEAgent udp4 -p 8888"
sleep 8

echo "Simulation started."
while true; do
    sleep 5
done

