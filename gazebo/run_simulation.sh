#!/bin/bash

echo "==============================================="
echo "             Iniciando Simulacion              "
echo "==============================================="

source /opt/ros/jazzy/setup.bash

# 1. Lanzar Gazebo
echo "[1/3] Abriendo Gazebo..."
(trap '' SIGINT; exec gz sim terrain_world.sdf) &
GAZEBO_PID=$!
sleep 10

# 2. Lanzar el puente
echo "[2/3] Conectando el puente de topicos..."
(trap '' SIGINT; exec ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=bridge.yaml) &
BRIDGE_PID=$!
sleep 5

# 3. Lanzar el inyector de perturbaciones
echo "[3/3] Inyectando perturbaciones aerodinamicas..."
(trap '' SIGINT; exec python3 flight_disturbances.py) &
PERT_PID=$!

echo "==============================================="
echo "            Simulacion en curso                "
echo "==============================================="

cleanup() {
    echo -e "\n\n[Cerrando] Desconectando componentes..."
    
    if kill -0 $PERT_PID 2>/dev/null; then kill -TERM $PERT_PID 2>/dev/null; fi
    if kill -0 $BRIDGE_PID 2>/dev/null; then kill -TERM $BRIDGE_PID 2>/dev/null; fi
    
    sleep 0.5
    
    if kill -0 $GAZEBO_PID 2>/dev/null; then kill -TERM $GAZEBO_PID 2>/dev/null; fi

    sleep 0.5
    echo "Simulacion terminada."
    exit 0
}

trap cleanup SIGINT
wait