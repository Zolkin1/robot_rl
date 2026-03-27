# syntax=docker/dockerfile:1


# ============================================
# Stage 1: Base / Simulation
# ============================================
FROM ubuntu:22.04 as sim
SHELL ["/bin/bash", "-c"]

# username, uid, gid
ARG USER=user
ARG UID=1000
ARG GID=1000
ARG OBELISK_ROOT=/
ENV USER=$USER
# ENV UID=1000
# ENV GID=1000
ENV UID=$UID
ENV GID=$GID
ENV OBELISK_ROOT=$OBELISK_ROOT

# set timezone
ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# basic dependencies from docker_setup.sh (up until sudo and below)
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    cmake \
    clang-tools-12 \
    nano \
    vim \
    git \
    libeigen3-dev \
    x11-apps \
    locales \
    iputils-ping \
    evtest \
    net-tools \
    sudo && \
    rm -rf /var/lib/apt/lists/* && \
    locale-gen en_US.UTF-8

# create non-root user with sudo privileges for certain commands
RUN groupadd --gid $GID $USER && \
    useradd --uid $UID --gid $GID -m $USER -d /home/${USER} --shell /usr/bin/bash && \
    echo "${USER}:password" | chpasswd && \
    usermod -aG sudo ${USER} && \
    echo "%${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# switch to new user and workdir
USER ${UID}


RUN echo "Pulling Obelisk..."
# clone Obelisk into this docker image at the $OBELISK_ROOT path
RUN git clone -b feb_hot_fix_scan https://github.com/Caltech-AMBER/obelisk.git /home/${USER}/obelisk
ENV OBELISK_ROOT=/home/${USER}/obelisk

# Install other dependencies here...

# you must run the setup script from a directory where the user has permissions
# run docker setup script in Dockerfile
WORKDIR /home/${USER}
RUN source /home/${USER}/obelisk/setup.sh --downstream-setup --mujoco --unitree
WORKDIR /home/${USER}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

# ============================================
# Stage 2: Hardware (adds lidar stack)
# ============================================
FROM sim as hardware
# Lidar odometry
# Install PCL and other dependencies for the lidar
RUN sudo apt-get update && sudo apt-get install -y \
    libpcl-dev \
    pcl-tools \
    ros-humble-pcl-ros \
    ros-humble-pcl-conversions

# Livox SDK
WORKDIR /home/${USER}
RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git && \
 cd Livox-SDK2 && \
 mkdir build && cd build && \
 cmake .. && \
 make -j && \
 sudo make install && \
 cd /home/${USER} && \
 mkdir -p livox_driver_ws/src && \
 cd livox_driver_ws/src && \
 git clone https://github.com/Livox-SDK/livox_ros_driver2.git && cd livox_ros_driver2

RUN cd /home/${USER}/livox_driver_ws/src/livox_ros_driver2 && \
    source /opt/ros/humble/setup.sh && ./build.sh humble && source ../../install/setup.sh && \
    cd /home/${USER} && mkdir -p lidar_odom_ws/src && cd lidar_odom_ws/src && \
    git clone -b ROS2 https://github.com/hku-mars/FAST_LIO.git --recursive && cd .. && \
    sudo rosdep init && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    colcon build --symlink-install && \
    echo "source /home/${USER}/lidar_odom_ws/install/setup.bash" >> /home/${USER}/.bashrc


# cat > config/MID360_config.json << 'EOF'
#{
#"lidar_summary_info" : {
# "lidar_type": 8
#},
#"MID360": {
# "lidar_net_info" : {
#   "cmd_data_port": 56100,
#   "push_msg_port": 56200,
#   "point_data_port": 56300,
#   "imu_data_port": 56400,
#   "log_data_port": 56500
# },
# "host_net_info" : {
#   "cmd_data_ip" : "192.168.123.222",
#   "cmd_data_port": 56101,
#   "push_msg_ip": "192.168.123.222",
#   "push_msg_port": 56201,
#   "point_data_ip": "192.168.123.222",
#   "point_data_port": 56301,
#   "imu_data_ip" : "192.168.123.222",
#   "imu_data_port": 56401,
#   "log_data_ip" : "",
#   "log_data_port": 56501
# }
#},
#"lidar_configs" : [
# {
#   "ip" : "192.168.123.120",
#   "pcl_data_type" : 1,
#   "pattern_mode" : 0,
#   "extrinsic_parameter" : {
#     "roll": 0.0,
#     "pitch": 0.0,
#     "yaw": 0.0,
#     "x": 0,
#     "y": 0,
#     "z": 0
#   }
# }
#]
#}
#EOF