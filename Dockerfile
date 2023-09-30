FROM ubuntu:latest


RUN apt update && apt install -y python3 python3-pip libegl1-mesa libgl1-mesa-dri libxcb-xfixes0-dev \
    libglib2.0-0 mesa-vulkan-drivers libgl1-mesa-glx libxkbcommon0 libdbus-1-3

# manually install some dependencies to speed up
RUN pip3 install numpy pynbody matplotlib pillow wgpu jupyter_rfb tqdm opencv-python PySide6

COPY src /app/src
COPY tests /app/tests
COPY pyproject.toml /app/
COPY README.md /app/

WORKDIR /app


RUN pip3 install .[test]

ENTRYPOINT ["/bin/bash"]