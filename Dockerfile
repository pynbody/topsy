FROM ubuntu:24.04


RUN apt update && apt install -y python3 python3-pip libgl1-mesa-dri libxcb-xfixes0-dev \
    libglib2.0-0 mesa-vulkan-drivers libegl1 libgl1 libglx-mesa0 libxkbcommon0 libdbus-1-3

# manually install some dependencies to speed up
RUN pip3 install --break-system-packages numpy pynbody matplotlib pillow tqdm opencv-python PySide6 pytest pytest-ipywidgets
RUN playwright install
RUN playwright install-deps

COPY src /app/src
COPY tests /app/tests
COPY pyproject.toml /app/
COPY README.md /app/

WORKDIR /app

RUN pip3 install --break-system-packages .[test]

ENTRYPOINT ["/bin/bash"]