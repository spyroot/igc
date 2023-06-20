# Get-NetAdapterLso -Name * -IncludeHidden
# Enable-NetAdapterLso -Name "*" -IPv4 -IPv6 * -IncludeHidden
# Get-NetAdapterChecksumOffload  -Name "*" -IncludeHidden

FROM  nvidia/cuda:11.7.0-base-ubuntu22.04
RUN apt-get update
RUN apt-get -y upgrade
RUN mkdir -p /run/systemd
RUN apt-get install -y tzdata
RUN apt-get install -y build-essential devscripts debhelper fakeroot
RUN apt-get install -y gcc meson git wget make curl ninja-build python3-pip unzip zip gzip curl
RUN apt-get install -y vim openssh-server zsh fzf
RUN apt-get install -y cuda-toolkit-11-8 cuda

#RUN pip install python3-wheel python3-libnvinfer
EXPOSE 22/tcp
#EXPOSE 54321/tcp
ENV JUPYTER_PORT=8888
ENV TENSORBOARD_PORT=6006

ENV set CONDA_OVERRIDE_CUDA=11.7

#pip install torchopt --extra-index-url https://download.pytorch.org/whl/cu117
# conda env create --file conda-recipe-minimal.yaml

EXPOSE 8888
EXPOSE 6006
EXPOSE 6064

RUN apt-get install -y ffmpeg
RUN apt-get install -y tzdata ffmpeg libsndfile1 libtiff-dev libpng-dev meson ninja-build cmake yamllint
#RUN apt-get install -y nvidia-cuda-toolkit cuda
#RUN apt-get install -y cuda
RUN apt-get -y install libgl1-mesa-dev libgl1-mesa-glx
RUN apt-get install -y fonts-powerline zsh vim

RUN mkdir /var/run/sshd
RUN echo 'root:root' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd


RUN wget \
    https://github.com/deepmind/mujoco/releases/download/2.3.0/mujoco-2.3.0-linux-x86_64.tar.gz \
    && mkdir /root/.mujoco \
    && tar -xvf mujoco-2.3.0-linux-x86_64.tar.gz -C /root/.mujoco \
    && rm -f mujoco-2.3.0-linux-x86_64.tar.gz


RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN pip install torch torchvision torchaudio
RUN pip install mujoco
RUN pip install torchopt --extra-index-url https://download.pytorch.org/whl/cu117

RUN export PATH="/root/miniconda3/bin:$PATH"
ENV PATH="/root/miniconda3/bin:$PATH"
RUN conda update -n base -c defaults conda

RUN apt-get install zsh-autosuggestions -y
RUN apt-get install bash-completion -y

# WSL2 specific
RUN rm -rf \
    /usr/lib/x86_64-linux-gnu/libcuda.so* \
    /usr/lib/x86_64-linux-gnu/libnvcuvid.so* \
    /usr/lib/x86_64-linux-gnu/libnvidia-*.so* \
    /usr/lib/firmware \
    /usr/local/cuda/compat/lib 2> /dev/null

RUN apt-get -y install locales
RUN git clone https://github.com/bhilburn/powerlevel9k.git ~/powerlevel9k

RUN echo 'source  /root/powerlevel9k/powerlevel9k.zsh-theme' >> ~/.zshrc
RUN echo 'export TERM=xterm-256color' >> /root/.bashrc
RUN echo 'export TERM=xterm-256color' >> /root/.zshrc

RUN wget  \
    https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh \
       &&  sh install.sh

RUN echo 'ZSH_THEME="powerlevel9k/powerlevel9k'
RUN echo 'plugins=(\n\
  git\n\
  bundler\n\
  dotenv\n\
  macos\n\
  rake\n\
  rbenv\n\
  ruby\
)' >> /root/.zshrc

# x11
RUN apt xauth \
    && mkdir /var/run/sshd \
    && mkdir /root/.ssh \
    && chmod 700 /root/.ssh \
    && ssh-keygen -A \
    && sed -i "s/^.*X11Forwarding.*$/X11Forwarding yes/" /etc/ssh/sshd_config \
    && sed -i "s/^.*X11UseLocalhost.*$/X11UseLocalhost no/" /etc/ssh/sshd_config \
    && grep "^X11UseLocalhost" /etc/ssh/sshd_config || echo "X11UseLocalhost no" >> /etc/ssh/sshd_config

RUN apt-get install -y libglew-dev

# jax
RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

#
#RUN wget https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.5.1/local_repos/nv-tensorrt-local-repo-ubuntu2204-8.5.1-cuda-11.8_1.0-1_amd64.deb \
#    && dpkg -i nv-tensorrt-local-repo-ubuntu2204-8.5.1-cuda-11.8_1.0-1_amd64.deb \
#    $$ cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/ \
#RUN apt-get install -y python3-libnvinfer-dev

# ENTRYPOINT ["sh", "-c", "/usr/sbin/sshd && tail -f /dev/null"]

# run recipe
RUN conda init
RUN apt-get install -y speedtest-cli

RUN echo '\n\
name: meta_critic\n\
channels:\n\
  - pytorch\n\
  - nvidia/label/cuda-11.7.1\n\
  - defaults\n\
  - conda-forge\n\
dependencies:\n\
  - python = 3.10\n\
  - pip\n\
  - pytorch::pytorch >= 1.13\n\
  - pytorch::torchvision\n\
  - pytorch::pytorch-mutex = *=*cuda*\n\
  - pip:\n\
      - torchviz\n\
  - nvidia/label/cuda-11.7.1::cuda-toolkit = 11.7\n\
  - cmake >= 3.11\n\
  - make\n\
  - cxx-compiler\n\
  - gxx = 10\n\
  - nvidia/label/cuda-11.7.1::cuda-nvcc\n\
  - nvidia/label/cuda-11.7.1::cuda-cudart-dev\n\
  - pybind11 >= 2.10.1\n\
  - optree >= 0.4.1\n\
  - typing-extensions >= 4.0.0\n\
  - numpy\n\
  - python-graphviz\n\
  - mujoco\n\
  - matplotlib\n\
  - pyyaml\n\
  - tensorboard\n\
  - tqdm\
' >> /root/conda-recipe.yaml
RUN conda env create --file /root/conda-recipe.yaml
RUN apt-get install swig -y
RUN /root/miniconda3/bin/conda activate meta_critic
RUN pip install gym[all] -U
RUN pip uninstall mujoco-py
RUN pip install mujoco -U
#CMD ["/usr/sbin/sshd", "-D"]
CMD ["/bin/zsh"]

