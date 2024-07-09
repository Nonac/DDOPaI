FROM tensorflow/tensorflow:2.7.0rc0-gpu

RUN sed -i 's/^/#/' /etc/apt/sources.list.d/cuda.list && \
    apt-get update && \
    apt-get install -y openssh-server && \
    mkdir /var/run/sshd

RUN echo 'root:123456' | chpasswd

RUN echo "PasswordAuthentication yes" >> /etc/ssh/sshd_config
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

RUN pip install tcav
RUN pip install scikit-image

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
