FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime

RUN git clone https://github.com/Ghadjeres/DeepBach.git
WORKDIR DeepBach
RUN conda env create --name deepbach_pytorch -f environment.yml

RUN apt update && apt install wget
RUN bash dl_dataset_and_models.sh


COPY entrypoint.sh entrypoint.sh
RUN chmod u+x entrypoint.sh

EXPOSE 5000
ENTRYPOINT ["./entrypoint.sh"]
