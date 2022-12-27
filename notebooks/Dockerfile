FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/notebooks/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /code/notebooks/requirements.txt


COPY ./Data_Generation.ipynb /code/notebooks/Data_Generation.ipynb

COPY ./Train_ArabicName_model.ipynb /code/notebooks/Train_ArabicName_model.ipynb

COPY ./_Datasets /code/Datasets

COPY ./Data_Generation.py /code/script/Data_Generation.py

COPY ./Train_ArabicName_model.py /code/script/Train_ArabicName_model.py



CMD ["jupyter-notebook","--notebook-dir=/code/notebooks/","--port=8888","--ip=0.0.0.0","--no-browser","--allow-root"]
