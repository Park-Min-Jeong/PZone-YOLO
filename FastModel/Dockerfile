FROM python:3.9.0

WORKDIR /home/

RUN echo "testing"

RUN git clone https://github.com/Park-Min-Jeong/YOLO-Test.git 

WORKDIR /home/YOLO-Test/FastModel/

RUN python -m pip install --upgrade pip

RUN pip install -r requirements.txt

# RUN pip install gunicorn

EXPOSE 8000

CMD ["bash","-c","uvicorn main:app --reload --host=0.0.0.0 --port=8000"]