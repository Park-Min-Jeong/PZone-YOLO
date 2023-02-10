from tempfile import NamedTemporaryFile
from typing import IO

from yolo.yolo import Detector, Score
from fastapi import FastAPI,UploadFile,File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}


async def save_file(file: IO):
    # s3 업로드라고 생각해 봅시다. delete=True(기본값)이면
    # 현재 함수가 닫히고 파일도 지워집니다.
    with NamedTemporaryFile("wb", delete=False) as tempfile:
        tempfile.write(file.read())
        return tempfile.name


@app.post("/file/store")
async def store_file(file: UploadFile = File(...)):
    path = await save_file(file.file)

    return Score(path)


# # @app.post("/photo")
# # async def upload_photo(file: UploadFile):
# #     UPLOAD_DIR = "./photo"  # 이미지를 저장할 서버 경로
# #
# #     content = await file.read()
# #     filename = f"{str(uuid.uuid4())}.jpg"  # uuid로 유니크한 파일명으로 변경
# #     with open(os.path.join(UPLOAD_DIR, filename), "wb") as fp:
# #         fp.write(content)  # 서버 로컬 스토리지에 이미지 저장 (쓰기)
# #     lst=Detector(filename)
# #     length=lst[0]
# #     # Kinds=lst[1]
# #
# #     return {"length":length}
#
#
# # @app.post("/uploadfile/")
# # # async def create_upload_file(file: UploadFile):
# # async def create_file(file:UploadFile):
# #     # img = Image.open(file.file)
# #     # img.show()
# #     # return_lst=detect_image(file.file)
# #     class_id=return_lst[0]
# #     # length=return_lst[1]
# #     return detect_image(file.file)
#     # file = lambda x: base64.b64decode(x)(file)
#     # file= lambda x: Image.open(x)(file)
#
#     # return {"file": file.content_type}
# #
# # @app.post("/uploadfile/")
#
#
# # # This is a sample Python script.
# #
# # # Press Alt+Shift+X to execute it or replace it with your code.
# # # Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# #
# #
# # def print_hi(name):
# #     # Use a breakpoint in the code line below to debug your script.
# #     print(f'Hi, {name}')  # Press Ctrl+Shift+B to toggle the breakpoint.
# #
# #
# # # Press the green button in the gutter to run the script.
# # if __name__ == '__main__':
# #     print_hi('PyCharm')
# #
# # # See PyCharm help at https://www.jetbrains.com/help/pycharm/
