from fastapi import FastAPI

app = FastAPI()

@app.get("/")       # @ : 데코레이터(app 인스턴스에 "/"경로로 get요청이 들어오면 아래함수(read_root)를 실행시키겠다.)
def read_root():
    return{"message" : "Hello, FastAPI"}

@app.get("/item/{item_id}")         # {item_id} : 변수
def read_item(item_id):             # read_item 함수가 변수를 받아서
    return {"item_id" : item_id}    # 딕셔너리 형태로 반환해준다.

@app.get("/items/")
def read_items(skip=0, limit=10):   # 디폴트 변수
    return {'skip' : skip, "limit" : limit}
# http://localhost:8000/items/?skip=5&limit=15 -> {"skip":"5","limit":"15"}