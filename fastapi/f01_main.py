from fastapi import FastAPI

app = FastAPI()

@app.get("/")   # @ : 데코레이터(app 인스턴스에 "/"경로로 get요청이 들어오면 아래함수(read_root)를 실행시키겠다.)
def read_root():
    return {"message" : "Hello, World"}

@app.get("/hello/")
def read_root():
    return {"message" : "Hello, Hello, Hello"}

@app.get("/hi/")
def read_root():
    return {"message" : "Hi, Hi, Hi"}

