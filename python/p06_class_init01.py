### <<46>>

class Father:
    # __init__ : 생성자메소드(호출시 자동실행)
    def __init__(self, name):   # 클래스의 매개변수는 __init__에 들어감(name)
        self.name1 = name
        print("Father __init__ 실행됨")
        print(self.name1, "아빠")
    babo = 4
        
aaa = Father('재현')
# Father __init__ 실행됨
# 재현 아빠

class Son(Father):      # Father 클래스 상속받음.
    def __init__(self, name):
        print("Son __init__ 시작")
        self.name2 = name
        print(self.name2, "아들")
        super().__init__(name)  # 부모클래스(Father)의 init에 name을 건냄.
        print("Son __init__ 끝")
        
bbb = Son('흥민')
# Son __init__ 시작
# 흥민 아들
# Father __init__ 실행됨
# 흥민 아빠
# Son __init__ 끝