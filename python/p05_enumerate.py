### <<31>>

list = ['a', 'b', 'c', 'd', 5]
print(list)

for i in list:
    print(i)
print("==================================================")

# enumerate는 ‘열거하다’라는 뜻이다. 이 함수는 순서가 있는 데이터(리스트, 튜플, 문자열)를 입력으로 받아 인덱스 값을 포함하는 enumerate 객체를 리턴한다.
for index, value in enumerate(list):
    print(index, value)