### <<22>>

# 반복 구문을 적용할 수 있는 리스트와 같은 객체를 반복 가능(iterable) 객체라 한다.

# nums = iter([1,2,3])
lst = [1,2,3]
nums = iter(lst)
# print(nums.next())  # 파이썬 2.0 문법. 파이썬 iter객체에서는 쓸수없다.
print(next(nums))   # 1
print(next(nums))   # 2
print(next(nums))   # 3
print(next(nums))   # StopIteration