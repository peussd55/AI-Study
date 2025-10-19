from machine.car import drive
from machine.tv import watch

drive()
watch()
# 운전하다
# 시청하다

print("========================================================")

from machine import car
from machine import tv
from machine import car, tv

car.drive()
tv.watch()
# 운전하다
# 시청하다

print("===================== test =============================")

from machine.test.car import drive
from machine.test.tv import watch

drive()
watch()
# machine.test 운전하다
# machine.test 시청하다

from machine.test import car, tv

car.drive()
tv.watch()
# machine.test 운전하다
# machine.test 시청하다

from machine import test

test.car.drive()
test.tv.watch()
# machine.test 운전하다
# machine.test 시청하다