#-*- coding:utf-8 -*-

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np                #  뭔가 많이 import 시켰다. 그냥 그른가부다 하고 넘어가자.
import matplotlib.pyplot as plt   #  여러분이 코딩을 할땐 그냥 복붙하여 사용하면 된다.
# import 어쩌고 as 저쩌고 라는 코드는 from 어쩌고 import 와 비슷하다. 다만 코드를 칠때마다 "저쩌고.어쩌고 안의 기능"이라고 해야 코드가 작동한다.
# 바로 아래코드도 보면 mpl.rcParams[~~~] 이라고 써있다. rcParams 를 바로 쓰지 않는다.
mpl.rcParams['legend.fontsize'] = 10			# 그냥 오른쪽 위에 뜨는 글자크기 설정이다.

fig = plt.figure()								# 이건 꼭 입력해야한다.
ax = fig.gca(projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)	# 각도의 범위는 -4파이 에서 +4파이
z = np.linspace(-2, 2, 100)						# z는 -2부터 2까지 올라간다.
r = z**2 + 1									# z값이 변함에 따라 반지름이 바뀔 것이다. 
x = r * np.sin(theta)							# 나선구조를 만들기 위해 x는 sin함수
y = r * np.cos(theta)							# 나선구조를 만들기 위해 y는 cos함수
ax.plot(x, y, z, label='parametric curve')		# 위에서 정의한 x,y,z 가지고 그래프그린거다.
ax.legend()										# 오른쪽 위에 나오는 글자 코드다. 이거 없애면 글자 사라진다. 없애도 좋다.

plt.show()