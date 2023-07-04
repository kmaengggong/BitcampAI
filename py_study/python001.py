# 1
print("matkit \"code\" lab")
print("she's gone")

# 2
a = 10
b = 20
print("a의 값은", a)
print("b의 값은", b)
print("a와 b의 합은", a+b)

# 3
a = 10
b = 'matkit '
print(a * 3)
print(b * 3)

# 4
a = ['메이킷', '우진', '시은']
print(a)
for i in range(0, len(a)):
    print(a[i])

# 5
a.insert(2, '제임스')
print(a[:2])
print(a[1:])
print(a[2:])
print(a)

# 6
a = ['우진', '시은']
b = ['메이킷', '소피아', '하워드']
a.extend(b)
print(a)
print(b)

# 7
a = a[:2]
b.extend(a)
print(a)
print(b)

# 8
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
print("Original :\n", a)
a_transpose = np.transpose(a)
print("Transpose :\n", a_transpose)

# 9
a_reshape = np.reshape(a, (3,2))
print("Reshape :\n", a_reshape)