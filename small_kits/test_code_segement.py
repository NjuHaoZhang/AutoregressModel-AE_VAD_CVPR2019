def test_np_roll():
    import numpy as np
    # example1
    a = np.arange(10)
    print(a,"\n")
    for i in range(5):
        a = np.roll(a, 2) # +2, 向右移动2
        print(a)
    print("\n\n")
    #
    # example1
    a = np.arange(10)
    print(a, "\n")
    for i in range(5):
        a = np.roll(a, -1)  # -1，向左移动1
        print(a)
    print("\n\n")
    #
    # example 2
    b = np.reshape(a, (2, 5))
    print(b, "\n")
    for j in range(5):
        b = np.roll(b, shift=1, axis=1) # 对每一行做 shift=1 的roll
        print(b)
    print("\n\n")
    #
    # example 3
    b = np.reshape(a, (2, 5))
    print(b, "\n")
    for j in range(5):
        b = np.roll(b, shift=1, axis=0) # 对每一列 做 shift=1 的 roll，但是默认正方向是？
        print(b)
    print("\n\n")
    #
    # example 3
    c = np.arange(25)
    c = np.reshape(c, (5, 5))
    print("c: ", c)
    for k in range(5):
        # 先横着翻，再竖着翻
        c1 = np.roll(c, shift=1, axis=0) # 默认向下为正方向
        print("roll_c1: ", c1)
        c2 = np.roll(c1, shift=1, axis=1) # 默认向右为正方向
        print("roll_c2: ", c2)
        print("\n")


if __name__ == '__main__':
    test_np_roll()