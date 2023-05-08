#探索木の作成（1度に言える数,終了する数)
def make_tree(m, n):
    tree = [] #探索木用の配列
    i = 0 #層の深さを数える
    while i <= n: #探索木の深さ分繰り返す
        lis = [] #1層分を格納する配列
        j = i #各層の左端は、層の深さiと一致
        while j <= n and j <= m * i: #終了する数nか、m*iに辿り着くまで層を作成
            if n - j != 0: #終了していない場合
                lis.append(0) #0を記録
            if n - j == 0: #終了している場合
                if i % 2 == 0: #後攻（偶数層）で終わった場合
                    lis.append(1) #先攻の勝利（1を記録）
                if i % 2 == 1: #先攻（奇数層）で終わった場合
                    lis.append(-1) #後攻の勝利(-1を記録）)
            j += 1
        i += 1
        tree.append(lis) #層を探索木に加える
    return tree #完成した探索木を返す

#minimax探索を行う(探索木, 1度に言える数)
def search(tree, m):
    n = len(tree) - 1 #探索木の深さを求める(根を0と数える)
    i = 0 #層の深さを数える
    while n >= i: #層の深さ分繰り返す
        for x in range(len(tree[n-i])): #深い方から探索していく
            if tree[n-i][x] == 0: #左からx番目に要素0が格納されていた場合
                lis = [] #子ノードを格納する配列
                j = 0 
                while j < m and j + x < i: #1度に言える数増加する、もしくは探索木の右端にたどり着くまで繰り返す
                    lis.append(tree[n-i+1][x+j]) #子ノードを配列に加える
                    j += 1
                if (n-i) % 2 == 0: #先攻(偶数層)の場合、1をとってくる
                    tree[n-i][x] = max(lis)
                if (n-i) % 2 == 1: #後攻(奇数層)の場合、-1をとってくる
                    tree[n-i][x] = min(lis)
        i += 1
    return tree[0][0] #勝者を返す(先攻なら1、後攻なら-1)


        
if __name__ == "__main__":
    m = 4 #1度に言える数mを指定
    for n in range(22):
        if 6 <= n and n <= 21:
            tree = make_tree(m,n)
            winer = search(tree, m)

            print("N =", n)
            for lis in tree:
                print(lis)
            if winer == 1:
                print("先攻の勝ち\n")
            elif winer == -1:
                print("後攻の勝ち\n")