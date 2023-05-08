import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image


#迷路1の作成(行数、列数)
def make_maze_1(n_rows, n_lines):
    maze = np.zeros((n_lines, n_rows)) #２次元配列を０で初期化
    maze[3][4] = 2 #ゴールを作成
    return maze #迷路を示す配列を返す

#迷路2の作成(行数、列数)
def make_maze_2(n_rows, n_lines):
    maze = np.zeros((n_lines, n_rows)) #配列を０で初期化
    #行列に壁を作成
    maze[1][1] = 1
    maze[1][2] = 1
    maze[1][3] = 1
    maze[1][4] = 1
    maze[3][2] = 1
    maze[4][2] = 1
    maze[4][4] = 1
    maze[4][5] = 1
    #行列にゴールを作成
    maze[3][4] = 2 #ゴール１
    maze[4][1] = 3 #ゴール２
    return maze #迷路２を示す配列を返す

#Qテーブルの作成(行数、列数)
def make_q_table(n_rows, n_lines):
    n_columns = 4 #各状況下での選択肢の数
    return np.zeros((n_lines, n_rows, n_columns)) #0で初期化したQテーブルを返す

#行動を決定する(現在の位置、ε)
def get_action(position, epsilon):
    max_list = [] #Q値が最大となる行動を格納するリスト
    not_max_list = [] #Q値が最大でない行動を格納するリスト
    #Q値が最大である行動と、そうでない行動を分ける
    data_max = np.max(q[position[0]][position[1]]) #Q値のうち最大の値を求める
    for i in range(4):
        if q[position[0]][position[1]][i] == data_max: #Q値が最大の場合
            max_list.append(i)
        else: #Q値が最大でない場合
            not_max_list.append(i)
    if np.random.rand() < 1 - epsilon: #確率1-ε(ランダムに生成した数が1-ε未満の場合)
        direction = random.choice(max_list) #Q値が最大となった行動を選択する
    else: #確率ε
        if not_max_list == []: #値がすべて一緒（すべての行動のQ値が最大値）の場合
            direction = random.choice(max_list) #すべての行動からランダムで選ぶ
        else: #最大値を持つ行動がある場合
            direction = random.choice(not_max_list) #Q値が最大でない行動からランダムで選ぶ

    return direction #選んだ行動を返す

#迷路を１つ進み、報酬を判断する(迷路、現在の位置、行動)
def move(maze, position, direction):
    #現在の位置をx,yに代入する
    y = position[0]
    x = position[1]
    #行動によって位置を移動する
    if direction == 0:
        y -= 1
    elif direction == 1:
        x += 1
    elif direction == 2:
        y += 1
    elif direction == 3:
        x -= 1
    #移動先によって、次の座標と報酬を返す
    if 0 <= x <= 5 and 0 <= y <= 5: #6×6から出ない場合
        if maze[y][x] == 0: #迷路の進路内
            return [y,x], 0 #移動し、報酬０
        elif maze[y][x] == 1: #壁
            return [position[0],position[1]], -0.1 #その場にとどまり、報酬-0.1
        elif maze[y][x] == 2: #ゴール１
            return [y,x], 2 #ゴールに移動し、報酬2
        elif maze[y][x] == 3: #ゴール２
            return [y,x], 1 #ゴールに移動し、報酬１
    else: #6×6から出る場合
        return [position[0],position[1]], -0.1 #その場にとどまり、報酬-0.1

#Q 値を更新する(Qテーブル、現在の位置、移動先の位置、報酬、学習係数α、割引率γ)
def renew_q(q, position, next_position, direction, r, alpha, gamma):
    q[position[0]][position[1]][direction] = \
        (1 - alpha) * q[position[0]][position[1]][direction] + \
            alpha * (r + gamma * np.max(q[next_position[0]][next_position[1]])) #Q値を更新する

#Q学習を実行する(迷路、Qテーブル、初期位置、反復回数。εの初期値、学習係数α、割引率γ)
def q_learning(maze, q, start_position, episode, initial_epsilon, alpha, gamma):
    goal_time = [] #ゴールまでの探査回数を記録するリスト
    goal = [] #ゴールした座標を記録するリスト
    for i in range(episode): #反復回数繰り返す
        epsilon = initial_epsilon * (episode-i) / episode #εの値の変更
        search_times = 0 #探査回数のカウント
        while 1:
            position = [random.randint(0,5), random.randint(0,5)]
            if maze[position[0]][position[1]] == 0:
                break
        while 1:
            direction = get_action(position, epsilon) #行動を決定
            next_position, r = move(maze, position, direction) #移動先と、報酬を判断
            renew_q(q, position, next_position, direction, r, alpha, gamma) #Q値を更新する
            position = next_position #位置を移動する
            search_times += 1 #探索回数をカウントする
            if r > 0: #ゴールについた場合
                position_plus = (position[1] + 1, position[0] + 1) #ゴールした座標
                print(search_times,"回の移動で", position_plus , "にゴール!") #探査回数とゴールした座標を表示
                goal_time.append(search_times) #探査回数を記録
                goal.append(position_plus) #ゴールした座標を記録
                break
    return goal_time, goal #探査回数の記録、ゴール地点の記録を返す

#Q値をプロットする(ラベルのリスト、値のリスト、保存するファイル名、マーカーのリスト)
#https://qiita.com/1007/items/80406e098a4212571b2e を参考に作成
def plot_polar(labels, values, imgname, markers):
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1, endpoint=True)
    values = np.concatenate((values, [values[0]]))  # 閉じた多角形にする
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    for i in range(4):
        ax.plot(angles[i], values[i], marker = markers[i], markersize=15, markerfacecolor="r")  # 外枠
    ax.fill(angles, values, alpha=0.25)  # 塗りつぶし
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels, fontname="MS Gothic")  # 軸ラベル
    ax.set_rlim(-1 ,2)
    fig.savefig(imgname)
    plt.close(fig)

#折れ線グラフを作成する（値のリスト、値の数）
def plot_line_graph(data,times):
    x = list(range(1,times+1))
    y = data
    plt.plot(x,y)
    plt.show()

#データの最後１０％部分の折れ線グラフの作成（値のリスト、値の数）
def plot_line_graph2(data,times):
    x = list(range(1,times+1))
    y = data
    plt.xlim([times*0.9,times])
    plt.ylim([0,15])
    plt.plot(x,y)
    plt.show()

#点のグラフの作成（値のリスト、値の数）
def plot_dot_graph(data,times):
    x = list(range(1,times+1))
    y = data
    plt.plot(x,y, marker=".", markersize = 0.2, linewidth=0)
    plt.show()

"""以下３つは画像を結合する関数
https://raw.githubusercontent.com/nkmk/python-snippets/4e232ef06628025ef6d3c4ed7775f5f4e25ebe19/notebook/data/dst/pillow_concat_tile_resize.jpg から引用 """

#画像を結合する関数(画像が入った２次元配列)
def get_concat_tile_resize(im_list_2d, resample=Image.BICUBIC):
    im_list_v = [get_concat_h_multi_resize(im_list_h, resample=resample) for im_list_h in im_list_2d]
    return get_concat_v_multi_resize(im_list_v, resample=resample)

#画像を横に並べる関数(画像の入った配列)
def get_concat_h_multi_resize(im_list, resample=Image.BICUBIC):
    height = im_list[0].height
    total_width = sum(im.width for im in im_list)
    dst = Image.new('RGB', (total_width, height))
    pos_x = 0
    for im in im_list:
        dst.paste(im, (pos_x, 0))
        pos_x += im.width
    return dst

#画像を縦に並べる関数(画像の入った配列)
def get_concat_v_multi_resize(im_list, resample=Image.BICUBIC):
    width = im_list[0].width
    total_height = sum(im.height for im in im_list)
    dst = Image.new('RGB', (width, total_height))
    pos_y = 0
    for im in im_list:
        dst.paste(im, (0, pos_y))
        pos_y += im.height
    return dst



if __name__ == "__main__":
    n_rows = 6 #行の数を指定
    n_lines = 6 #列の数を指定
    maze_name = "迷路2" #迷路１か迷路２を指定
    #maze = make_maze_1(n_rows, n_lines) #迷路２、もしくは迷路１をmazeに代入
    maze = make_maze_2(n_rows, n_lines)
    np.set_printoptions(suppress=True) #数値の表示をeを用いないようにする
    q = make_q_table(n_rows, n_lines) #Qテーブルの初期化

    start_position = (0,0) #初期位置の指定
    initial_epsilon = 0.5 #εの初期値の指定
    alpha = 0.1 #学習係数αの指定
    gamma = 0.9 #割引率γの指定
    episode = 10000 #反復回数の指定
    
    goal_time, goal = q_learning(maze, q, start_position, episode, initial_epsilon, alpha, gamma) #Q学習を実行する。 探査回数と、ゴールの座標の入った配列を返す
    print(q) #学習したQ値を表示

    #各地点のQ値をレーダーチャートで表した画像の作成
    for row in range(n_rows):
        for line in range(n_lines):
            labels = ["右", "上", "左", "下"]
            values = [q[row][line][1],q[row][line][0],q[row][line][3],q[row][line][2]]
            name = maze_name+"_Q値_"+ str(row+1) +"_" + str(line+1) +".png" #保存名の指定
            markers = []
            for i in range(4): #Q値が最大となる行動にのみ、点をつける
                if  values[i] == np.max(values):
                    markers.append("o") 
                else:
                    markers.append("None")
            plot_polar(labels, values, name, markers)
    
    #各地点のQ値の画像を結合
    lis = []
    for i in range(1,n_rows+1):
        li = []
        for j in range(1,n_lines+1):
            li.append(Image.open(maze_name+"_Q値_"+str(i)+"_"+str(j)+".png")) #1つ１つ画像を開く
        lis.append(li) #開いた画像をリストに格納
    get_concat_tile_resize(lis).save(maze_name+"_Q_結果.png")

    #探査回数の推移をグラフで表示
    plot_line_graph(goal_time, episode)
    plot_line_graph2(goal_time, episode)

    #ゴールした地点を点で表したグラフを表示
    goal_str = []
    for x in range(episode):
        if  goal[x] == (2,5):
            goal_str.append("(2,5)")
        elif goal[x] == (5,4):
            goal_str.append("(5,4)")
    plot_dot_graph(goal_str, episode)