import numpy as np
import math
import pandas as pd
from PIL import Image
import random
import matplotlib.pyplot as plt

#CSVファイルを読み込む
def read_csv_data():
    csv_input = pd.read_csv(filepath_or_buffer=r"C:\Users\zzzke\Documents\機械学習\em\mnist_em.csv") #ファイルパスを指定して読み込む
    x = csv_input.values #読み込んだ値を数値に変換する
    return x #読み込んだデータの入った配列を返す

#Eステップ、Mステップを行う(クラスター数M,標準偏差σ,データ数T,クラスター中心μ,データx,変化量格納変数dif)
def e_m_step(m, sigma, T, u, x, dif):
    #重み付き和を格納する配列を初期化
    weighted_sum_1 = np.zeros((m))
    weighted_sum_x = np.zeros((m,784))
    u_new = np.zeros((m, 784)) #更新するμを入れる配列を初期化
    for i in range(m):
        p = np.zeros(m)
        for t in range(T):
            #Pの分母となる部分を計算する
            sum_pd = 0
            for j in range(m): 
                sum_pd += math.exp(-np.linalg.norm(x[t]- u[j]) ** 2 / (2 * sigma ** 2))
            
            p[i] = math.exp(- np.linalg.norm(x[t]- u[i]) ** 2 / (2 * sigma ** 2)) / sum_pd #Pを計算する。

            #重み付き和を計算する
            weighted_sum_1[i] += p[i] 
            weighted_sum_x[i] += x[t] * p[i]
        
        u_new[i] = weighted_sum_x[i] / weighted_sum_1[i] #更新するμの値を計算する
        dif[i] = np.linalg.norm(u_new[i] - u[i]) #μの更新量を計算する
        print(dif[i]) #μの更新量を表示する
    dif_mean = sum(dif) / m #μの更新量の平均を計算する
    print("μの更新量の平均は",dif_mean) #μの更新量の平均を表示する
    for i in range(m): #μの値を更新する
        u[i] = u_new[i]    

#EMアルゴリズムを実行する(クラスター数M,標準偏差σ,データx,データ数T,クラスター中心μ,EMステップの最大反復回数)
def em_algorithm(m, sigma, x, T, u, max_time, difs, j):
    for num in range(max_time): #EMステップを繰り返す
        print("今", num+1, "回目") #EMステップの繰り返し回数を表示する
        dif = np.zeros(m) #更新量を格納する変数difを初期化する
        e_m_step(m, sigma, T, u, x, dif) #EMステップを行う

        difs[j].append(sum(dif)/m)

        counter = 0
        for i in range(m): #終了条件を判定する
            if dif[i] < 10e-15:
                counter += 1
        if counter == m:
            return

#K平均法を実行する(クラスター数M,データx,データ数T,クラスター中心μ,EMステップの最大反復回数)
def k_means(m, x, T, u, max_time, difs, j):
    for num in range(max_time): #繰り返す
        print("今", num+1, "回目") #繰り返し回数の表示

        #データをクラスターに分類する
        judge = []
        for i in range(m):
            judge.append([])
        #各データを最も近いクラスター中心のリストに格納する
        for t in range(T):
            dis = []
            for i in range(m):
                dis.append(np.linalg.norm(x[t] - u[i]))
            index_min = np.argmin(dis)
            judge[index_min].append(x[t])
        
        u_new = np.zeros((m, 784))
        judge = np.array(judge)
        dif = np.zeros(m)
        for i in range(m):
            u_new[i] = np.sum(judge[i], axis = 0) / len(judge[i]) #μの値を、平均で更新する
            dif[i] = np.linalg.norm(u_new[i] - u[i]) #μの更新量を計算する
            print(dif[i]) #μの更新量を表示する
            u[i] = u_new[i]

        difs[j].append(sum(dif)/m)

        counter = 0
        for i in range(m): #終了条件を判定する
            if dif[i] < 10e-15:
                counter += 1
        if counter == m:
            return

#配列を画像にして保存する(画像データの入った配列μ)
def img_show(u, j):
    for i in range(len(u)): #画像枚数の分繰り返す
        image = u[i].reshape(28, 28) #784この値の配列を28×28に変換する
        pil_img = Image.fromarray(np.uint8(image)) #配列を画像に変換する
        l = ["04","07","1","2", "5", "10", "k-means"]
        pil_img.save(r'em\結果画像\m_'+str(len(u))+"_"+str(i+1)+"_"+l[j]+".png") #ファイル名を指定して画像を保存する

def graph(difs):
    fig, ax = plt.subplots(figsize=(12.0, 10.0))

    c = ["blue","green","red","black","c", "y", "m"]      # 各プロットの色
    l = ["0.4","0.7","1","2", "5", "10", "k-means"]   # 各ラベル

    plt.yscale("log")
    ax.set_xlabel('反復回数',fontname="MS Gothic")  # x軸ラベル
    ax.set_ylabel('更新量',fontname="MS Gothic")  # y軸ラベル
    ax.set_title('k平均法と、AMアルゴリズムの標準偏差σの変化による動作の違い',fontname="MS Gothic") # グラフタイトル
    #ax.set_aspect('equal') # スケールを揃える
    ax.grid()            # 罫線
    #ax.set_xlim([-10, 10]) # x方向の描画範囲を指定
    #ax.set_ylim([0, 1])    # y方向の描画範囲を指定
    for i in range(len(difs)):
        t = range(1,len(difs[i])+1)
        y = difs[i]
        ax.plot(t, y, color=c[i], label=l[i])

    ax.legend(loc=0)    # 凡例
    #fig.tight_layout()  # レイアウトの設定
    #plt.savefig('hoge.png') # 画像の保存
    plt.show()


if __name__ == "__main__":
    
    m = 3 #クラスタ数Mを指定する
    #sigma = 1 #標準偏差σを指定する
    max_time = 1000 #最大反復回数を指定する
    
    x = read_csv_data() / 255 #csvファイルを読み込む
    T = len(x) #データ数Tを定義する

    #データから、μの初期値となる値をランダムに取り出す処理をする
    x = x.tolist()
    random.seed(2)
    u = random.sample(x, m)
    x = np.array(x)
    u = np.array(u)

    sigma_list = [0.4, 0.7, 1, 2, 5, 10]
    
    i = 0
    difs = [[],[],[],[],[],[],[]]
    for sigma in sigma_list:
        #EMアルゴリズムもしくはk-means法を実行する
        em_algorithm(m, sigma, x, T, u, max_time, difs, i)
        img_show(u * 255, i)
        i += 1
        #μを画像に変換して保存する
        x = x.tolist()
        random.seed(2)
        u = random.sample(x, m)
        x = np.array(x)
        u = np.array(u)

    k_means(m, x, T, u, max_time, difs, i)
    
    #μを画像に変換して保存する
    img_show(u * 255, i)
    
    """difs = np.load(
    file="difs_save.npy",        # npyかnpz拡張子のファイルを指定
    allow_pickle=True,      # npy/npzで保存されたpickleファイルを読み込むかどうか (デフォルトではTrue)
    fix_imports=True,       # Python2との互換機能、Trueならpy2製ファイルを読み込める　（デフォルトではTrue）
    encoding="ASCII",       # Python2で作られた文字列を読み込むときの文字コード指定　（デフォルトではASCII
    )

    print(difs)
    
    graph(difs)
    
    #np.save("difs_save", difs)
    """
