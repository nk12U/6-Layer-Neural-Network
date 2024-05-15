#include "nn.h"
#include "MT.h"
#include <time.h>
void print(int m, int n, const float *x)
{
    // mは行列の行、nは行列の列、xはm行n列の行列
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%.4f ", x[n * i + j]);
        }
        printf("\n");
    }
}
void fc(int m, int n, const float *x, const float *A, const float *b, float *y)
{
    // mは行列の行、nは行列の列、xはn行の列ベクトル、Aはm行n列の行列、bはm行列ベクトル、yはm行列ベクトル
    // 行列の積和演算
    for (int i = 0; i < m; i++)
    {
        y[i] = b[i]; // y, bはm行の列ベクトル
        for (int j = 0; j < n; j++)
        {
            // Aはm行n列の行列、xはn行の列ベクトル
            y[i] += A[(i * n) + j] * x[j]; // 行数が増えるごとに(行数-1)*nを足せばAの行ベクトルを生成できる
        }
    }
}
void relu(int n, const float *x, float *y)
{
    // nは行列の行数、xはn行列ベクトル、yはn行列ベクトル
    // Rectified Liner Unit(活性化関数)
    for (int i = 0; i < n; i++)
    {
        if (x[i] > 0)
            y[i] = x[i];
        else
            y[i] = 0;
    }
}
void softmax(int n, const float *x, float *y)
{
    // nは行列の行数、xはn行の列ベクトル、yはn行列ベクトル
    // Softmax演算 複数の出力値の合計が1になるような値を返す
    float x_Max = x[0];
    float S = 0;

    // xの最大値をx_Maxに格納している
    for (int i = 0; i < n; i++)
    {
        if (x[i] >= x_Max)
            x_Max = x[i];
    }
    // 分母の和の計算
    for (int i = 0; i < n; i++)
    {
        S += exp(x[i] - x_Max);
    }
    for (int i = 0; i < n; i++)
    {
        y[i] = exp(x[i] - x_Max) / S;
    }
}
void softmaxwithloss_bwd(int n, const float *y, unsigned char t, float *dEdx)
{
    // nは行列の行数、yはn行の列ベクトル、tは正解ラベル、dEdxはn行列ベクトル
    // 誤差逆伝搬(Softmax層)
    for (int i = 0; i < n; i++)
    {
        if (i == t)
            // one-hot表現
            // t[k]はt == kのとき1、t != kのとき0
            dEdx[i] = y[i] - 1;
        else
            dEdx[i] = y[i];
    }
}
void relu_bwd(int n, const float *x, const float *dEdy, float *dEdx)
{
    // nは行列の行数、x,dEdy,dEdxはn行列ベクトル
    // 誤差逆伝搬(ReLU層)
    for (int i = 0; i < n; i++)
    {
        if (x[i] > 0)
            dEdx[i] = dEdy[i];
        else
            dEdx[i] = 0;
    }
}
void fc_bwd(int m, int n, const float *x, const float *dEdy, const float *A,
            float *dEdA, float *dEdb, float *dEdx)
{
    // mは行列の行数、nは行列の列数、xはn列行ベクトル、dEdyはm行列ベクトル、A, dEdAはm行n列の行列、
    // dEdbはm行列ベクトル、dEdxはn列行ベクトル
    // 誤差逆伝搬(FC層)
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            // dEdAはm行n列の行列、dEdyはm行列ベクトル、xはn列行ベクトル
            // 行列の成分を計算
            dEdA[(n * i) + j] = dEdy[i] * x[j];
        }
    }

    for (int i = 0; i < m; i++)
    {
        dEdb[i] = dEdy[i];
    }

    for (int i = 0; i < n; i++)
    {
        dEdx[i] = 0;
        for (int j = 0; j < m; j++)
        {
            // a_iはAの第i列ベクトル
            // 内積の式
            dEdx[i] += A[i + (n * j)] * dEdy[j];
        }
    }
}
void shuffle(int n, int *x)
{
    // nは行列の行数、xはn行の列ベクトル
    int t = 0;
    for (int i = 0; i < n; i++)
    {
        // rand() / (RAND_MAX + 1.0) 0以上1未満の実数を返す
        int j = (int)(n * rand() / (RAND_MAX + 1.0)); // 0~n-1の自然数を返す
        // rand()は0からRAND_MAXまでの数字を取る
        //(RAND_MAX+1.0)が浮動小数点の値になるから全体が浮動少数点数になる
        t = x[i];
        x[i] = x[j];
        x[j] = t;
    }
}
float cross_entropy_error(const float *y, int t)
{
    // 正解ラベルが1に対応する出力の自然対数のみ計算する。
    // yはニューラルネットワークの出力、tは正解ラベルで[0:9]の自然数を取り、そのときのインデックスが1となる。
    // 1e-7は10の-7乗
    return -log(y[t] + 1e-7);
}
void add(int n, const float *x, float *o)
{
    // nは行列の行数、x, oはn行の列ベクトル
    for (int i = 0; i < n; i++)
    {
        o[i] += x[i];
    }
}
void scale(int n, const float x, float *o)
{
    // nは行列の行数、x, oはn行の列ベクトル
    // 配列にxを掛ける
    for (int i = 0; i < n; i++)
    {
        o[i] *= x;
    }
}
void init(int n, const float x, float *o)
{
    // nは行列の行数、x, oはn行の列ベクトル
    // init=Unixでの初期化
    for (int i = 0; i < n; i++)
    {
        o[i] = x;
    }
}
void rand_init(int n, float *o)
{
    // nは行列の行数、oはn行の列ベクトル
    for (int i = 0; i < n; i++)
    {
        o[i] = -1 + (2.0 * rand() / RAND_MAX);
        // (double)(rand() / RAND_MAX) は[0:1] の実数を返すrand()が整数であることに注意
    }
}
double Mersenne_Twister1(void)
{
    // genrand_real3()はMT.hに定義されている(0:1)の一様乱数を返す関数
    return genrand_real3();
}
double Mersenne_Twister2(void)
{
    return genrand_real3();
}
double rand_normal(double h, double sigma)
{
    // hは平均, sigmaは標準偏差
    // ボックス=ミュラー法
    double z = sqrt(-2.0 * log(Mersenne_Twister1())) * sin(2.0 * M_PI * Mersenne_Twister2());
    // 生成した正規分布を線形変換する
    return h + sigma * z;
}
void gauss_init(int n, float *o)
{
    // nは行列の行数、oはn行の列ベクトル
    // 入力がn次元の時, 平均0, 標準偏差sqrt(2/n) のガウス分布を用いる
    for (int i = 0; i < n; i++)
    {
        o[i] = rand_normal(0, sqrt(2.0 / n));
    }
}
int inference3(const float *A, const float *b, const float *x, float *y)
{
    // Aは784*10の配列、bはReLUの出力として用いる10行の列ベクトル、xは教師データの画像を読み込む配列で28*28の行列を表し、yはsoftmaxの出力として用いる10行の列ベクトル
    float *y1 = malloc(sizeof(float) * 10);
    fc(10, 784, x, A, b, y1);
    relu(10, y1, y);
    softmax(10, y, y);
    int inference = 0;
    float max = y[0];
    for (int i = 0; i < 10; i++)
    {
        if (y[i] > max)
        {
            max = y[i];
            inference = i;
        }
    }
    free(y1);
    return inference;
}
void backward3(const float *A, const float *b, const float *x, unsigned char t,
               float *y, float *dEdA, float *dEdb)
{
    // Aは784*10の配列、bはReLUの出力として用いる10行の列ベクトル、xは教師データの画像を読み込む配列で28*28の行列を表し、tは教師データの画像に対する正解ラベル
    // yはsoftmaxの出力として用いる10行の列ベクトル、dEdAは784*10の配列、dEdbは10行の列ベクトル
    // y1はFCの出力として保存しておき逆誤差伝搬法で用いるための配列
    float *y1 = malloc(sizeof(float) * 10);
    // inference3実行部分
    fc(10, 784, x, A, b, y1);
    relu(10, y1, y);
    softmax(10, y, y);

    float *dEdx1 = malloc(sizeof(float) * 10);
    float *dEdx2 = malloc(sizeof(float) * 784);
    softmaxwithloss_bwd(10, y, t, dEdx1);
    relu_bwd(10, y1, dEdx1, dEdx1);
    fc_bwd(10, 784, x, dEdx1, A, dEdA, dEdb, dEdx2);
    free(y1);
    free(dEdx1);
    free(dEdx2);
}
int inference6(const float *A1, const float *b1, const float *A2, const float *b2,
               const float *A3, const float *b3, const float *x, float *y)
{
    // A1はFC1で28行28列(28*28=784)の画像を読み取り50行の列ベクトルとしてReLU1に受け渡すので、配列のサイズは784*50
    // b1は50行の列ベクトル
    // A2はFC2で50行の列ベクトルを受け取り、100行の列ベクトルとしてReLU2に受け渡すので、配列のサイズは50*100
    // b2は100行の列ベクトル
    // A3はFC3で100行の列ベクトルを受け取り、10行の列ベクトルとしてSoftmaxに受け渡すので、配列のサイズは100*10
    // b3は10行の列ベクトル
    // xは教師データの画像を読み込む配列で28*28の行列を表し、yはsoftmaxの出力として用いる10行の列ベクトル
    float *y1 = malloc(sizeof(float) * 50);
    float *y2 = malloc(sizeof(float) * 100);
    fc(50, 784, x, A1, b1, y1);
    relu(50, y1, y1);
    fc(100, 50, y1, A2, b2, y2);
    relu(100, y2, y2);
    fc(10, 100, y2, A3, b3, y);
    softmax(10, y, y);
    float max = y[0];
    int inference = 0;
    for (int i = 0; i < 10; i++)
    {
        if (max < y[i])
        {
            max = y[i];
            inference = i;
        }
    }
    free(y1);
    free(y2);
    return inference;
}
void backward6(const float *A1, const float *b1, const float *A2, const float *b2, const float *A3, const float *b3, const float *x,
               unsigned char t, float *y, float *dEdA1, float *dEdb1, float *dEdA2, float *dEdb2, float *dEdA3, float *dEdb3)
{
    // A1はFC1で28行28列(28*28=784)の画像を読み取り50行の列ベクトルとしてReLU1に受け渡すので、配列のサイズは784*50
    // b1は50行の列ベクトル
    // A2はFC2で50行の列ベクトルを受け取り、100行の列ベクトルとしてReLU2に受け渡すので、配列のサイズは50*100
    // b2は100行の列ベクトル
    // A3はFC3で100行の列ベクトルを受け取り、10行の列ベクトルとしてSoftmaxに受け渡すので、配列のサイズは100*10
    // b3は10行の列ベクトル
    // xは教師データの画像を読み込む配列で28*28の行列を表し、yはsoftmaxの出力として用いる10行の列ベクトル
    // tは教師データの画像に対する正解ラベル
    // dEdA1, dEdb1, dEdA2, dEdb2, dEdA3, dEdb3はA1-b3の勾配で配列の大きさは勾配を取る前に等しい
    // yf1,yr1,yf2,yr2はそれぞれFC1,ReLU1,FC2,ReLU2の出力を保存しておき逆誤差伝搬法で使うための配列
    float *yf1 = malloc(sizeof(float) * 50);
    float *yr1 = malloc(sizeof(float) * 50);
    float *yf2 = malloc(sizeof(float) * 100);
    float *yr2 = malloc(sizeof(float) * 100);
    // inference6実行部分
    fc(50, 784, x, A1, b1, yf1);   // FC1
    relu(50, yf1, yr1);            // ReLU1
    fc(100, 50, yr1, A2, b2, yf2); // FC2
    relu(100, yf2, yr2);           // ReLU2
    fc(10, 100, yr2, A3, b3, y);   // FC3
    softmax(10, y, y);             // Softmax

    float *dEdx1 = malloc(sizeof(float) * 10);
    float *dEdx2 = malloc(sizeof(float) * 100);
    float *dEdx3 = malloc(sizeof(float) * 50);
    float *dEdx4 = malloc(sizeof(float) * 784);
    softmaxwithloss_bwd(10, y, t, dEdx1);                 // Softmax
    fc_bwd(10, 100, yr2, dEdx1, A3, dEdA3, dEdb3, dEdx2); // FC3
    relu_bwd(100, yf2, dEdx2, dEdx2);                     // ReLU2
    fc_bwd(100, 50, yr1, dEdx2, A2, dEdA2, dEdb2, dEdx3); // FC2
    relu_bwd(50, yf1, dEdx3, dEdx3);                      // ReLU1
    fc_bwd(50, 784, x, dEdx3, A1, dEdA1, dEdb1, dEdx4);   // FC1
    free(yf1);
    free(yr1);
    free(yf2);
    free(yr2);
    free(dEdx1);
    free(dEdx2);
    free(dEdx3);
    free(dEdx4);
}
void save(const char *filename, int m, int n, const float *A, const float *b)
{
    // filenameはセーブするdatファイル名、mは行列の行、nは行列の列、Aはm行n列の行列、bはm行の列ベクトル
    // バイナリファイルなので注意！ wbを使う
    FILE *fp;
    if ((fp = fopen(filename, "wb")) == NULL)
    {
        printf("File cannot open.\n");
        exit(1);
    }
    fwrite(A, sizeof(float), m * n, fp);
    fwrite(b, sizeof(float), m, fp);
    printf("%s is saved.\n", filename);
    fclose(fp);
}
int main(int argc, char *argv[])
{
    float *train_x = NULL;
    unsigned char *train_y = NULL;
    int train_count = -1;
    float *test_x = NULL;
    unsigned char *test_y = NULL;
    int test_count = -1;
    int width = -1;
    int height = -1;
    load_mnist(&train_x, &train_y, &train_count, &test_x, &test_y, &test_count, &width, &height);
    srand(time(NULL));

    float *y = malloc(sizeof(float) * 10);
    float *A1 = malloc(sizeof(float) * 784 * 50);
    float *b1 = malloc(sizeof(float) * 50);
    float *A2 = malloc(sizeof(float) * 50 * 100);
    float *b2 = malloc(sizeof(float) * 100);
    float *A3 = malloc(sizeof(float) * 100 * 10);
    float *b3 = malloc(sizeof(float) * 10);
    float *dEdA1 = malloc(sizeof(float) * 784 * 50);
    float *dEdb1 = malloc(sizeof(float) * 50);
    float *dEdA2 = malloc(sizeof(float) * 50 * 100);
    float *dEdb2 = malloc(sizeof(float) * 100);
    float *dEdA3 = malloc(sizeof(float) * 100 * 10);
    float *dEdb3 = malloc(sizeof(float) * 10);
    float *ave_dEdA1 = malloc(sizeof(float) * 784 * 50);
    float *ave_dEdb1 = malloc(sizeof(float) * 50);
    float *ave_dEdA2 = malloc(sizeof(float) * 50 * 100);
    float *ave_dEdb2 = malloc(sizeof(float) * 100);
    float *ave_dEdA3 = malloc(sizeof(float) * 100 * 10);
    float *ave_dEdb3 = malloc(sizeof(float) * 10);
    int *index = malloc(sizeof(int) * train_count);
    // エポック回数を決める
    int epoch = 10;
    // ミニバッチサイズを決める
    int minibatchsize = 100;
    // 学習率を決める
    double learningrate = 0.1;
    printf("Epoch size: %7d\nMinibatch size: %2d\nLearning rate: %.2f\n", epoch, minibatchsize, learningrate);
    printf("Calcurating...\n");
    // A, bを平均0、標準偏差sqrt(2.0/入力次数)の正規分布で初期化する
    gauss_init(784 * 50, A1);
    gauss_init(50, b1);
    gauss_init(50 * 100, A2);
    gauss_init(100, b2);
    gauss_init(100 * 10, A3);
    gauss_init(10, b3);
    // 以下をエポック回数だけ繰り返す
    for (int i = 0; i < epoch; i++)
    {
        //(a)indexをランダムに並び変える indexは0~59999の自然数
        // train_count=60000
        for (int i = 0; i < train_count; i++)
            index[i] = i;
        shuffle(train_count, index);
        //(b)以下のミニバッチ学習をtrain_count / minibatchsize (=600)回繰り返す
        for (int j = 0; j < train_count / minibatchsize; j++)
        {
            //(1)平均勾配を0で初期化
            init(784 * 50, 0, ave_dEdA1);
            init(50, 0, ave_dEdb1);
            init(50 * 100, 0, ave_dEdA2);
            init(100, 0, ave_dEdb2);
            init(100 * 10, 0, ave_dEdA3);
            init(10, 0, ave_dEdb3);
            //(2)配列indexから次のn=minibatchsize個を取り出す
            for (int k = 0; k < minibatchsize; k++)
            {
                //(3)対応する学習データと正解データを用いる
                // train_xは28*28=784の画像データなので
                // j*minibatchsizeで現在のミニバッチ学習を示しており、それをforループ変数kで一つずつ動かす。
                backward6(A1, b1, A2, b2, A3, b3, train_x + 784 * index[(j * minibatchsize) + k],
                          train_y[index[(j * minibatchsize) + k]], y, dEdA1, dEdb1, dEdA2, dEdb2, dEdA3, dEdb3);
                add(784 * 50, dEdA1, ave_dEdA1);
                add(50, dEdb1, ave_dEdb1);
                add(50 * 100, dEdA2, ave_dEdA2);
                add(100, dEdb2, ave_dEdb2);
                add(100 * 10, dEdA3, ave_dEdA3);
                add(10, dEdb3, ave_dEdb3);
            }
            //(4)平均勾配を得る
            // 平均勾配はave_dEdA1をminibatchsizeで割ったもの
            // それに学習率をかけている
            scale(784 * 50, -learningrate / (float)minibatchsize, ave_dEdA1);
            scale(50, -learningrate / (float)minibatchsize, ave_dEdb1);
            scale(50 * 100, -learningrate / (float)minibatchsize, ave_dEdA2);
            scale(100, -learningrate / (float)minibatchsize, ave_dEdb2);
            scale(100 * 10, -learningrate / (float)minibatchsize, ave_dEdA3);
            scale(10, -learningrate / (float)minibatchsize, ave_dEdb3);
            //(5)A,bを更新
            add(784 * 50, ave_dEdA1, A1);
            add(50, ave_dEdb1, b1);
            add(50 * 100, ave_dEdA2, A2);
            add(100, ave_dEdb2, b2);
            add(100 * 10, ave_dEdA3, A3);
            add(10, ave_dEdb3, b3);
        }
        float sum = 0;
        float loss_sum = 0;
        //(c)テストデータに対する損失関数および正解率を計算して表示
        // test_count = 10000
        for (int i = 0; i < test_count; i++)
        {
            if (inference6(A1, b1, A2, b2, A3, b3, test_x + i * 784, y) == test_y[i])
                sum++;
            loss_sum += cross_entropy_error(y, test_y[i]);
        }
        printf("Epoch:%2d//Accuracy:%.2f%%//Loss:%.2f\n",
               i + 1, sum * 100.0 / test_count, loss_sum / test_count);

        // 学習率を正解率によって更新
        if (sum * 100.0 / test_count >= 95.0 && sum * 100.0 / test_count < 96.0)
            learningrate = 0.05;

        if (sum * 100.0 / test_count >= 96.0 && sum * 100.0 / test_count < 97.0)
            learningrate = 0.025;

        if (sum * 100.0 / test_count >= 97.0)
            learningrate = 0.01;
    }
    printf("Do you save? Y-0 N-1\n");
    int flag = 0;
    scanf("%d", &flag);
    if (flag == 0)
    {
        save(argv[1], 50, 784, A1, b1);
        save(argv[2], 100, 50, A2, b2);
        save(argv[3], 10, 100, A3, b3);
    }
    // 使ったメモリ解放
    free(y);
    free(A1);
    free(b1);
    free(A2);
    free(b2);
    free(A3);
    free(b3);
    free(dEdA1);
    free(dEdb1);
    free(dEdA2);
    free(dEdb2);
    free(dEdA3);
    free(dEdb3);
    free(ave_dEdA1);
    free(ave_dEdb1);
    free(ave_dEdA2);
    free(ave_dEdb2);
    free(ave_dEdA3);
    free(ave_dEdb3);
    free(index);
    return 0;
}