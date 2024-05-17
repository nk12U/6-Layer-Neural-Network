#include "nn.h"
#include "MT.h"
#include <time.h>
#define M_PI        3.14159265358979323846264338327950288

void print(int m, int n, const float *x)
{
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
    for (int i = 0; i < m; i++)
    {
        y[i] = b[i]; // y,bはm行1列
        for (int j = 0; j < n; j++)
        {
            y[i] += A[j + i * n] * x[j]; //行数が増えるごとに(行数-1)*nを足せばAの行ベクトルを生成できる
        }
    }
}
void relu(int n, const float *x, float *y)
{ // Rectified Liner Unit(活性化関数)
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
    // xはn行の列ベクトル
    float x_Max = x[0];
    float S = 0;

    //まずx_Maxを求める
    for (int i = 0; i < n; i++)
    {
        if (x[i] >= x_Max)
            x_Max = x[i];
    }
    //分母の和の計算
    for (int i = 0; i < n; i++)
    {
        S += exp(x[i] - x_Max);
    }
    //最終的な式
    for (int i = 0; i < n; i++)
    {
        y[i] = exp(x[i] - x_Max) / S;
    }
}
void softmaxwithloss_bwd(int n, const float *y, unsigned char t, float *dEdx)
{
    for (int i = 0; i < n; i++)
    {
        if (i == t)
            dEdx[i] = y[i] - 1;
        else
            dEdx[i] = y[i];
    }
}
void relu_bwd(int n, const float *x, const float *dEdy, float *dEdx)
{
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
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            // dEdAはm行n列の行列、dEdyはm行列ベクトル、xはn列行ベクトル
            //行列の成分を計算
            dEdA[n * i + j] = dEdy[i] * x[j];
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
            //このa_kはAの第i行列ベクトル
            dEdx[i] += A[i + j * n] * dEdy[j];
        }
    }
}
void shuffle(int n, int *x)
{
    int t = 0;
    for (int i = 0; i < n; i++)
    {
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
    return -log(y[t] + 1e-7);
}
void add(int n, const float *x, float *o)
{
    for (int i = 0; i < n; i++)
    {
        o[i] += x[i];
    }
}
void scale(int n, const float x, float *o)
{
    for (int i = 0; i < n; i++)
    {
        o[i] *= x;
    }
}
void init(int n, const float x, float *o)
{
    for (int i = 0; i < n; i++)
    {
        o[i] = x;
    }
}
void rand_init(int n, float *o)
{
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
    // hは平均,sigmaは標準偏差
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
    float *y1 = malloc(sizeof(float) * 10);
    // inference3
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
void save(const char *filename, int m, int n, const float *A, const float *b)
{
    FILE *fp;
    // Aはm行n列の行列、bはm行の列ベクトル
    //バイナリファイルなので注意！ wbを使う
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
/* 浮動小数点例外で停止することを確認するためのコード */
#if 0
  volatile float x = 0;
  volatile float y = 0;
  volatile float z = x/y;
#endif
    // これ以降，３層NN の係数 A_784x10 および b_784x10 と，
    // 訓練データ train_x + 784*i (i=0,...,train_count-1), train_y[0]～train_y[train_count-1],
    // テストデータ test_x + 784*i (i=0,...,test_count-1), test_y[0]～test_y[test_count-1],
    // を使用することができる．
    /*
    fc(10, 784, train_x, A_784x10, b_784x10, y);
    relu(10, y, y);
    softmax(10, y, y);
    print(1, 10, y);
    */
    //ここまでで補題~4
    /*
    int ans = inference3(A_784x10, b_784x10, train_x);
    printf("%d %d\n", ans, train_y[0]);
    */
    //ここまでで補題5
    /*
    int i = 0;
    save_mnist_bmp(train_x + 784*1, "train_%05d.bmp", i);
    */
    //ここまでで補題6
    /*
    int sum = 0;
    float* y = malloc(sizeof(float) * 10);
    for (int i = 0; i < test_count; i++)
    {
        if(inference3(A_784x10, b_784x10, test_x + i * width * height, y, y) == test_y[i])
          sum++;
    }
    printf("%f%%\n", sum * 100.0 / test_count);
    */
    //ここまでで補題7
    /*
    float *y = malloc(sizeof(float) * 10);
    float *dEdA = malloc(sizeof(float) * 784 * 10);
    float *dEdb = malloc(sizeof(float) * 10);
    backward3(A_784x10, b_784x10, train_x + 784 * 8, train_y[8], y, dEdA, dEdb);
    print(10, 784, dEdA);
    print(1, 10, dEdb);
    */
    //ここまでで補題11
    /*
    int* index = malloc(sizeof(int) * train_count);
    for (int i = 0; i < train_count; i++)
    {
        index[i] = i;
    }
    shuffle(train_count, index);
    */
    //ここまでで補題12
    //必要なメモリ確保
    float *y = malloc(sizeof(float) * 10);
    float *A = malloc(sizeof(float) * 784 * 10);
    float *b = malloc(sizeof(float) * 10);
    float *dEdA = malloc(sizeof(float) * 784 * 10);
    float *dEdb = malloc(sizeof(float) * 10);
    float *ave_dEdA = malloc(sizeof(float) * 784 * 10);
    float *ave_dEdb = malloc(sizeof(float) * 10);
    int *index = malloc(sizeof(int) * train_count);
    //エポック回数を決める
    int epoch = 10;
    //ミニバッチサイズを決める
    int minibatchsize = 100;
    //学習率を決める
    double learningrate = 0.1;
    printf("Epoch size: %7d\nMinibatch size: %2d\nLearning rate: %.2f\n", epoch, minibatchsize, learningrate);
    printf("Calcurating...\n");
    // A,bを[-1:1]の乱数で初期化する
    rand_init(784 * 10, A);
    rand_init(10, b);
    //以下をエポック回数(=10)だけ繰り返す
    for (int i = 0; i < epoch; i++)
    {
        //(a)indexをランダムに並び変える
        for (int i = 0; i < train_count; i++)
            index[i] = i;
        shuffle(train_count, index);
        //(b)以下のミニバッチ学習をN/n(=600)回繰り返す
        // train_count=60000
        for (int j = 0; j < train_count / minibatchsize; j++)
        {
            //(1)平均勾配を0で初期化
            init(784 * 10, 0, ave_dEdA);
            init(10, 0, ave_dEdb);
            //(2)配列indexから次のn=100個を取り出す
            for (int k = 0; k < minibatchsize; k++)
            {
                //(3)対応する学習データと正解データを用いる
                backward3(A, b, train_x + 784 * index[k + minibatchsize * j],
                          train_y[index[k + minibatchsize * j]], y, dEdA, dEdb);
                add(784 * 10, dEdA, ave_dEdA);
                add(10, dEdb, ave_dEdb);
            }
            //(4)平均勾配を得る
            scale(784 * 10, -learningrate / (float)minibatchsize, ave_dEdA);
            scale(10, -learningrate / (float)minibatchsize, ave_dEdb);
            //(5)A,bを更新
            add(784 * 10, ave_dEdA, A);
            add(10, ave_dEdb, b);
        }
        float sum = 0;
        float loss_sum = 0;
        //(c)テストデータに対する損失関数および正解率を計算して表示
        // test_count = 10000
        for (int i = 0; i < test_count; i++)
        {
            if (inference3(A, b, test_x + i * 784, y) == test_y[i])
                sum++;
            loss_sum += cross_entropy_error(y, test_y[i]);
        }
        printf("Epoch:%2d//Accuracy:%.2f%%//Loss:%.2f\n",
               i + 1, sum * 100.0 / test_count, loss_sum / test_count);
    }
    printf("Do you save? Y-0 N-1\n");
    int flag = 0;
    scanf("%d", &flag);
    if (flag == 0)
    {
        save(argv[1], 10, 784, A, b);
    }
    free(y);
    free(A);
    free(b);
    free(dEdA);
    free(dEdb);
    free(ave_dEdA);
    free(ave_dEdb);
    free(index);
    return 0;
}