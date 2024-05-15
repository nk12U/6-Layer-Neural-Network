#include "nn.h"
void fc(int m, int n, const float *x, const float *A, const float *b, float *y)
{
    // mは行列の行、nは行列の列、xはn行の列ベクトル、Aはm行n列の行列、bはm行列ベクトル、yはm行列ベクトル
    // 行列の積和演算
    for (int i = 0; i < m; i++)
    {
        y[i] = b[i]; // y,bはm行1列
        for (int j = 0; j < n; j++)
        {
            y[i] += A[(i * n) + j] * x[j]; //行数が増えるごとに(行数-1)*nを足せばAの行ベクトルを生成できる
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
    for (int i = 0; i < n; i++)
    {
        y[i] = exp(x[i] - x_Max) / S;
    }
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
void load(const char *filename, int m, int n, float *A, float *b)
{
    // filenameはロードするdatファイル名、Aはm行n列の行列、bはm行の列ベクトル
    // バイナリファイルなので注意！ rbを使う
    FILE *fp;
    if ((fp = fopen(filename, "rb")) == NULL)
    {
        printf("File cannot open.\n");
        exit(1);
    }
    fread(A, sizeof(float), m * n, fp);
    fread(b, sizeof(float), m, fp);
    fclose(fp);
}
int main(int argc, char *argv[])
{
    float *y = malloc(sizeof(float) * 10);
    float *A1 = malloc(sizeof(float) * 784 * 50);
    float *b1 = malloc(sizeof(float) * 50);
    float *A2 = malloc(sizeof(float) * 50 * 100);
    float *b2 = malloc(sizeof(float) * 100);
    float *A3 = malloc(sizeof(float) * 100 * 10);
    float *b3 = malloc(sizeof(float) * 10);
    float *x = load_mnist_bmp(argv[4]);
    load(argv[1], 50, 784, A1, b1);
    load(argv[2], 100, 50, A2, b2);
    load(argv[3], 10, 100, A3, b3);
    int inference = inference6(A1, b1, A2, b2, A3, b3, x, y);
    printf("\nThis number is infered to be %d.\n\n", inference);
    free(y);
    free(A1);
    free(b1);
    free(A2);
    free(b2);
    free(A3);
    free(b3);
    free(x);
    return 0;
}