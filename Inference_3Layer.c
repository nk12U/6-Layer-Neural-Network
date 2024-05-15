#include "nn.h"
void fc(int m, int n, const float *x, const float *A, const float *b, float *y)
{
    //行列の積和演算
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
    // Softmax演算 複数の出力値の合計が1になるような値を返す
    // xはn行の列ベクトル
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
void load(const char *filename, int m, int n, float *A, float *b)
{
    FILE *fp;
    // Aはm行n列の行列、bはm行の列ベクトル
    //バイナリファイルなので注意！ rbを使う
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
    float *A = malloc(sizeof(float) * 784 * 10);
    float *b = malloc(sizeof(float) * 10);
    float *x = load_mnist_bmp(argv[2]);
    int flag = load(argv[1], 10, 784, A, b);
    if (flag == 1)
        return 0;
    int inference = inference3(A, b, x, y);
    printf("\nThis number is infered to be %d.\n\n", inference);
    free(y);
    free(A);
    free(b);
    free(x);
    return 0;
}