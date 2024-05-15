#include "nn.h"
int main(void){
    float *train_x = NULL;
    unsigned char *train_y = NULL;
    int train_count = -1;

    float *test_x = NULL;
    unsigned char *test_y = NULL;
    int test_count = -1;

    int width = -1;
    int height = -1;
    load_mnist(&train_x, &train_y, &train_count,
             &test_x, &test_y, &test_count,
             &width, &height);    
    int i = 0;
    printf("Input integer: ");
    scanf("%d", &i);
    save_mnist_bmp(train_x + 784 * i, "train_%05d.bmp", i);
    return 0;
}