#define alloc_error_check(p) { \
    if ((p) == NULL) { \
        fprintf(stderr, "Allocation Failure!\n"); \
        exit(1); \
    } \
}
#include <iostream>
#include <stdlib.h>
#include <cstdlib>
#include <fstream>
#include <random>
#include <vector>
#include <math.h>

using namespace std;

void free_2d_double(double **dd){
    free(dd[0]);
    free(dd);
}

double **alloc_2d_double(int n1, int n2){
    double **dd, *d;
    int k;
    dd = (double **) malloc(sizeof(double *)*n1);
    alloc_error_check(dd);
    d = (double *) malloc(sizeof(double) * n1 * n2);
    alloc_error_check(d);
    dd[0] = d;
    for (k=1; k<n1; k++){
        dd[k] = dd[k-1] + n2;
    }
    return dd;
}

double random_uniform(double first=4, double last=16){
    uniform_real_distribution<double> distribution(first, last);
    random_device rd;
    default_random_engine generator( rd() );
    return distribution(generator);
}

double SN(double a[], double b[]){
    double x = 0;
    for(int i = 0; i < 2; i++){
        x += pow((a[i] - b[i]),2);
    }
    return sqrt(x);
}

double TN(double sn, double sigma){
    double x = -1 * pow(sn,2) / (2 * pow(sigma,2));
    return exp(x);
}

int indexWithMinValue(double obj[]){
    double value = obj[0];
    int idx = 0;
    for(int i = 0; i < 75; i++){
        if (value > obj[i]){
            value = obj[i];
            idx = i;
        }
    }
    return idx;
}

int main(){
    ifstream myFile;
    int Nx = 600;
    int Ny = 2;
    int i,j,winner_idx = 0, jumlahNeuron = 75;
    double sigma, t_sigma, t_n = 2;
    double n = 0.1;
    double sn, tn_x_n, delta_x_w1, delta_y_w2;
    double winner_val[2],x[2],y[2];
    printf("Perulangan 1\n");
    double **arr = alloc_2d_double(Nx+1, Ny+1);
    double **neuron = alloc_2d_double(jumlahNeuron + 1, Ny);
    double obj_ft_neuron[jumlahNeuron];
    int tetangga[jumlahNeuron];
    myFile.open("Dataset.csv");
    printf("Perulangan 2\n");
    i = 0;
    while (myFile.good() && i < Nx){
        string line, perElement;
        getline(myFile, line, '\n');
        arr[i][0] = atof(line.substr(0, line.find(',')).c_str());
        arr[i][1] = atof(line.substr(line.find(',')+1, line.length()).c_str());
        i++;
    }
    for(int k = 0; k < jumlahNeuron; k++){
        for(int l = 0; l < Ny; l++){
            neuron[k][l] = random_uniform();
        }
    }
    printf("Perulangan 3\n");
    for(int epoch = 0; epoch < 5; epoch++){
        for(int objek = 0; objek < Nx; objek++){
            for(int neur = 0; neur < jumlahNeuron; neur++){
                obj_ft_neuron[neur] = SN(arr[objek], neuron[neur]);
            }
            winner_idx = indexWithMinValue(obj_ft_neuron);
            winner_val[0] = neuron[winner_idx][0];
            winner_val[1] = neuron[winner_idx][1];
            for(int te = 0; te < jumlahNeuron; te++){
                if(SN(neuron[te], winner_val) <= sigma){
                    tetangga[te] = 1;
                }else{
                    tetangga[te] = 0;
                }
            }
            for(int r = 0; r < jumlahNeuron; r++){
                if(tetangga[r] == 1){
                    sn = SN(neuron[r], winner_val);
                    tn_x_n = n * TN(sn, sigma);
                    delta_x_w1 = tn_x_n * (arr[objek][0] - neuron[r][0]);
                    delta_y_w2 = tn_x_n * (arr[objek][1] - neuron[r][1]);
                    neuron[r][0] += delta_x_w1;
                    neuron[r][1] += delta_y_w2;
                }
            }
        }
        sigma = sigma * exp(-1*epoch/t_sigma);
        n = n * exp(-1*epoch/t_n);
    }
    printf("Perulangan 4\n");
    int used[jumlahNeuron];
    for(int y = 0; y < jumlahNeuron; y++){
        used[y] = 0;
    }
    printf("Perulangan 5\n");
    for(int obj = 0; obj < Nx; obj++){
        for(int neur = 0; neur < jumlahNeuron; neur++){
            obj_ft_neuron[neur] = SN(arr[obj], neuron[neur]);
        }
        winner_idx = indexWithMinValue(obj_ft_neuron);
        used[winner_idx] = 1;
    }
    i = 0;
    printf("Perulangan 6\n");
    while(i < jumlahNeuron){
        if (used[i] == 1){
            x[0] = neuron[i][0];
            x[1] = neuron[i][1];
            j = 0;
            while(j < jumlahNeuron){
                if (used[j] == 1){
                    y[0] = neuron[j][0];
                    y[1] = neuron[j][1];
                    double jarak = SN(x,y);
                    if (i != j && jarak < 0.75){
                        used[i] = 0;
                    }
                }
                j++;
            }
        }
        i++;
    }
    printf("Perulangan 7\n");
    for(int k = 0; k < jumlahNeuron; k++){
        cout << used[k] << ", ";
    }
    cout << endl;
    free_2d_double(arr);
    free_2d_double(neuron);
    return 0;
}