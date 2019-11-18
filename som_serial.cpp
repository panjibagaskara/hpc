#define alloc_error_check(p) { \
    if ((p) == NULL) { \
        fprintf(stderr, "Allocation Failure!\n"); \
        exit(1); \
    } \
}
#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <random>
#include <vector>
#include <math.h>

using namespace std;

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

double *alloc_1d_double(int n1){
    double *d;
    
    d = (double *) malloc(sizeof(double) * n1);
    alloc_error_check(d);
    return d;
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
    int Ny, sigma, t_sigma, t_n = 2;
    int i,j,winner_idx = 0;
    double n = 0.1;
    double sn, tn_x_n, delta_x_w1, delta_y_w2;
    double winner_val[2];
    vector<int> tetangga;
    // vector<vector<double>> arr(Nx, vector<double>(Ny, 0));
    double **arr = alloc_2d_double(Nx+1, Ny+1);
    double **neuron = alloc_2d_double(76, Ny);
    double *obj_ft_neuron = alloc_1d_double(76);
    myFile.open("Dataset.csv");
    while (myFile.good()){
        string line, perElement;
        getline(myFile, line, '\n');
        arr[i][j] = atof(line.substr(0, line.find(',')).c_str());
        arr[i][j+1] = atof(line.substr(line.find(',')+1, line.length()).c_str());
        i++;
    }
    for(int k = 0; k < 75; k++){
        for(int l = 0; l < Ny; l++){
            neuron[k][l] = random_uniform();
        }
    }
    for(int epoch = 0; epoch < 5; epoch++){
        for(int objek = 0; objek < Nx; objek++){
            for(int neur = 0; neur < 75; neur++){
                obj_ft_neuron[neur] = SN(arr[objek], neuron[neur]);
            }
            winner_idx = indexWithMinValue(obj_ft_neuron);
            winner_val[0] = neuron[winner_idx][0];
            winner_val[1] = neuron[winner_idx][1];
            for(int te = 0; te < 75; te++){
                if(SN(neuron[te], winner_val) <= sigma){
                    tetangga.push_back(te);
                }
            }
            for(int r = 0; r < tetangga.size(); r++){
                sn = SN(neuron[tetangga[r]], winner_val);
                tn_x_n = n * TN(sn, sigma);
                delta_x_w1 = tn_x_n * (arr[objek][0] - neuron[tetangga[r]][0]);
                delta_y_w2 = tn_x_n * (arr[objek][1] - neuron[tetangga[r]][1]);
                neuron[tetangga[r]][0] += delta_x_w1;
                neuron[tetangga[r]][1] += delta_y_w2;
            }
        }
        //sigma *= exp(-epoch/t_sigma);
        //n *= exp(-epoch/t_n);
    }
    // for(int k = 0; k < 75; k++){
    //     cout << k << ". ";
    //     for(int l = 0; l < Ny; l++){
    //         cout << neuron[k][l] << " ";
    //     }
    //     cout << endl;
    // }
    return 0;
}