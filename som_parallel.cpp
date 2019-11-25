#define alloc_error_check(p)                          \
    {                                                 \
        if ((p) == NULL)                              \
        {                                             \
            fprintf(stderr, "Allocation Failure!\n"); \
            exit(1);                                  \
        }                                             \
    }
#include <iostream>
#include <stdlib.h>
#include <cstdlib>
#include <fstream>
#include <random>
#include <vector>
#include <math.h>
#include "omp.h"

using namespace std;

void free_2d_double(double **dd)
{
    free(dd[0]);
    free(dd);
}

double **alloc_2d_double(int n1, int n2)
{
    double **dd, *d;
    int k;
    dd = (double **)malloc(sizeof(double *) * n1);
    alloc_error_check(dd);
    d = (double *)malloc(sizeof(double) * n1 * n2);
    alloc_error_check(d);
    dd[0] = d;
    for (k = 1; k < n1; k++)
    {
        dd[k] = dd[k - 1] + n2;
    }
    return dd;
}

double random_uniform(double first = 4, double last = 16)
{
    uniform_real_distribution<double> distribution(first, last);
    random_device rd;
    default_random_engine generator(rd());
    return distribution(generator);
}

double SN(double a[], double b[])
{
    double x = 0;
    for (int i = 0; i < 2; i++)
    {
        x += pow((a[i] - b[i]), 2);
    }
    return sqrt(x);
}

double TN(double sn, double sigma)
{
    double x = -1 * pow(sn, 2) / (2 * pow(sigma, 2));
    return exp(x);
}

int indexWithMinValue(double obj[], int jumlahNeuron)
{
    double value = obj[0];
    int idx = 0;
    for (int i = 0; i < jumlahNeuron; i++)
    {
        if (value > obj[i])
        {
            value = obj[i];
            idx = i;
        }
    }
    return idx;
}

int main()
{
    // omp_set_num_threads(4);
    // printf("%d\n", omp_get_num_threads());
    double begin = omp_get_wtime();
    ifstream myFile;
    int Nx = 600;
    int Ny = 2;
    int i, j, winner_idx = 0, jumlahNeuron = 150;
    double sigma, t_sigma, t_n = 2;
    double n = 0.1;
    double sn, tn_x_n, delta_x_w1, delta_y_w2;
    double winner_val[2], x[2], y[2];
    double **arr = alloc_2d_double(Nx + 1, Ny + 1);
    double **neuron = alloc_2d_double(jumlahNeuron + 1, Ny);
    double *obj_ft_neuron = new double[jumlahNeuron];
    int *tetangga = new int[jumlahNeuron];
    int *used = new int[jumlahNeuron];
    myFile.open("Dataset.csv");

    i = 0;
    while (myFile.good() && i < Nx)
    {
        string line, perElement;
        getline(myFile, line, '\n');
        arr[i][0] = atof(line.substr(0, line.find(',')).c_str());
        arr[i][1] = atof(line.substr(line.find(',') + 1, line.length()).c_str());
        i++;
    }
#pragma omp parallel for collapse(2)
    for (int k = 0; k < jumlahNeuron; k++)
    {
        for (int l = 0; l < Ny; l++)
        {
            neuron[k][l] = random_uniform();
        }
    }
#pragma omp parallel for \
private(winner_idx, winner_val, obj_ft_neuron) reduction(*: sigma) reduction(*: n)
    for (int epoch = 0; epoch < 1000; epoch++)
    {
        for (int objek = 0; objek < Nx; objek++)
        {
            obj_ft_neuron = new double[jumlahNeuron];
            for (int neur = 0; neur < jumlahNeuron; neur++)
            {
                obj_ft_neuron[neur] = SN(arr[objek], neuron[neur]);
            }
            winner_idx = indexWithMinValue(obj_ft_neuron, jumlahNeuron);
            winner_val[0] = neuron[winner_idx][0];
            winner_val[1] = neuron[winner_idx][1];
            for (int te = 0; te < jumlahNeuron; te++)
            {
                if (SN(neuron[te], winner_val) <= sigma)
                {
                    tetangga[te] = 1;
                }
                else
                {
                    tetangga[te] = 0;
                }
            }
            for (int r = 0; r < jumlahNeuron; r++)
            {
                if (tetangga[r] == 1)
                {
                    sn = SN(neuron[r], winner_val);
                    tn_x_n = n * TN(sn, sigma);
                    delta_x_w1 = tn_x_n * (arr[objek][0] - neuron[r][0]);
                    delta_y_w2 = tn_x_n * (arr[objek][1] - neuron[r][1]);
                    neuron[r][0] += delta_x_w1;
                    neuron[r][1] += delta_y_w2;
                }
            }
        }
        sigma = sigma * exp(-1 * epoch / t_sigma);
        n = n * exp(-1 * epoch / t_n);
        if (epoch < jumlahNeuron)
        {        
            used[epoch] = 0;
        }
    }
    #pragma omp parallel for private(obj_ft_neuron, winner_idx) shared(used)
    for (int obj = 0; obj < Nx; obj++)
    {
        obj_ft_neuron = new double[jumlahNeuron];
        for (int neur = 0; neur < jumlahNeuron; neur++)
        {
            obj_ft_neuron[neur] = SN(arr[obj], neuron[neur]);
        }
        winner_idx = indexWithMinValue(obj_ft_neuron, jumlahNeuron);
        used[winner_idx] = 1;
    }
#pragma omp parallel for private(x, y)
    for (int i = 0; i < jumlahNeuron; i++)
    {
        if (used[i] == 1)
        {
            x[0] = neuron[i][0];
            x[1] = neuron[i][1];
            for (int j = 0; j < jumlahNeuron; j++)
            {
                if (used[j] == 1)
                {
                    y[0] = neuron[j][0];
                    y[1] = neuron[j][1];
                    double jarak = SN(x, y);
                    if (i != j && jarak < 0.75)
                    {
                        used[i] = 0;
                    }
                }
                j++;
            }
        }
        i++;
    }
    free_2d_double(arr);
    free_2d_double(neuron);
    double end = omp_get_wtime();
    printf("Parallel Time: %g detik\n", end - begin);
    return 0;
}