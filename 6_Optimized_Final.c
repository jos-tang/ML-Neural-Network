#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

#define ROWS 100
#define COLS 10
double n = 0.05; //learning rate
double beta = 0.9, epsilon = 1e-08;//RMSProp parameters
int iteration = 1;
double Training[90][10],Testing[10][10],prediction[90],testprediction[10];
double MAE,MMSE,initialMMSE,initialtestMMSE,finaltestMMSE;
double Zi[90][7],swishsigmoid[90][7],swish[90][7],testZi[10][7],testswishsigmoid[10][7],testswish[10][7]; //Swish Activation for Training & Testing
double hiddenweight[9][7],hiddenbias[7],derivativehw[9][7],derivativehb[7]; //Weight and bias from 9 Input Layers to 7 Hidden Layers
double outputZi[90],sigmoid[90],testoutputZi[10],testsigmoid[10]; //Sigmoid Activation for Training & Testing
double outputweight[7],bias,derivativeow[7],derivativeb; //Weight and bias from 7 Hidden Layers to 1 Output Layer
double rmspropow[7],rmspropb,rmsprophw[9][7],rmsprophb[7];//RMSProp Optimizer variable for each Weight and Bias
float TN,TP,FN,FP, tTN, tTP, tFN, tFP; //Confusion Matrix for Training and Testing set
float accuracy,testaccuracy;//Accuracy of confusion matrix, taking (TP + TN)/Total
double exectime = 0;
clock_t begin = 0,end = 0;
void genrand();//Randomizes all weights and bias, once at the start, in the range -1<x<1
void GetDataset();//Retrieves data from fertility.txt file and splits them into Training and Testing sets
void MachineTraining();//Consists of forward propagation, MAE, backward propagation and Weight & Bias updates
void SwishActivation();//First forward propagation from Input to Hidden layer using Swish Activation
void SigmoidActivation();//Second forward propagation from Hidden to Output layer using Sigmoid Activation
void GetMAE();//Calculates MAE
void BackProp();//First backward propagation from Output to Hidden to find derivative
void HiddenBackProp();//Second backward propagation from Hidden to Input to find derivative
void WeightBiasUpdate();//Updates weight and bias using derivatives calculated from backward propagation
void GetMMSE();//Calculate MMSE
void GetTesting();//With the finalised weight and bias, forward propagate testing values.
void GetPrediction();//With the finalised weight and bias, gets the prediction of semen diagnosis by rounding up sigmoid value
void ConfusionMatrix();//Matches our prediction with the predicted values.
void ExecutionTime();//Calculates how long our program takes
void PlotGraph();//Plots MAE vs Iteration graph

int main()
{   
    begin = clock();//Starts the clock timer at the beginning of our program
    GetDataset();//First we extract the data from the file
    genrand();//Then we generate random values for our weight and bias in the range -1<x<1
    GetTesting();
    MachineTraining();//Then we run Machine Learning until we get our desired MAE 
    GetTesting();//We use the updated weights and bias to forward propagate our testing set
    GetMMSE();//After training, we get the final MMSE value
    GetPrediction();//Get prediction for the training and testing set with the final weight and bias
    ConfusionMatrix();//Match the predicted values with the actual value
    printf("\n-----------Training Set-----------\n");
    printf("The initial MMSE is %f\n",initialMMSE);
    printf("The final MMSE is %f\n", MMSE);
    printf("Confusion Matrix \nTP: %.0f  TN: %.0f\nFP: %.0f  FN: %.0f\n", TP,TN,FP,FN);
    printf("Accuracy for Training set is %.2f%%\n",accuracy);
    printf("\n-----------Testing Set-----------");
    printf("\nThe initial MMSE is %f\n",initialtestMMSE);
    printf("The final MMSE is %f\n", finaltestMMSE);
    printf("Testing set Confusion Matrix \nTP: %.0f  TN: %.0f\nFP: %.0f  FN: %.0f\n", tTP,tTN,tFP,tFN);
    printf("Accuracy for Testing set is %.2f%%\n",testaccuracy);
    end = clock();//Stop our clock timer
    ExecutionTime();//Find the difference between end time and start time to get the duration of our program
    printf("\nExecution Time is: %lf seconds\n",exectime);
    //PlotGraph();
}   

void GetDataset()
{
    FILE *file_ptr;
    file_ptr = fopen ("fertility_Diagnosis_Data_Group5_8.txt", "r"); //Opens the text file which has to be in the same folder as our code

    for (int i = 0; i < ROWS; i++)
    {
        for (int j = 0; j < COLS; j++)
            if ( i < 90)
           {
                fscanf(file_ptr, "%lf%*c", &Training[i][j]); //Add data in file to the training array
           }
           else
           {
                fscanf(file_ptr, "%lf%*c", &Testing[i-90][j]); //Add data in file to the testing array
           }
    }
    fclose (file_ptr);//Close our text file
}

void genrand()
{
    srand(time(NULL));//We use time as our seed for rand() so that the randomised values will always change per program execution
    for(int a=0;a<7;a++)
    {
        for (int i=0;i<9;i++)
        {
            hiddenweight[i][a] = ((double)rand()/(double)RAND_MAX)*(1-(-1))+(-1);
        }
        hiddenbias[a] = ((double)rand()/(double)RAND_MAX)*(1-(-1))+(-1);
        outputweight[a] = ((double)rand()/(double)RAND_MAX)*(1-(-1))+(-1);
    }
    bias = ((double)rand()/(double)RAND_MAX)*(1-(-1))+(-1);
}

void SwishActivation()
{
    for(int a=0; a < 7 ; a++) //7 Hidden layer neurons
    {
        for(int i = 0; i < 90; i++) //90 volunteers
        {
            Zi[i][a]=0;//Resets to 0 after each iteration        
            for(int j = 0; j < 9; j++) //9 weights per neuron
            {
                Zi[i][a] += (hiddenweight[j][a]*Training[i][j]);
            }
            Zi[i][a] += hiddenbias[a];
            swishsigmoid[i][a] = 1/(1+exp(-1*Zi[i][a]));
            swish[i][a] = Zi[i][a]*swishsigmoid[i][a]; 
        }
    }
}

void SigmoidActivation()
{
    for(int i = 0; i<90;i++)//90 Volunteers
    {
        outputZi[i]=0;//Resets to 0 after each iteration
        for(int a=0 ; a<7 ; a++)
        {
            outputZi[i] += (outputweight[a]*swish[i][a]);
        }
    outputZi[i] += bias;
    sigmoid[i] = 1/(1+exp(-1*outputZi[i]));
    }
}

void GetMAE()
{
    MAE = 0;
    for (int i=0;i<90;i++)
    {
    MAE += fabs(sigmoid[i]-Training[i][9]);
    }
    MAE /= 90;
}

void BackProp()
{
    for (int a = 0; a < 7; a++)
    {
        derivativeow[a] = 0;//Resets to 0 after each iteration
        for (int i = 0; i < 90; i++)
        {
            derivativeow[a] += (sigmoid[i]-Training[i][9])*(exp(outputZi[i])/pow(1+exp(outputZi[i]),2))*swish[i][a];//Derivative of the weight
        }
        derivativeow[a] /= 90;
        rmspropow[a] = (beta*rmspropow[a])+((1-beta)*pow(derivativeow[a],2));//RMSProp Optimizer variable
    }
    derivativeb = 0;//Resets to 0 after each iteration
    for (int i = 0; i < 90; i++)
    {
        derivativeb += (sigmoid[i]-Training[i][9])*(exp(outputZi[i])/pow(1+exp(outputZi[i]),2));//Derivative of the bias
    }
    derivativeb /= 90;
    rmspropb = (beta*rmspropb)+((1-beta)*pow(derivativeb,2));//RMSProp Optimizer variable

}

void HiddenBackProp()
{
    for (int a =0;a<7;a++)
    {
        for (int j = 0; j < 9; j++) 
        {
            derivativehw[j][a] =0;//Resets to 0 after each iteration
            for (int i = 0; i < 90; i++)
            {
                {
                derivativehw[j][a] += (sigmoid[i]-Training[i][9])*(exp(outputZi[i])/pow(1+exp(outputZi[i]),2))*outputweight[a]*(swish[i][a]+(swishsigmoid[i][a]*(1-swish[i][a])))*Training[i][j];//Derivative of weight
                }
            }
            derivativehw[j][a] /= 90;
            rmsprophw[j][a] = (beta*rmsprophw[j][a])+((1-beta)*pow(derivativehw[j][a],2));//RMSProp Optimizer variable
        }
    
    derivativehb[a] = 0;//Resets to 0 after each iteration
    for (int i = 0; i < 90; i++)
    {
        derivativehb[a] += (sigmoid[i]-Training[i][9])*(exp(outputZi[i])/pow(1+exp(outputZi[i]),2))*outputweight[a]*(swish[i][a]+(swishsigmoid[i][a]*(1-swish[i][a])));//Derivative of bias
    }
    derivativehb[a] /= 90;
    rmsprophb[a] = (beta*rmsprophb[a])+((1-beta)*pow(derivativehb[a],2));//RMSProp Optimizer variable
    }
}

void GetMMSE()
{
    if(iteration==1)
    {
        for(int a=0; a<90; a++)
        {   
            if(a<10)
            {
                initialtestMMSE += pow((testsigmoid[a]-Testing[a][9]),2);
            }
            initialMMSE += pow((sigmoid[a]-Training[a][9]),2);
        }
        initialtestMMSE /= 10;
        initialMMSE /= 90;
    }
    else
    {
        for(int a=0; a<90; a++)
        {   
            if(a<10)
            {
                finaltestMMSE += pow((testsigmoid[a]-Testing[a][9]),2);
            }    
            MMSE += pow((sigmoid[a]-Training[a][9]),2);
        }
        finaltestMMSE /= 10;
        MMSE /= 90;
    } 
}

void ConfusionMatrix()
{
    for(int a=0; a<90;a++)//Confusion Matrix for Training Set
    {
        if(prediction[a]==Training[a][9])
        {
            if(prediction[a]==0)
            {
                TN++;//If both predicted and actual value is 0, then it is a True Negative
            }
            else
            {
                TP++;//If both prediction and actual value is 1, then it is a True Positive
            }
        }
        else
        {
            if(prediction[a]==1)
            {
                FP++;//If prediction is 1 but actual is 0, then it is a False Positive
            }
            else
            {
                FN++;//If prediction is 0 but actual is 1, then it is a False Negative
            }
        }
    }
    for (int a=0; a<10; a++)//Confusion Matrix for testing set
    {
        if(testprediction[a]==Testing[a][9])
        {
            if(testprediction[a]==0)
            {
                tTN++;
            }
            else
            {
                tTP++;
            }
        }
        else
        {
            if(testprediction[a]==1)
            {
                tFP++;
            }
            else
            {
                tFN++;
            }
        }
    }
    accuracy = (TP+TN)/90*100;
    testaccuracy = (tTP+tTN)/10*100;
}

void GetPrediction()
{
    for(int a =0; a<90; a++)//Prediction for training
    {
        prediction[a] = round(sigmoid[a]);
        if (a < 10)//Prediction for testing
        {
            testprediction[a] = round(testsigmoid[a]);
        }
        
    }
}

void WeightBiasUpdate()
{
    for(int a=0;a<7;a++)//Updates weight and bias using RMSProp Optimizer
    {    
        for(int i = 0; i < 9;i++)
        {
            hiddenweight[i][a] -= n*derivativehw[i][a]*(1/(sqrt(rmsprophw[i][a])+epsilon));
        }
        hiddenbias[a] -= n*derivativehb[a]*(1/(sqrt(rmsprophb[a])+epsilon));
        outputweight[a] -= n*derivativeow[a]*(1/(sqrt(rmspropow[a])+epsilon));
    }
    bias -= n*derivativeb*(1/(sqrt(rmspropb)+epsilon));
}

void MachineTraining()
{
    FILE *GraphFile_ptr;
    GraphFile_ptr = fopen("GraphFile.txt","w");
    do
    {    
    SwishActivation();
    SigmoidActivation();
    if (iteration == 1)
    {
        GetMMSE();
    }
    GetMAE();
    BackProp();
    HiddenBackProp();
    WeightBiasUpdate();
    printf("Iter %d is MAE %f\n",iteration,MAE);
    fprintf(GraphFile_ptr, "%d %lf\n",iteration,MAE);
    iteration++;
    }
    while (MAE>0.046);
    fclose(GraphFile_ptr);
}

void GetTesting()
{
    for(int a=0; a < 7 ; a++) //7 hidden neurons
    {
        for(int i = 0; i < 10; i++) //10 volunteers
        {
            testZi[i][a] = 0;
            for(int j = 0; j < 9; j++) // 9 weights per neuron
            {
                testZi[i][a] += (hiddenweight[j][a]*Testing[i][j]);
            }
            testZi[i][a] += hiddenbias[a];
            testswishsigmoid[i][a] = 1/(1+exp(-1*testZi[i][a]));
            testswish[i][a] = testZi[i][a]*testswishsigmoid[i][a]; 
        }
    }
    for(int i = 0; i< 10;i++)
    {
        testoutputZi[i]=0;
        for(int a=0 ; a<7 ; a++)
        {
            testoutputZi[i] += (outputweight[a]*testswish[i][a]);
        }
    testoutputZi[i] += bias;
    testsigmoid[i] = 1/(1+exp(-1*testoutputZi[i]));
    }
}

void ExecutionTime()
{
    exectime = (double)(end-begin) / CLOCKS_PER_SEC;
}
 
void PlotGraph()
{
    FILE *script;
    char *plotFileName = "GraphFile.txt";
    char *scriptFileName = "PlotGraph.txt";
    script = fopen("PlotGraph.txt", "w");
    fprintf(script, "plot \"%s\" using 1:2 with lines\n", plotFileName);
    fclose(script);
    system(".\\gnuplot\\bin\\gnuplot -p PlotGraph.txt");
    remove(plotFileName);
    remove(scriptFileName);
}


