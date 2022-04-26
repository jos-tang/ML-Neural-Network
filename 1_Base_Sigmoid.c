#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

#define ROWS 100
#define COLS 10
double n = 0.05; //learning rate
int iteration = 1;
double Training[90][10],Testing[10][10],prediction[90],testprediction[10];
double MAE,MMSE,initialMMSE,initialtestMMSE,finaltestMMSE;
double Zi[90],testZi[10],sigmoid[90],testsigmoid[10]; //Sigmoid Activation for Training & Testing
double weight[9],derivativew[9],bias,derivativeb; //Weight and bias from 9 Input Layers to 7 Hidden Layers
float TN,TP,FN,FP, tTN, tTP, tFN, tFP; //Confusion Matrix for Training and Testing set
float accuracy,testaccuracy;//Accuracy of confusion matrix, taking (TP + TN)/Total
double exectime = 0;
clock_t begin = 0,end = 0;
void genrand();//Randomizes all weights and bias, once at the start, in the range -1<x<1
void GetDataset();//Retrieves data from fertility.txt file and splits them into Training and Testing sets
void MachineTraining();//Consists of forward propagation, MAE, backward propagation and Weight & Bias updates
void SwishActivation();//First forward propagation from Input to Hidden layer using Swish Activation
void GetMAE();//Calculates MAE
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
    PlotGraph();
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
    
    
        for (int i=0;i<9;i++)
        {
            weight[i] = ((double)rand()/(double)RAND_MAX)*(1-(-1))+(-1);
        }
    
    bias = ((double)rand()/(double)RAND_MAX)*(1-(-1))+(-1);
}

void SigmoidActivation()
{
    
        for(int i = 0; i < 90; i++) //90 volunteers
        {
            Zi[i]=0;//Resets to 0 after each iteration        
            for(int j = 0; j < 9; j++) //9 weights per neuron
            {
                Zi[i] += (weight[j]*Training[i][j]);
            }
            Zi[i] += bias;
            sigmoid[i] = 1/(1+exp(-Zi[i])); 
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
    
        for (int j = 0; j < 9; j++) 
        {
            derivativew[j] =0;//Resets to 0 after each iteration
            for (int i = 0; i < 90; i++)
            {
                {
                derivativew[j] += (sigmoid[i]-Training[i][9])*(exp(Zi[i])/pow(1+exp(Zi[i]),2))*Training[i][j];//Derivative of weight
                }
            }
            derivativew[j] /= 90;
        }
    
    derivativeb = 0;//Resets to 0 after each iteration
    for (int i = 0; i < 90; i++)
    {
        derivativeb += (sigmoid[i]-Training[i][9])*(exp(Zi[i])/pow(1+exp(Zi[i]),2));//Derivative of bias
    }
    derivativeb /= 90;
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
      
        for(int i = 0; i < 9;i++)
        {
            weight[i] -= n*derivativew[i];
        }
    bias -= n*derivativeb;
}

void MachineTraining()
{
    FILE *GraphFile_ptr;
    GraphFile_ptr = fopen("GraphFile.txt","w");
    do
    {    
    SigmoidActivation();
    if (iteration == 1)
    {
        GetMMSE();
    }
    GetMAE();
    BackProp();
    WeightBiasUpdate();
    printf("Iter %d is MAE %f\n",iteration,MAE);
    fprintf(GraphFile_ptr, "%d %lf\n",iteration,MAE);
    iteration++;
    }
    while (MAE>0.15);
    fclose(GraphFile_ptr);
}

void GetTesting()
{
    
        for(int i = 0; i < 10; i++) //10 volunteers
        {
            testZi[i] = 0;
            for(int j = 0; j < 9; j++) // 9 weights per neuron
            {
                testZi[i] += (weight[j]*Testing[i][j]);
            }
            testZi[i] += bias;
            testsigmoid[i] = 1/(1+exp(-testZi[i]));
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


