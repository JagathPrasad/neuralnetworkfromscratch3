using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork2
{
    public class NeuralNetwork
    {

        //jagged array for AND Gate
        int[][][] inputs = new int[4][][]
                             {
                                 new int[2][]
                                {
                                     new int[2] {  0,0},
                                    new int[1] {  0}

                                },
                                new int[2][]
                                {
                                     new int[2] {  0,1},
                                    new int[1] {  0}

                                },
                                new int[2][]
                                {
                                     new int[2] {  1,0},
                                    new int[1] {  0}

                                },
                                new int[2][]
                                {
                                     new int[2] {  1,1},
                                    new int[1] {  1}

                                },
                             };

        double learningRate = 0.05;


        public NeuralNetwork()
        {
            Random random = new Random();
            double[] initialweights = new double[] { random.NextDouble(), random.NextDouble() };
            bool flag = true;
            int epoch = 0;
            double error = 0;
            while (flag)//training the neural network.
            {
                flag = false;
                error = 0;
                double result = 0;
                Console.WriteLine("Before Training");
                for (int i = 0; i < inputs.Length; i++)
                {
                    result = 0;
                    double sum = CalculateSum(inputs[i][0], initialweights);//forward 
                    result = ActivationFunction(sum);//activation function
                    error = inputs[i][1][0] - result;//calculating the loss/error
                    if (error != 0) flag = true;
                    double[] adjustedWeights = AdjustedWeights(inputs[i][0], initialweights, error);//backpropagate
                    initialweights = adjustedWeights;//updating the weights.
                }
                Console.WriteLine($"After Training :{ epoch}");
                Console.WriteLine($"Output {result} ");
                epoch++;
            }
            Console.ReadLine();
        }

        /// <summary>
        /// calculating the sum with inputs and weights
        /// </summary>
        public double CalculateSum(int[] inputvalues, double[] weights)
        {
            double sum = 0;
            for (int i = 0; i < inputvalues.Length; i++)
            {
                sum = sum + (inputvalues[i] * weights[i]);
            }
            return sum;
        }
        /// <summary>
        /// using the ReLU Activation function to find the propablities of an inputs
        /// </summary>
        public double ActivationFunction(double sum)
            => sum > 1 ? 1 : 0;

        /// <summary>
        /// updating the weights using the backpropagation flow with simple steps.
        /// </summary>
        public double[] AdjustedWeights(int[] inputvalues, double[] weights, double error)
        {
            double[] adujustedWeights = new double[weights.Length];
            for (int i = 0; i < weights.Length; i++)
            {
                adujustedWeights[i] = (learningRate * error * inputvalues[i]) + weights[i];
            }
            return adujustedWeights;
        }

    }

}
