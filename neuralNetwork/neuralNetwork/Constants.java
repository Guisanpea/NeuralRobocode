package neuralNetwork;

public class Constants {
    // Minimum allowable battlefield size is 400
    final static int MAXBOARDSIZE = 2000;

    // Minimum allowable gun cooling rate is 0.1
    final static double MAXCOOLINGRATE = 10;
    final static int NUMBATTLEFIELDSIZES = 601;
    final static int NUMCOOLINGRATES = 501;
    final static int NUMSAMPLES = 100;
    // Number of inputs for the multilayer perceptron (size of the input
    // vectors)
    final static int NUM_NN_INPUTS = 2;

    // Number of hidden neurons of the neural network

    final static int NUM_NN_HIDDEN_UNITS = 20;

    // Number of epochs for training
    final static int NUM_OF_ROUNDS = 10;
}
