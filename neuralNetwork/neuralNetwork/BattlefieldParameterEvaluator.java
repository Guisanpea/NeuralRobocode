package neuralNetwork;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import robocode.BattleResults;
import robocode.control.*;
import robocode.control.events.BattleAdaptor;
import robocode.control.events.BattleCompletedEvent;
import robocode.control.events.BattleErrorEvent;
import robocode.control.events.BattleMessageEvent;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import static neuralNetwork.Constants.*;

public class BattlefieldParameterEvaluator {

    private static double maxScore = 0;
    private static int ndxBattle;

    private static double[] finalScore1;
    private static double[] finalScore2;

    private static double[][] RawInputs;
    private static double[][] RawOutputs;

    private static Random rng;
    private static double[] battlefieldSize;
    private static double[] gunCoolingRate;

    public static void main(String[] args){
        initializeVariables();

        RobocodeEngine engine = setupEngine();

        long inactivityTime = 1000;
        int sentryBorderSize = 50;
        RobotSpecification[] sampleRobots = engine.getLocalRepository("sample.RamFire,sample.TrackFire");
        RobotSetup[] robotSetups = new RobotSetup[2];

        runBattles(engine, inactivityTime, sentryBorderSize, false, sampleRobots, robotSetups);
        engine.close();

        printInfo();
        gatherSamples();

        double trainingPercentage = (double) 3 / (double) 4;
        double[][] learningInputs = Arrays.copyOfRange(RawInputs, 0, (int) (NUMSAMPLES * trainingPercentage));
        double[][] learningOutputs = Arrays.copyOfRange(RawOutputs, 0, (int) (NUMSAMPLES * trainingPercentage));
        double[][] testingInputs = Arrays.copyOfRange(RawInputs, (int) (NUMSAMPLES * trainingPercentage), NUMSAMPLES);
        double[][] testingOutputs = Arrays.copyOfRange(RawOutputs, (int) (NUMSAMPLES * trainingPercentage), NUMSAMPLES);

        // Create and train the neural network
        MLDataSet learningSet = new BasicMLDataSet(learningInputs, learningOutputs);
        MLDataSet testingSet = new BasicMLDataSet(testingInputs, testingOutputs);

        BasicNetwork network = createNetwork();

        System.out.println("Starting the training with " +  trainingPercentage + " training percentage");
        network = doTraining(learningSet, testingSet, network);

        System.out.println("Training completed, " + "the best epoch is: " + NetworkTrainer.bestEpoch);

        double error = network.calculateError(testingSet);
        System.out.println("The error is: " + error);

        System.out.println("Testing the network and printing image");
        printImage(network);

        System.exit(0);

    }

    private static void printImage(BasicNetwork network) {
        int[] outputRGBint = new int[NUMBATTLEFIELDSIZES * NUMCOOLINGRATES];
        Color myColor;

        double myValue;
        double[][] myTestData = new double[NUMBATTLEFIELDSIZES * NUMCOOLINGRATES][NUM_NN_INPUTS];
        for (int ndxBattleSize = 0; ndxBattleSize < NUMBATTLEFIELDSIZES; ndxBattleSize++) {
            for (int ndxCooling = 0; ndxCooling < NUMCOOLINGRATES; ndxCooling++)

            {
                myTestData[ndxCooling + ndxBattleSize * NUMCOOLINGRATES][0] = 0.1
                        + 0.9 * ((double) ndxBattleSize) / NUMBATTLEFIELDSIZES;

                myTestData[ndxCooling + ndxBattleSize * NUMCOOLINGRATES][1] = 0.1
                        + 0.9 * ((double) ndxCooling) / NUMCOOLINGRATES;

            }
        }

        // Simulate the neural network with the test samples and fill a matrix
        for (int ndxBattleSize = 0; ndxBattleSize < NUMBATTLEFIELDSIZES; ndxBattleSize++) {
            for (int ndxCooling = 0; ndxCooling < NUMCOOLINGRATES; ndxCooling++)

            {
                double[] myResult = new double[1];
                network.compute(myTestData[ndxCooling + ndxBattleSize * NUMCOOLINGRATES], myResult);
                myValue = ClipColor(myResult[0]);

                myColor = new Color((float) myValue, (float) myValue, (float) myValue);
                outputRGBint[ndxCooling + ndxBattleSize * NUMCOOLINGRATES] = myColor.getRGB();

            }
        }
        System.out.println("Testing completed.");

        // Plot the training samples

        for (int ndxSample = 0; ndxSample < NUMSAMPLES / 2; ndxSample++) {

            myValue = ClipColor(finalScore1[ndxSample] / maxScore);
            myColor = new Color((float) myValue,

                    (float) myValue, (float) myValue);

            int myPixelIndex = (int) (Math
                    .round(NUMCOOLINGRATES * ((gunCoolingRate[ndxSample] / MAXCOOLINGRATE) - 0.1) / 0.9)
                    + Math.round(NUMBATTLEFIELDSIZES * ((battlefieldSize[ndxSample] / MAXBOARDSIZE) - 0.1) / 0.9)
                    * NUMCOOLINGRATES);

            if ((myPixelIndex >= 0) && (myPixelIndex < NUMCOOLINGRATES * NUMBATTLEFIELDSIZES))

            {
                outputRGBint[myPixelIndex] = myColor.getRGB();
            }
        }

        BufferedImage img = new BufferedImage(NUMCOOLINGRATES, NUMBATTLEFIELDSIZES, BufferedImage.TYPE_INT_RGB);

        img.setRGB(0, 0, NUMCOOLINGRATES, NUMBATTLEFIELDSIZES, outputRGBint, 0, NUMCOOLINGRATES);

        File f = new File("hello.png");
        try {

            ImageIO.write(img, "png", f);
        } catch (IOException ioE) {

            // TODO Autoâ€�generated catchblock
            ioE.printStackTrace();

        }

        System.out.println("Image generated.");
    }

    private static BasicNetwork createNetwork() {
        BasicNetwork network = new BasicNetwork();
        network.addLayer(new BasicLayer(null, true, 2)); // Input
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, NUM_NN_HIDDEN_UNITS)); // Hidden
        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1)); // Output
        network.getStructure().finalizeStructure();
        network.reset();
        return network;
    }

    private static BasicNetwork doTraining(MLDataSet learningSet, MLDataSet testingSet, BasicNetwork network) {
        MLTrain train = new ResilientPropagation(network, learningSet);

        int minimum_epochs = 40000;
        int extra_epochs = 2000;
        NetworkTrainer networkTrainer = new NetworkTrainer(testingSet, network, minimum_epochs, extra_epochs);

        do {
            train.iteration();
        } while (!networkTrainer.hasFinished(network)); // testear

        network = networkTrainer.getBestNetwork();
        return network;
    }

    private static void gatherSamples() {
        RawInputs = new double[NUMSAMPLES][NUM_NN_INPUTS];
        RawOutputs = new double[NUMSAMPLES][1];

        for (int NdxSample = 0; NdxSample < NUMSAMPLES; NdxSample++) {
            RawInputs[NdxSample][0] = battlefieldSize[NdxSample] / MAXBOARDSIZE;
            RawInputs[NdxSample][1] = gunCoolingRate[NdxSample] / MAXCOOLINGRATE;
            maxScore = maxOfArray(finalScore1);
            RawOutputs[NdxSample][0] = finalScore1[NdxSample] / (maxScore);
        }
    }

    private static void printInfo() {
        System.out.println(Arrays.toString(battlefieldSize));
        System.out.println(Arrays.toString(gunCoolingRate));
        System.out.println(Arrays.toString(finalScore1));
        System.out.println(Arrays.toString(finalScore2));
    }

    private static void runBattles(RobocodeEngine engine, long inactivityTime, int sentryBorderSize, boolean hideEnemyNames, RobotSpecification[] competingRobots, RobotSetup[] robotSetups) {
        for (ndxBattle = 0; ndxBattle < NUMSAMPLES; ndxBattle++) {

            // Choose the battlefield size and gun cooling rate
            battlefieldSize[ndxBattle] = MAXBOARDSIZE * (0.1 + 0.9 * rng.nextDouble());
            gunCoolingRate[ndxBattle] = MAXCOOLINGRATE * (0.1 + 0.9 * rng.nextDouble());

            // Create the battlefield

            int size = (int) Math.max(400, battlefieldSize[ndxBattle]);
            BattlefieldSpecification battlefield = new BattlefieldSpecification(size,
                    size);

            // Set the robot positions

            robotSetups[0] = new RobotSetup(battlefieldSize[ndxBattle] / 2.0, battlefieldSize[ndxBattle] / 3.0, 0.0);

            robotSetups[1] = new RobotSetup(battlefieldSize[ndxBattle] / 2.0, 2.0 * battlefieldSize[ndxBattle] / 3.0,
                    0.0);

            // Prepare the battle specification
            BattleSpecification battleSpec = new BattleSpecification(battlefield, NUM_OF_ROUNDS, inactivityTime,
                    gunCoolingRate[ndxBattle], sentryBorderSize, hideEnemyNames, competingRobots, robotSetups);

            // Run our specified battle and let it run till it is over
            engine.runBattle(battleSpec, true); // waits till the battle
            // finishes

        }
    }

    private static RobocodeEngine setupEngine() {
        RobocodeEngine.setLogMessagesEnabled(false);
        RobocodeEngine engine = new RobocodeEngine(new File("/home/ubuntie/robocode"));
        engine.addBattleListener(new BattleObserver());
        engine.setVisible(false);
        return engine;
    }

    private static void initializeVariables() {
        rng = new Random(15L);

        battlefieldSize = new double[NUMSAMPLES];
        gunCoolingRate = new double[NUMSAMPLES];

        finalScore1 = new double[NUMSAMPLES];

        finalScore2 = new double[NUMSAMPLES];
    }

    /*
     *
     * Clip a color value (double precision) to lie in the valid range [0,1]
     */

    private static double maxOfArray(double[] arr) {
        double ans = arr[0];
        for (double anArr : arr) {
            ans = Math.max(ans, anArr);
        }
        return ans;
    }

    private static double ClipColor(double value) {

        if (value < 0.0) {
            value = 0.0;
        }
        if (value > 1.0) {
            value = 1.0;
        }

        return value;

    }

    //

    // Our private battle listener for handling the battle event we are
    // interested in.

    //
    static class BattleObserver extends BattleAdaptor {

        // Called when the battle is completed successfully with battle results
        public void onBattleCompleted(BattleCompletedEvent e) {
            System.out.println("â€�â€� Battle has completed â€�â€�");

            // Get the indexed battle results
            BattleResults[] results = e.getIndexedResults();

            // Print out the indexed results with the robot names
            System.out.println("Battle results:");
            for (BattleResults result : results) {
                System.out.println(" " + result.getTeamLeaderName() +

                        ": " + result.getScore());
            }
            // Store the scores of the robots
            BattlefieldParameterEvaluator.finalScore1[ndxBattle] = Math.pow(results[0].getScore(), 2);
            BattlefieldParameterEvaluator.finalScore2[ndxBattle] = Math.pow(results[1].getScore(), 2);
        }

        // Called when the game sends out an information message during the
        // battle
        public void onBattleMessage(BattleMessageEvent e) {
            System.out.println("Msg> " + e.getMessage());
        }

        // Called when the game sends out an error message during the battle
        public void onBattleError(BattleErrorEvent e) {
            System.out.println("Err> " + e.getError());
        }
    }

}
