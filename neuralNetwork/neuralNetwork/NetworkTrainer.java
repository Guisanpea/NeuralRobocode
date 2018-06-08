package neuralNetwork;

import org.encog.ml.data.MLDataSet;
import org.encog.neural.networks.BasicNetwork;

import static java.util.Objects.isNull;
import static java.util.Objects.nonNull;

class NetworkTrainer {

	static int bestEpoch = 0;
	
	private final int MINIMUM_EPOCHS;
	private final int EXTRA_EPOCHS;
	
	private BasicNetwork bestNetwork;
	
	private int totalEpochs;
	private int epochsN;
	private double minimumError;
	
	private MLDataSet  validationSet;
	
	NetworkTrainer(MLDataSet validationSet, BasicNetwork network, int minimum_epochs, int extra_epochs){
		this.validationSet = validationSet;
		this.MINIMUM_EPOCHS = minimum_epochs;
		this.EXTRA_EPOCHS = extra_epochs;
		this.bestNetwork = network;

		totalEpochs = 0;
	}
	
	
	BasicNetwork getBestNetwork(){
		return bestNetwork;
	}

	boolean hasFinished(BasicNetwork network){
		totalEpochs++;
		double error = network.calculateError(validationSet);
		
		if(isNull(minimumError) || error < minimumError){
			bestEpoch = totalEpochs;
			bestNetwork = (BasicNetwork) network.clone();
			minimumError = error;
			epochsN = 0;
		}else{
			epochsN++;
		}
			return (epochsN > EXTRA_EPOCHS && totalEpochs > MINIMUM_EPOCHS);
	}	
}
