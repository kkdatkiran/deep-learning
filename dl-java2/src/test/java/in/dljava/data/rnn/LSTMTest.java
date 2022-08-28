package in.dljava.data.rnn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import in.dljava.recurrent.CustomLSTMConf;

class LSTMTest {

	@Test
	void test() throws IOException {
		final int batchSize = 128;

		var mnistTrain = new MnistDataSetIterator(batchSize, true, 123);
		var mnistTest = new MnistDataSetIterator(batchSize, false, 123);

		MultiLayerConfiguration conf = (new NeuralNetConfiguration.Builder()).updater(new Adam(0.0001)).list()
				.layer(new CustomLSTMConf.Builder().nIn(28 * 28).nOut(128).activation(Activation.RELU).build())
				.layer(new CustomLSTMConf.Builder().nIn(128).nOut(128).activation(Activation.RELU).build())
				.layer(new DenseLayer.Builder().nIn(128).nOut(32).activation(Activation.RELU).build())
				.layer(new OutputLayer.Builder(LossFunction.RECONSTRUCTION_CROSSENTROPY).nIn(256).nOut(10)
						.activation(Activation.SOFTMAX).build())
				.setInputType(InputType.recurrent(28 * 28)).build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener());

		List<DataSet> trainBatch = new ArrayList<>();
		List<DataSet> testBatch = new ArrayList<>();

		System.out.println("Reshaping Training Data : ... \n");
		while (mnistTrain.hasNext()) {

			DataSet ds = mnistTrain.next();
			var f = ds.getFeatures();
			var s = f.shape();

			trainBatch.add(new DataSet(f.reshape(new int[] { (int) s[0], 28 * 28, 1 }), ds.getLabels()));
		}

		System.out.println("Reshaping Test Data : ... \n");
		while (mnistTest.hasNext()) {

			DataSet ds = mnistTest.next();
			var f = ds.getFeatures();
			var s = f.shape();
			testBatch.add(new DataSet(f.reshape(new int[] { (int) s[0], 28 * 28, 1 }), ds.getLabels()));
		}

		ListDataSetIterator<DataSet> trainBatchIterator = new ListDataSetIterator<>(trainBatch);
		ListDataSetIterator<DataSet> testBatchIterator = new ListDataSetIterator<>(testBatch);

		System.out.println();
		long start = System.currentTimeMillis();
		for (int epoch = 0; epoch < 3; epoch++) {

			System.out.println("Epoch : " + (epoch + 1));
			net.fit(trainBatchIterator);

			net.rnnClearPreviousState();
		}
		System.out.println("\n" + ((System.currentTimeMillis() - start) / (60d * 1000)) + " minutes");
		System.out.println("");

		var eval = net.evaluate(testBatchIterator);

		System.out.print(eval.stats());
	}
}
