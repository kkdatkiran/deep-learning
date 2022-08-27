package in.dljava.trainer;

import java.util.ArrayList;
import java.util.List;

import in.dljava.data.DoubleData;
import in.dljava.data.Range;
import in.dljava.layer.Layer;
import in.dljava.model.Sequential;
import in.dljava.optimizer.Optimizer;
import in.dljava.optimizer.Tuple2;

public class Trainer {

	private Sequential net;
	private Optimizer optim;
	private double bestLoss;

	public Trainer(Sequential net, Optimizer optim) {

		this.net = net;
		this.optim = optim;
		this.bestLoss = 1e9;

	}

	public List<Tuple2<DoubleData, DoubleData>> generateBatches(DoubleData x, DoubleData y, int size) {

		if (size < 1)
			size = 32;

		assert x.getShape().dimensions()[0] == y.getShape().dimensions()[0];

		var n = x.getShape().dimensions()[0];

		List<Tuple2<DoubleData, DoubleData>> lst = new ArrayList<>();

		for (int i = 0; i < n; i += size) {
			lst.add(new Tuple2<>(x.subDataNth(i), y.subDataNth(i)));
		}

		return lst;
	}

	public void fit(DoubleData xtrain, DoubleData ytrain, DoubleData xtest, DoubleData ytest, int epochs, int evalEvery,
			int batchSize, boolean restart) {
		this.fit(xtrain, ytrain, xtest, ytest, epochs, evalEvery, batchSize, restart, false);
	}

	public void fit(DoubleData xtrain, DoubleData ytrain, DoubleData xtest, DoubleData ytest, int epochs, int evalEvery,
			int batchSize, boolean restart, boolean convTesting) {

		this.optim.setupDecay();

		if (restart) {
			for (Layer layer : this.net.getLayers()) {
				layer.setFirst(true);
			}
			this.bestLoss = 1e9;
		}

		var testBatches = this.generateBatches(xtest, ytest, 1);
		var batches = this.generateBatches(xtrain, ytrain, batchSize);

//		Sequential lastModel = null;
		for (int e = 0; e < epochs; e++) {

//			if ((e + 1) % evalEvery == 0) {
//				lastModel = this.net.deepCopy();
//			}

			for (int ii = 0; ii < batches.size(); ii++) {

				var batch = batches.get(ii);
				this.net.trainBatch(batch.getT1(), batch.getT2(), false);
				this.optim.step(e);

				if (convTesting && (ii + 1) % 10 == 0) {
					var testPreds = this.net.forward(batch.getT1(), true);
					var batchLoss = this.net.getLoss().forward(testPreds, batch.getT2());
					System.out.println("Batch Validation Loss after " + (ii + 1) + " batches : " + batchLoss);
				}
			}

			if ((e + 1) % evalEvery == 0) {

				double loss = 0;
				for (int i = 0; i < testBatches.size(); i++) {
					var testPreds = this.net.forward(testBatches.get(i).getT1(), true);
					loss += this.net.getLoss().forward(testPreds, testBatches.get(i).getT2());
				}
				loss /= xtest.getShape().dimensions()[0];

				System.out.println("Validation loss after " + (e + 1) + " epochs is " + loss);

				if (loss < this.bestLoss) {
					this.bestLoss = loss;
				}
			}

			if (this.optim.getFinalLearningRate() != 0d) {
				this.optim.decayLearningRate();
			}
		}
	}

}
