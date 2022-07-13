package in.dljava.models;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;

import in.dljava.data.Data;
import in.dljava.data.DataException;
import in.dljava.data.DoubleData;
import in.dljava.data.Shape;
import in.dljava.functions.loss.Loss;
import in.dljava.functions.metrics.MetricsFunction;
import in.dljava.functions.optimizer.OptimizerFunction;
import in.dljava.layers.Input;
import in.dljava.layers.Layer;
import in.dljava.util.ArrayUtil;
import in.dljava.util.Tuple2;
import in.dljava.util.Tuples;

public class Sequential {

	private LinkedList<Layer> layers = new LinkedList<>();

	private OptimizerFunction optimizer;
	private Loss loss;
	private List<MetricsFunction> metrics;

	private List<Data> x = new ArrayList<>();
	private List<Data> y = new ArrayList<>();

	private boolean isCompiled = false;

	public void addLayer(Layer layer) {
		this.layers.add(layer);
	}

	public void compile(OptimizerFunction optimizer, Loss loss, List<MetricsFunction> metrics) {

		this.loss = loss;
		this.optimizer = optimizer;
		this.metrics = metrics;

		Layer prev = this.layers.get(0);
		prev.compile("input", null);

		for (int i = 1; i < layers.size(); i++) {
			Layer current = this.layers.get(i);
			current.compile("" + i, prev);
			prev = current;
		}

		this.isCompiled = true;
	}

	public String summary() {

		if (!this.isCompiled) {
			return "Please compile to show the report.";
		}

		Tuple2<Integer, Integer> totals = layers.stream().map(Layer::parameters)
				.reduce((a, b) -> Tuples.of(a.t1() + b.t1(), a.t2() + b.t2())).orElse(Tuples.of(0, 0));

		return "_______________________________________________________________________\n"
				+ "Type (layer)                  Output Shape                  Param #    \n"
				+ "=======================================================================\n"
				+ layers.stream().map(Layer::summary).collect(Collectors.joining("\n\n", "", "\n"))
				+ "=======================================================================\n\n" + "Total params: "
				+ (totals.t1() + totals.t2()) + "\n" + "Trainable params: " + (totals.t1()) + "\n"
				+ "Non-trainable params: " + (totals.t2()) + "\n";
	}

	public void fit(Data x, Data y) {

		this.fit(x, y, 0, 1, true);
	}

	public void fit(Data x, Data y, int batchSize, int epochs, boolean verbose) {

		if (this.layers.isEmpty()) {
			throw new DataException("No layers to process");
		}

		if (!(this.layers.get(0) instanceof Input)) {
			throw new DataException("First layer has to be an Input layer");
		}

		this.x.add(x);
		this.y.add(y);

		if (batchSize <= 0)
			batchSize = 32;

		if (batchSize > x.getShape().dimensions()[0])
			batchSize = x.getShape().dimensions()[0];

		int iterationSize = x.getShape().dimensions()[1];
		for (int i = 2; i < x.getShape().dimensions().length; i++)
			iterationSize *= x.getShape().dimensions()[i];

		Shape eachOutputShape = y.getShape().oneOfHigherDimension();
		double[] metricsCount = new double[this.metrics.size()];
		String[] metricsNames = new String[metricsCount.length];

		for (int i = 0; i < metricsCount.length; i++) {
			metricsNames[i] = this.metrics.get(i).toString();
		}

		for (int e = 0; e < epochs; e++) {

			if (verbose)
				System.out.println("Epoch : " + e);

			for (int di = 0; di < x.getShape().dimensions()[0]; di += batchSize) {

				DoubleData batchLoss = new DoubleData(new Shape(this.layers.getLast().getUnits()),
						new double[this.layers.getLast().getUnits()]);
				Arrays.fill(metricsCount, 0d);
				for (int b = di; b < di + batchSize; b++) {

					int from = (b) * iterationSize;
					int to = from + iterationSize;
					this.layers.get(0).feedForward((DoubleData) x.subData(new Shape(1, iterationSize), from, to));
					for (int i = 1; i < this.layers.size(); i++) {
						this.layers.get(i).feedForward(this.layers.get(i - 1).getOutput());
					}

					int fromSubData = (b) * eachOutputShape.total();
					int toSubData = fromSubData + eachOutputShape.total();
					DoubleData out = this.layers.get(this.layers.size() - 1).getOutput();
					DoubleData exp = (DoubleData) ((DoubleData) y).subData(eachOutputShape, fromSubData, toSubData);
					batchLoss = batchLoss.add(loss.compute(out, exp));
					for (int i = 0; i < metricsCount.length; i++) {
						metricsCount[i] += this.metrics.get(i).compute(out, exp);
					}
				}

				batchLoss = batchLoss.divide(batchSize);
				for (int i = 0; i < metricsCount.length; i++) {
					metricsCount[i] /= batchSize;
				}

				if (verbose) {
					System.out.println("Batch : " + (di / batchSize) + " - Loss : " + batchLoss + ", Metrics : "
							+ Arrays.toString(ArrayUtil.zipToString(metricsNames, metricsCount)));
				}

				this.layers.get(this.layers.size() - 1).updateWeights(batchLoss, optimizer);

				for (int i = this.layers.size() - 2; i > 0; i--) {
					this.layers.get(i).updateWeights(this.layers.get(i + 1).getErrors(), optimizer);
				}
			}

			if (verbose)
				System.out.println();
		}

	}

	public Data predict(Data predictX) {

		return null;
	}

	public void print() {

		this.layers.stream().forEach(Layer::print);
	}
}
