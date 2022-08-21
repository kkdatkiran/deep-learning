package in.dljava.model;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.stream.Stream;

import in.dljava.data.DoubleData;
import in.dljava.layer.Layer;
import in.dljava.loss.Loss;
import in.dljava.optimizer.Tuple2;

public class Sequential {

	private List<Layer> layers;
	private Loss loss;
	private int seed;

	public Sequential(List<Layer> layers, Loss loss, Integer seed) {

		this.layers = layers;
		this.loss = loss;
		this.seed = seed == null ? 1 : seed.intValue();
		if (seed != 0) {
			for (Layer layer : this.layers)
				layer.setSeed(this.seed);
		}
	}

	public DoubleData forward(DoubleData batch) {
		var out = batch;
		for (Layer layer : this.layers) {
			out = layer.forward(out, false);
		}

		return out;
	}

	public void backward(DoubleData lossGrad) {

		var grad = lossGrad;
		for (int i = this.layers.size() - 1; i >= 0; i--) {
			grad = this.layers.get(i).backward(grad);
		}
	}

	public double trainBatch(DoubleData xBatch, DoubleData yBatch) {

		var predictions = this.forward(xBatch);
		var lossValue = this.loss.forward(predictions, yBatch);
		this.backward(this.loss.backward());
		return lossValue;
	}

	public Iterator<DoubleData> params() {

		return this.layers.stream().map(Layer::getParams).flatMap(List::stream).iterator();
	}

	public Iterator<DoubleData> paramGrads() {

		return this.layers.stream().map(Layer::getParamGrads).flatMap(List::stream).iterator();
	}

	public Stream<Tuple2<DoubleData, DoubleData>> zipParamsParamGrads() {
		return this.layers.stream().map(l -> {
			List<Tuple2<DoubleData, DoubleData>> ret = new ArrayList<>();
			for (int i = 0; i < l.getParams().size(); i++)
				ret.add(new Tuple2<>(l.getParams().get(i), l.getParamGrads().get(i)));
			return ret;
		}).flatMap(List::stream);
	}

	public List<Layer> getLayers() {
		return this.layers;
	}

	public Sequential deepCopy() {

		return new Sequential(this.layers.stream().map(Layer::deepCopy).toList(), this.loss.deepCopy(), this.seed);
	}

	public Loss getLoss() {
		return this.loss;
	}
}
