package in.dljava.models;

import java.util.ArrayList;
import java.util.List;

import in.dljava.functions.loss.LossFunction;
import in.dljava.functions.metrics.MetricsFunction;
import in.dljava.functions.optimizer.OptimizerFunction;
import in.dljava.layers.Layer;

public class Sequential {

	private List<Layer> layers = new ArrayList<>();
	
	private OptimizerFunction optimizer;
	private LossFunction loss;
	private List<MetricsFunction> metrics;
	
	
	public void addLayer(Layer layer) {
		this.layers.add(layer);
	}

	public void compile(OptimizerFunction optimizer, LossFunction loss, List<MetricsFunction> metrics) {
		
		this.loss = loss;
		this.optimizer = optimizer;
		this.metrics = metrics;
	}

	public String summary() {
		
		
	}
	
	public void fit() {
		
	}
	
	public void predict() {
		
	}
}
