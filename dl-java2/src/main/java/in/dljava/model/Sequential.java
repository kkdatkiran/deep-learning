package in.dljava.model;

import java.util.LinkedList;

import in.dljava.layer.Layer;

public class Sequential {
	
	private LinkedList<Layer> layers = new LinkedList<>();

	public void addLayer(Layer layer) {
		
		this.layers.add(layer);
	}

}
