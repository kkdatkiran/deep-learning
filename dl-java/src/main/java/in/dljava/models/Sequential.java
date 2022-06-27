package in.dljava.models;

import java.util.ArrayList;
import java.util.List;

import in.dljava.layers.Layer;

public class Sequential {

	private List<Layer> layers = new ArrayList<>();
	
	public void addLayer(Layer layer) {
		this.layers.add(layer);
	}

	public void compile() {
		// TODO Auto-generated method stub
		
	}
	
}
