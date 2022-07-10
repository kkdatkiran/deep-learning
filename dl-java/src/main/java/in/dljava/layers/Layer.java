package in.dljava.layers;

import in.dljava.data.DoubleData;
import in.dljava.util.Tuple2;

public interface Layer {

	public void compile(String name, Layer previous);

	public int getUnits();

	public String summary();

	public Tuple2<Integer, Integer> parameters();

	public DoubleData feedForward(DoubleData prevLayerData);

	public DoubleData getOutput();
	
	public void print();
}