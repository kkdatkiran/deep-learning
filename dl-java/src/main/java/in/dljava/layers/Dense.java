package in.dljava.layers;

import in.dljava.data.DoubleData;
import in.dljava.data.Shape;
import in.dljava.functions.initializers.Initializer;
import in.dljava.functions.initializers.InitializerFunction;
import in.dljava.functions.optimizer.OptimizerFunction;
import in.dljava.funtions.activation.Activation;
import in.dljava.funtions.activation.ActivationFunction;
import in.dljava.util.StringUtil;
import in.dljava.util.Tuple2;
import in.dljava.util.Tuples;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.experimental.Accessors;

@Data
@Accessors(chain = true)
@AllArgsConstructor
public class Dense implements Layer {

	private int units;
	private Activation activation = ActivationFunction.LINEAR.make();
	private boolean useBias = false;
	private Initializer kernelInitializer;
	private Initializer biasInitializer;

	private DoubleData w;
	private DoubleData b;
	private DoubleData o;
	private DoubleData z;
	private DoubleData bDelta;
	private DoubleData wDelta;
	private int prevLayerUnits;
	private String name;

	public Dense(int units, Initializer kernal, Initializer bias) {

		this.units = units;

		this.kernelInitializer = kernal;
		this.biasInitializer = bias;
	}

	public Dense(int units) {
		this(units, InitializerFunction.GLOROT_UNIFORM.make(Double.class),
				InitializerFunction.ZEROS.make(Double.class));
	}

	@Override
	public void compile(String name, Layer previous) {

		this.name = name;

		b = (DoubleData) biasInitializer.initalize(new Shape(1, this.units));
		bDelta = new DoubleData(b.getShape(), new double[b.getShape().total()]);

		this.prevLayerUnits = previous.getUnits();
		w = (DoubleData) kernelInitializer.initalize(new Shape(this.prevLayerUnits, this.units));
		wDelta = new DoubleData(w.getShape(), new double[w.getShape().total()]);
	}

	@Override
	public String summary() {
		return String.format("%30s%30s%d", StringUtil.padEnding("Dense (" + this.name + ")", 30),
				StringUtil.padEnding(new Shape(this.units).toString(), 30),
				this.w.getShape().total() + this.b.getShape().total());
	}

	@Override
	public Tuple2<Integer, Integer> parameters() {
		return Tuples.of(this.w.getShape().total() + this.b.getShape().total(), 0);
	}

	@Override
	public DoubleData feedForward(DoubleData prevLayerData) {
		this.z = prevLayerData.matrixMultiply(w).add(b);
		this.o = this.z.deepCopy();
		this.activation.apply(this.o.getData());		
		return this.o;
	}
	
	@Override
	public void backwardPropagation(DoubleData exp) {
		
		DoubleData error = exp.subtract(this.o);
	}

	@Override
	public DoubleData getOutput() {

		return this.o;
	}

	@Override
	public void print() {

		System.out.println(
				"Dense (" + this.name + ") - Parameters : " + (this.w.getShape().total() + this.b.getShape().total()));

		System.out.println("Weights : ");
		this.w.print();

		System.out.println("Bias : ");
		this.b.print();
	}

	@Override
	public void updateWeights(DoubleData doubleData, OptimizerFunction optimizer) {
		
		
	}

	@Override
	public DoubleData getErrors() {

		return new DoubleData(new Shape(1), new double[] { 1d });
	}
}
