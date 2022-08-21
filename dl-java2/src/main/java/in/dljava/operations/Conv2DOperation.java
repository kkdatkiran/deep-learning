package in.dljava.operations;

import java.util.ArrayList;

import in.dljava.data.DoubleData;
import in.dljava.data.Shape;
import in.dljava.functions.initializers.InitializerFunction;

public class Conv2DOperation extends ParameterOperation {

	private int paramSize;
	private int paramPad;

	public Conv2DOperation(DoubleData w) {
		super(w);
		this.paramSize = w.getShape().dimensions()[2];
		this.paramPad = this.paramSize / 2;
	}

	private DoubleData pad1D(DoubleData input) {

		var z = (DoubleData) InitializerFunction.ZEROS.make(Double.class).initalize(new Shape(1, this.paramPad));
		return z.concatenate(this.input, null).concatenate(z, null);
	}

	private DoubleData pad1DBatch(DoubleData input) {

		var dims = input.getShape().dimensions();
		var out = this.pad1D(input.subDataNth(0));
		for (int i = 1; i < dims[0]; i++) {
			out = out.concatenate(input.subDataNth(i), 0);
		}

		return out;
	}

	private DoubleData pad2DOBS(DoubleData input) {

		var inpPad = this.pad1DBatch(input);

		var other = (DoubleData) InitializerFunction.ZEROS.make(Double.class)
				.initalize(new Shape(this.paramPad, input.getShape().dimensions()[0] + this.paramPad * 2));
		return other.concatenate(inpPad, null).concatenate(other, null);
	}

	private DoubleData pad2DChannel(DoubleData input) {

		var dims = input.getShape().dimensions();

		var out = this.pad2DOBS(input.subDataNth(0));

		for (int i = 1; i < dims[0]; i++) {
			out = out.concatenate(input.subDataNth(i), 0);
		}

		return out;
	}

	private DoubleData getImagePatches(DoubleData input) {

		var dims = input.getShape().dimensions();

		var imgsBatchPad = this.pad2DChannel(input.subDataNth(0));
		for (int i = 1; i < dims[0]; i++) {
			imgsBatchPad = imgsBatchPad.concatenate(input.subDataNth(i), 0);
		}

		var patches = new ArrayList<DoubleData>();

		var imgHeight = imgsBatchPad.getShape().dimensions()[2];
		for (int h = 0; h < imgHeight - this.paramSize + 1; h++) {
			for (int w = 0; w < imgHeight - this.paramSize + 1; w++) {
//TODO: I need to work with dimensions here
			}
		}

		return new DoubleData(new Shape(1), new double[] { 1 });
	}

	@Override
	public DoubleData output(boolean inference) {

		var dims = this.input.getShape().dimensions();

		var batchSize = dims[0];
		var imgHeight = dims[2];
		var imgSize = dims[2] * dims[3];

		var paramDims = this.param.getShape().dimensions();
		var patchSize = paramDims[0] * paramDims[2] * paramDims[3];

		DoubleData patches = this.getImagePatches(this.input);

		var patchesReshaped = patches.transpose(1, 0, 2, 3, 4);
		patchesReshaped = patchesReshaped
				.reShape(new Shape(batchSize, imgSize, patchesReshaped.getShape().total() / (batchSize * imgSize)));

		var paramReshaped = this.param.transpose(0, 2, 3, 1);
		paramReshaped = paramReshaped.reShape(new Shape(patchSize, paramReshaped.getShape().total() / patchSize));

		var multiplied = patchesReshaped.matrixMultiply(paramReshaped);

		return multiplied.reShape(new Shape(batchSize, imgHeight, imgHeight,
				multiplied.getShape().total() / (batchSize * imgHeight * imgHeight))).transpose(0, 3, 1, 2);
	}

	@Override
	public DoubleData inputGradient(DoubleData outGradient) {

		int[] inpShape = this.input.getShape().dimensions();
		int batchSize = inpShape[0];
		int imgSize = inpShape[2] * inpShape[3];
		int imgHeight = inpShape[2];

		DoubleData outputPatches = this.getImagePatches(outGradient).transpose(1, 0, 2, 3, 4);

		outputPatches = outputPatches
				.reShape(new Shape(batchSize * imgSize, outputPatches.getShape().total() / (batchSize * imgSize)));

		DoubleData paramReshaped = this.param.newReShape(new Shape());
		paramReshaped = paramReshaped.transpose(1, 0);

		return outputPatches.matrixMultiply(paramReshaped)
				.reShape(batchSize, imgHeight, imgHeight, this.param.getShape().dimensions()[0]).transpose(0, 3, 1, 2);
	}

	@Override
	public DoubleData parameterGradient(DoubleData outGradient) {

		int[] inpShape = this.input.getShape().dimensions();
		int[] paramShape = this.param.getShape().dimensions();

		int batchSize = inpShape[0];
		int imgSize = inpShape[2] * inpShape[3];
		int inChannels = paramShape[0];
		int outChannels = paramShape[1];

		DoubleData inPatchesReshape = this.getImagePatches(this.input);
		inPatchesReshape.reShape(batchSize * imgSize, inPatchesReshape.getShape().total() / (batchSize * imgSize))
				.transpose(1, 0);

		DoubleData outGradReshape = outGradient.transpose(0, 2, 3, 1);
		outGradReshape = outGradReshape.reShape((batchSize * imgSize),
				outGradReshape.getShape().total() / (batchSize * imgSize));

		return inPatchesReshape.matrixMultiply(outGradReshape)
				.reShape(inChannels, this.paramSize, this.paramSize, outChannels).transpose(0, 3, 1, 2);
	}

	public int getParamSize() {
		return paramSize;
	}

	public int getParamPad() {
		return paramPad;
	}

	@Override
	public Operation deepCopy() {
		
		Conv2DOperation conv = new Conv2DOperation(this.param.deepCopy());
		conv.input = this.input.deepCopy();
		conv.out = this.out.deepCopy();
		conv.inpGradient = this.inpGradient.deepCopy();
		conv.paramGrad = this.paramGrad.deepCopy();
		conv.paramPad = this.paramPad;
		conv.paramSize = this.paramSize;
		
		return conv;
	}

}
