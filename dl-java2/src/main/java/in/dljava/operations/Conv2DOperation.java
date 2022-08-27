package in.dljava.operations;

import in.dljava.data.DoubleData;
import in.dljava.data.Range;
import in.dljava.data.Shape;

public class Conv2DOperation extends ParameterOperation {

	private int paramSize;
	private int paramPad;

	public Conv2DOperation(DoubleData w) {
		super(w);
		this.paramSize = w.getShape().dimensions()[2];
		this.paramPad = this.paramSize / 2;
	}

	private DoubleData getImagePatches(DoubleData input) {

		var dims = input.getShape().dimensions();

		DoubleData eachData = input.subDataNth(0);
		var imgsBatchPad = eachData.pad(this.paramPad);
		for (int i = 1; i < dims[0]; i++) {
			imgsBatchPad = imgsBatchPad.concatenate(input.subDataNth(i).pad(this.paramPad), 0);
		}

		dims = imgsBatchPad.getShape().dimensions();
		var newDims = new int[dims.length + 1];
		newDims[0] = dims[dims.length - 1] - this.paramSize + 1;
		newDims[0] *= newDims[0];
		for (int i = 1; i < newDims.length; i++) {
			if (i < newDims.length - 2)
				newDims[i] = dims[i - 1];
			else
				newDims[i] = this.paramSize;
		}

		DoubleData newD = null;

		for (int h = 0; h < dims[2] - this.paramSize + 1; h++) {
			for (int w = 0; w < dims[3] - this.paramSize + 1; w++) {

				DoubleData x = imgsBatchPad.indices(new Range(-1), new Range(-1), new Range(h, h + this.paramSize),
						new Range(w, w + this.paramSize));
				x = x.reShape(x.getShape().increaseOneDimension());
				if (newD == null) {
					newD = x;
				} else {
					newD = newD.concatenate(x, 0);
				}
			}
		}

		return newD;
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

		Shape paramShape = this.param.getShape();

		DoubleData paramReshaped = this.param
				.newReShape(new Shape(paramShape.dimensions(0), paramShape.total() / paramShape.dimensions(0)));
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
		inPatchesReshape = inPatchesReshape
				.reShape(batchSize * imgSize, inPatchesReshape.getShape().total() / (batchSize * imgSize))
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
