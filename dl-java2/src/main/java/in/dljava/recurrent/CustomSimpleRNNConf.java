package in.dljava.recurrent;

import java.util.Collection;
import java.util.Map;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseRecurrentLayer;
import org.deeplearning4j.nn.conf.layers.InputTypeUtil;
import org.deeplearning4j.nn.conf.layers.LayerValidation;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

public class CustomSimpleRNNConf extends BaseRecurrentLayer {

	private static final long serialVersionUID = 1L;
	private boolean hasLayerNorm = false;
	protected IWeightInit weightInitFnRecurrent;
	protected RNNFormat rnnDataFormat;

	protected CustomSimpleRNNConf(Builder builder) {
		super(builder);
		this.hasLayerNorm = builder.hasLayerNorm;

		this.rnnDataFormat = builder.rnnDataFormat;
	}

	@Override
	public InputType getOutputType(int layerIndex, InputType inputType) {
		if (inputType == null || inputType.getType() != InputType.Type.RNN) {
			throw new IllegalStateException(
					"Invalid input for RNN layer (layer index = " + layerIndex + ", layer name = \"" + getLayerName()
							+ "\"): expect RNN input type with size > 0. Got: " + inputType);
		}

		InputType.InputTypeRecurrent itr = (InputType.InputTypeRecurrent) inputType;

		return InputType.recurrent(nOut, itr.getTimeSeriesLength(), itr.getFormat());
	}

	@Override
	public void setNIn(InputType inputType, boolean override) {
		if (inputType == null || inputType.getType() != InputType.Type.RNN) {
			throw new IllegalStateException("Invalid input for RNN layer (layer name = \"" + getLayerName()
					+ "\"): expect RNN input type with size > 0. Got: " + inputType);
		}

		InputType.InputTypeRecurrent r = (InputType.InputTypeRecurrent) inputType;
		if (nIn <= 0 || override) {
			this.nIn = r.getSize();
		}

		if (rnnDataFormat == null || override)
			this.rnnDataFormat = r.getFormat();
	}

	@Override
	public InputPreProcessor getPreProcessorForInputType(InputType inputType) {
		return InputTypeUtil.getPreprocessorForInputTypeRnnLayers(inputType, rnnDataFormat, getLayerName());
	}

	@Override
	public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
			int layerIndex, INDArray layerParamsView, boolean initializeParams, DataType networkDataType) {
		LayerValidation.assertNInNOutSet("CustomSimpleRnn", getLayerName(), layerIndex, getNIn(), getNOut());

		CustomSimpleRNN ret = new CustomSimpleRNN(conf, networkDataType);
		ret.setListeners(trainingListeners);
		ret.setIndex(layerIndex);
		ret.setParamsViewArray(layerParamsView);
		Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
		ret.setParamTable(paramTable);
		ret.setConf(conf);
		return ret;
	}

	@Override
	public ParamInitializer initializer() {
		return CustomSimpleRNNParamInitializer.getInstance();
	}

	@Override
	public LayerMemoryReport getMemoryReport(InputType inputType) {
		return null;
	}

	public boolean hasLayerNorm() {
		return hasLayerNorm;
	}

	public static class Builder extends BaseRecurrentLayer.Builder<Builder> {

		@SuppressWarnings("unchecked")
		@Override
		public CustomSimpleRNNConf build() {
			return new CustomSimpleRNNConf(this);
		}

		private boolean hasLayerNorm = false;
		private RNNFormat rnnDataFormat = RNNFormat.NCW;

		public Builder hasLayerNorm(boolean hasLayerNorm) {
			this.hasLayerNorm = hasLayerNorm;
			return this;
		}

		public boolean getHasLayerNorm() {
			return this.hasLayerNorm;
		}

		public Builder setHasLayerNorm(boolean hasLayerNorm) {
			this.hasLayerNorm = hasLayerNorm;
			return this;
		}
	}
}
