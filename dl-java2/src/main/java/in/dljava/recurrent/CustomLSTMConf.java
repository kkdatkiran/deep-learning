package in.dljava.recurrent;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseRecurrentLayer;
import org.deeplearning4j.nn.conf.layers.LayerValidation;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSigmoid;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import lombok.NoArgsConstructor;

public class CustomLSTMConf extends BaseRecurrentLayer {

	private static final long serialVersionUID = -8732395797180485909L;

	protected boolean helperAllowFallback = true;

	private CustomLSTMConf(Builder builder) {
		super(builder);
		this.helperAllowFallback = builder.helperAllowFallback;
		initializeConstraints(builder);
	}

	@Override
	protected void initializeConstraints(org.deeplearning4j.nn.conf.layers.Layer.Builder<?> builder) {
		super.initializeConstraints(builder);
		if (((Builder) builder).getRecurrentConstraints() != null) {
			if (constraints == null) {
				constraints = new ArrayList<>();
			}
			for (LayerConstraint c : ((Builder) builder).getRecurrentConstraints()) {
				LayerConstraint c2 = c.clone();
				c2.setParams(Collections.singleton(CustomLSTMParamInitializer.RECURRENT_WEIGHT_KEY));
				constraints.add(c2);
			}
		}
	}

	@Override
	public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> trainingListeners,
			int layerIndex, INDArray layerParamsView, boolean initializeParams, DataType networkDataType) {
		LayerValidation.assertNInNOutSet("CustomLSTM", getLayerName(), layerIndex, getNIn(), getNOut());
		CustomLSTM ret = new CustomLSTM(conf, networkDataType);
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
		return CustomLSTMParamInitializer.getInstance();
	}

	@Override
	public LayerMemoryReport getMemoryReport(InputType inputType) {
		return null;
	}

	@NoArgsConstructor
	public static class Builder extends BaseRecurrentLayer.Builder<Builder> {

		@SuppressWarnings("unchecked")
		public CustomLSTMConf build() {
			return new CustomLSTMConf(this);
		}

		protected boolean helperAllowFallback = true;
	}

	public IActivation getGateActivationFn() {
		return new ActivationSigmoid();
	}

	public boolean isHelperAllowFallback() {
		return true;
	}

	public double getForgetGateBiasInit() {
		return 1;
	}

}
