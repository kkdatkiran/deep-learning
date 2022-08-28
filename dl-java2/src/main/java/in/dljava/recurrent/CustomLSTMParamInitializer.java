package in.dljava.recurrent;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import lombok.val;

public class CustomLSTMParamInitializer implements ParamInitializer {

	public static final String RECURRENT_WEIGHT_KEY = "RW";
	public static final String BIAS_KEY = DefaultParamInitializer.BIAS_KEY;
	public static final String INPUT_WEIGHT_KEY = DefaultParamInitializer.WEIGHT_KEY;

	private static final CustomLSTMParamInitializer INSTANCE = new CustomLSTMParamInitializer();

	private static final List<String> LAYER_PARAM_KEYS = Collections
			.unmodifiableList(Arrays.asList(INPUT_WEIGHT_KEY, RECURRENT_WEIGHT_KEY, BIAS_KEY));
	private static final List<String> WEIGHT_KEYS = Collections
			.unmodifiableList(Arrays.asList(INPUT_WEIGHT_KEY, RECURRENT_WEIGHT_KEY));
	private static final List<String> BIAS_KEYS = Collections.unmodifiableList(Collections.singletonList(BIAS_KEY));

	public static CustomLSTMParamInitializer getInstance() {
		return INSTANCE;
	}

	@Override
	public long numParams(NeuralNetConfiguration conf) {
		return numParams(conf.getLayer());
	}

	@Override
	public long numParams(Layer l) {
		CustomLSTMConf layerConf = (CustomLSTMConf) l;

		val nL = layerConf.getNOut();
		val nLast = layerConf.getNIn();

		return nLast * (4 * nL) + nL * (4 * nL) + 4 * nL;
	}

	@Override
	public List<String> paramKeys(Layer layer) {
		return LAYER_PARAM_KEYS;
	}

	@Override
	public List<String> weightKeys(Layer layer) {
		return WEIGHT_KEYS;
	}

	@Override
	public List<String> biasKeys(Layer layer) {
		return BIAS_KEYS;
	}

	@Override
	public boolean isWeightParam(Layer layer, String key) {
		return RECURRENT_WEIGHT_KEY.equals(key) || INPUT_WEIGHT_KEY.equals(key);
	}

	@Override
	public boolean isBiasParam(Layer layer, String key) {
		return BIAS_KEY.equals(key);
	}

	@Override
	public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
		Map<String, INDArray> params = Collections.synchronizedMap(new LinkedHashMap<String, INDArray>());
		CustomLSTMConf layerConf = (CustomLSTMConf) conf.getLayer();
		double forgetGateInit = layerConf.getForgetGateBiasInit();

		val nL = layerConf.getNOut();
		val nLast = layerConf.getNIn();

		conf.addVariable(INPUT_WEIGHT_KEY);
		conf.addVariable(RECURRENT_WEIGHT_KEY);
		conf.addVariable(BIAS_KEY);

		val length = numParams(conf);
		if (paramsView.length() != length)
			throw new IllegalStateException(
					"Expected params view of length " + length + ", got length " + paramsView.length());

		val nParamsIn = nLast * (4 * nL);
		val nParamsRecurrent = nL * (4 * nL);
		val nBias = 4 * nL;
		INDArray inputWeightView = paramsView.get(NDArrayIndex.interval(0, 0, true),
				NDArrayIndex.interval(0, nParamsIn));
		INDArray recurrentWeightView = paramsView.get(NDArrayIndex.interval(0, 0, true),
				NDArrayIndex.interval(nParamsIn, nParamsIn + nParamsRecurrent));
		INDArray biasView = paramsView.get(NDArrayIndex.interval(0, 0, true),
				NDArrayIndex.interval(nParamsIn + nParamsRecurrent, nParamsIn + nParamsRecurrent + nBias));

		if (initializeParams) {
			val fanIn = nL;
			val fanOut = nLast + nL;
			val inputWShape = new long[] { nLast, 4 * nL };
			val recurrentWShape = new long[] { nL, 4 * nL };

			IWeightInit rwInit;
			if (layerConf.getWeightInitFnRecurrent() != null) {
				rwInit = layerConf.getWeightInitFnRecurrent();
			} else {
				rwInit = layerConf.getWeightInitFn();
			}

			params.put(INPUT_WEIGHT_KEY, layerConf.getWeightInitFn().init(fanIn, fanOut, inputWShape,
					IWeightInit.DEFAULT_WEIGHT_INIT_ORDER, inputWeightView));
			params.put(RECURRENT_WEIGHT_KEY, rwInit.init(fanIn, fanOut, recurrentWShape,
					IWeightInit.DEFAULT_WEIGHT_INIT_ORDER, recurrentWeightView));
			biasView.put(new INDArrayIndex[] { NDArrayIndex.interval(0, 0, true), NDArrayIndex.interval(nL, 2 * nL) },
					Nd4j.valueArrayOf(new long[] { 1, nL }, forgetGateInit)); // Order: input, forget, output, input
																				// modulation, i.e., IFOG}

			params.put(BIAS_KEY, biasView);
		} else {
			params.put(INPUT_WEIGHT_KEY, WeightInitUtil.reshapeWeights(new long[] { nLast, 4 * nL }, inputWeightView));
			params.put(RECURRENT_WEIGHT_KEY,
					WeightInitUtil.reshapeWeights(new long[] { nL, 4 * nL }, recurrentWeightView));
			params.put(BIAS_KEY, biasView);
		}

		return params;
	}

	@Override
	public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
		CustomLSTMConf layerConf = (CustomLSTMConf) conf.getLayer();

		val nL = layerConf.getNOut();
		val nLast = layerConf.getNIn();

		val length = numParams(conf);
		if (gradientView.length() != length)
			throw new IllegalStateException(
					"Expected gradient view of length " + length + ", got length " + gradientView.length());

		val nParamsIn = nLast * (4 * nL);
		val nParamsRecurrent = nL * (4 * nL);
		val nBias = 4 * nL;
		INDArray inputWeightGradView = gradientView
				.get(NDArrayIndex.interval(0, 0, true), NDArrayIndex.interval(0, nParamsIn))
				.reshape('f', nLast, 4 * nL);
		INDArray recurrentWeightGradView = gradientView
				.get(NDArrayIndex.interval(0, 0, true), NDArrayIndex.interval(nParamsIn, nParamsIn + nParamsRecurrent))
				.reshape('f', nL, 4 * nL);
		INDArray biasGradView = gradientView.get(NDArrayIndex.interval(0, 0, true),
				NDArrayIndex.interval(nParamsIn + nParamsRecurrent, nParamsIn + nParamsRecurrent + nBias)); // already a
																											// row
																											// vector

		Map<String, INDArray> out = new LinkedHashMap<>();
		out.put(INPUT_WEIGHT_KEY, inputWeightGradView);
		out.put(RECURRENT_WEIGHT_KEY, recurrentWeightGradView);
		out.put(BIAS_KEY, biasGradView);

		return out;
	}
}