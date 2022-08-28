package in.dljava.recurrent;

import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.RNNFormat;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.HelperUtils;
import org.deeplearning4j.nn.layers.LayerHelper;
import org.deeplearning4j.nn.layers.recurrent.BaseRecurrentLayer;
import org.deeplearning4j.nn.layers.recurrent.FwdPassReturn;
import org.deeplearning4j.nn.layers.recurrent.LSTMHelper;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.util.TimeSeriesUtils;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

public class CustomLSTM extends BaseRecurrentLayer<CustomLSTMConf> {

	private static final long serialVersionUID = -3021428420063424541L;

	public static final String STATE_KEY_PREV_ACTIVATION = "prevAct";
	public static final String STATE_KEY_PREV_MEMCELL = "prevMem";
	protected transient LSTMHelper helper = null;
	protected transient FwdPassReturn cachedFwdPass;
	public static final String CUDNN_LSTM_CLASS_NAME = "org.deeplearning4j.cuda.recurrent.CudnnLSTMHelper";

	public CustomLSTM(NeuralNetConfiguration conf, DataType dataType) {
		super(conf, dataType);
		initializeHelper();
	}

	void initializeHelper() {
		helper = HelperUtils.createHelper(CUDNN_LSTM_CLASS_NAME, "", LSTMHelper.class, layerConf().getLayerName(),
				dataType);
	}

	@Override
	public Gradient gradient() {
		throw new UnsupportedOperationException(
				"gradient() method for layerwise pretraining: not supported for LSTMs (pretraining not possible) "
						+ layerId());
	}

	public int getHelperCountFail() {
		return this.helperCountFail;
	}

	public void setHelperCountFail(int h) {
		this.helperCountFail = h;
	}

	@Override
	public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
		return backpropGradientHelper(epsilon, false, -1, workspaceMgr);
	}

	@Override
	public Pair<Gradient, INDArray> tbpttBackpropGradient(INDArray epsilon, int tbpttBackwardLength,
			LayerWorkspaceMgr workspaceMgr) {
		return backpropGradientHelper(epsilon, true, tbpttBackwardLength, workspaceMgr);
	}

	private Pair<Gradient, INDArray> backpropGradientHelper(final INDArray epsilon, final boolean truncatedBPTT,
			final int tbpttBackwardLength, LayerWorkspaceMgr workspaceMgr) {
		assertInputSet(true);

		final INDArray inputWeights = getParamWithNoise(CustomLSTMParamInitializer.INPUT_WEIGHT_KEY, true,
				workspaceMgr);
		final INDArray recurrentWeights = getParamWithNoise(CustomLSTMParamInitializer.RECURRENT_WEIGHT_KEY, true,
				workspaceMgr);
		FwdPassReturn fwdPass;
		if (truncatedBPTT) {
			fwdPass = activateHelper(true, stateMap.get(STATE_KEY_PREV_ACTIVATION),
					stateMap.get(STATE_KEY_PREV_MEMCELL), true, workspaceMgr);

			tBpttStateMap.put(STATE_KEY_PREV_ACTIVATION, fwdPass.lastAct.detach());
			tBpttStateMap.put(STATE_KEY_PREV_MEMCELL, fwdPass.lastMemCell.detach());
		} else {
			fwdPass = activateHelper(true, null, null, true, workspaceMgr);
		}
		fwdPass.fwdPassOutput = permuteIfNWC(fwdPass.fwdPassOutput);
		Pair<Gradient, INDArray> p = Helper.backpropGradientHelper(this, this.conf,
				this.layerConf().getGateActivationFn(), permuteIfNWC(this.input), recurrentWeights, inputWeights,
				permuteIfNWC(epsilon), truncatedBPTT, tbpttBackwardLength, fwdPass, true,
				CustomLSTMParamInitializer.INPUT_WEIGHT_KEY, CustomLSTMParamInitializer.RECURRENT_WEIGHT_KEY,
				CustomLSTMParamInitializer.BIAS_KEY, gradientViews, null, false, helper, workspaceMgr,
				layerConf().isHelperAllowFallback());

		weightNoiseParams.clear();
		p.setSecond(permuteIfNWC(backpropDropOutIfPresent(p.getSecond())));
		return p;
	}

	@Override
	public INDArray activate(INDArray input, boolean training, LayerWorkspaceMgr workspaceMgr) {
		setInput(input, workspaceMgr);
		return activateHelper(training, null, null, false, workspaceMgr).fwdPassOutput;
	}

	@Override
	public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
		return activateHelper(training, null, null, false, workspaceMgr).fwdPassOutput;
	}

	private FwdPassReturn activateHelper(final boolean training, final INDArray prevOutputActivations,
			final INDArray prevMemCellState, boolean forBackprop, LayerWorkspaceMgr workspaceMgr) {
		assertInputSet(false);
		Preconditions.checkState(input.rank() == 3, "3D input expected to RNN layer expected, got " + input.rank());

		boolean nwc = TimeSeriesUtils.getFormatFromRnnLayer(layerConf()) == RNNFormat.NWC;

		INDArray origInput = input;
		if (nwc) {
			input = permuteIfNWC(input);
		}

		applyDropOutIfNecessary(training, workspaceMgr);

		cacheMode = CacheMode.NONE;

		if (forBackprop && cachedFwdPass != null) {
			FwdPassReturn ret = cachedFwdPass;
			cachedFwdPass = null;
			return ret;
		}

		final INDArray recurrentWeights = getParamWithNoise(CustomLSTMParamInitializer.RECURRENT_WEIGHT_KEY, training,
				workspaceMgr);
		final INDArray inputWeights = getParamWithNoise(CustomLSTMParamInitializer.INPUT_WEIGHT_KEY, training,
				workspaceMgr);

		final INDArray biases = getParamWithNoise(CustomLSTMParamInitializer.BIAS_KEY, training, workspaceMgr);
		FwdPassReturn fwd = Helper.activateHelper(this, this.conf, this.layerConf().getGateActivationFn(), input,
				recurrentWeights, inputWeights, biases, training, prevOutputActivations, prevMemCellState,
				(training && cacheMode != CacheMode.NONE) || forBackprop, true,
				CustomLSTMParamInitializer.INPUT_WEIGHT_KEY, maskArray, false, helper,
				forBackprop ? cacheMode : CacheMode.NONE, workspaceMgr, layerConf().isHelperAllowFallback());

		fwd.fwdPassOutput = permuteIfNWC(fwd.fwdPassOutput);

		if (training && cacheMode != CacheMode.NONE) {
			cachedFwdPass = fwd;
		}

		if (nwc) {
			input = origInput;
		}

		return fwd;
	}

	@Override
	public Type type() {
		return Type.RECURRENT;
	}

	@Override
	public boolean isPretrainLayer() {
		return false;
	}

	@Override
	public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
			int minibatchSize) {

		return new Pair<>(maskArray, MaskState.Passthrough);
	}

	@Override
	public INDArray rnnTimeStep(INDArray input, LayerWorkspaceMgr workspaceMgr) {
		setInput(input, workspaceMgr);
		FwdPassReturn fwdPass = activateHelper(false, stateMap.get(STATE_KEY_PREV_ACTIVATION),
				stateMap.get(STATE_KEY_PREV_MEMCELL), false, workspaceMgr);
		INDArray outAct = fwdPass.fwdPassOutput;

		stateMap.put(STATE_KEY_PREV_ACTIVATION, fwdPass.lastAct.detach());
		stateMap.put(STATE_KEY_PREV_MEMCELL, fwdPass.lastMemCell.detach());

		return outAct;
	}

	@Override
	public INDArray rnnActivateUsingStoredState(INDArray input, boolean training, boolean storeLastForTBPTT,
			LayerWorkspaceMgr workspaceMgr) {
		setInput(input, workspaceMgr);
		FwdPassReturn fwdPass = activateHelper(training, tBpttStateMap.get(STATE_KEY_PREV_ACTIVATION),
				tBpttStateMap.get(STATE_KEY_PREV_MEMCELL), false, workspaceMgr);
		INDArray outAct = fwdPass.fwdPassOutput;
		if (storeLastForTBPTT) {

			tBpttStateMap.put(STATE_KEY_PREV_ACTIVATION, fwdPass.lastAct.detach());
			tBpttStateMap.put(STATE_KEY_PREV_MEMCELL, fwdPass.lastMemCell.detach());
		}

		return outAct;
	}

	@Override
	public LayerHelper getHelper() {
		return helper;
	}
}