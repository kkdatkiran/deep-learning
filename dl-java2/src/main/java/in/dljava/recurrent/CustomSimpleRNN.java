package in.dljava.recurrent;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.recurrent.BaseRecurrentLayer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.common.base.Preconditions;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.primitives.Quad;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastCopyOp;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.api.ops.impl.transforms.custom.LayerNorm;
import org.nd4j.linalg.api.ops.impl.transforms.custom.LayerNormBp;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;

public class CustomSimpleRNN  extends BaseRecurrentLayer<CustomSimpleRNNConf> {
	
    private static final long serialVersionUID = 8558038949873383218L;
    
	public static final String STATE_KEY_PREV_ACTIVATION = "prevAct";


    public CustomSimpleRNN(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
    }

    @Override
    public INDArray rnnTimeStep(INDArray input, LayerWorkspaceMgr workspaceMgr) {
        setInput(input, workspaceMgr);
        INDArray last = stateMap.get(STATE_KEY_PREV_ACTIVATION);
        INDArray out = activateHelper(last, false, false, workspaceMgr).getFirst();
        try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()){
            stateMap.put(STATE_KEY_PREV_ACTIVATION, out.get(all(), all(), point(out.size(2)-1)).dup());
        }
        return out;
    }

    @Override
    public INDArray rnnActivateUsingStoredState(INDArray input, boolean training, boolean storeLastForTBPTT, LayerWorkspaceMgr workspaceMgr) {
        setInput(input, workspaceMgr);
        INDArray last = tBpttStateMap.get(STATE_KEY_PREV_ACTIVATION);
        INDArray out = activateHelper(last, training, false, workspaceMgr).getFirst();
        if(storeLastForTBPTT){
            try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()){
                tBpttStateMap.put(STATE_KEY_PREV_ACTIVATION, out.get(all(), all(), point(out.size(2)-1)).dup());
            }
        }
        return out;
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        return tbpttBackpropGradient(epsilon, -1, workspaceMgr);
    }

    @SuppressWarnings("resource")
	@Override
    public Pair<Gradient, INDArray> tbpttBackpropGradient(INDArray epsilon, int tbpttBackLength, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        if(epsilon.ordering() != 'f' || !Shape.hasDefaultStridesForShape(epsilon))
            epsilon = epsilon.dup('f');

        var nOut = layerConf().getNOut();

        INDArray input = this.input.castTo(dataType); 
        input = permuteIfNWC(input);


        Quad<INDArray,INDArray, INDArray, INDArray> p = activateHelper(null, true, true, workspaceMgr);

        INDArray w = getParamWithNoise(CustomSimpleRNNParamInitializer.WEIGHT_KEY, true, workspaceMgr);
        INDArray rw = getParamWithNoise(CustomSimpleRNNParamInitializer.RECURRENT_WEIGHT_KEY, true, workspaceMgr);
        INDArray b = getParamWithNoise(CustomSimpleRNNParamInitializer.BIAS_KEY, true, workspaceMgr);
        INDArray g = (hasLayerNorm() ? getParamWithNoise(CustomSimpleRNNParamInitializer.GAIN_KEY, true, workspaceMgr) : null);
        INDArray gx = (g != null ? g.get(interval(0, 0, true), interval(0, nOut)) : null);
        INDArray gr = (g != null ? g.get(interval(0, 0, true), interval(nOut, nOut * 2)) : null);

        INDArray wg = gradientViews.get(CustomSimpleRNNParamInitializer.WEIGHT_KEY);
        INDArray rwg = gradientViews.get(CustomSimpleRNNParamInitializer.RECURRENT_WEIGHT_KEY);
        INDArray bg = gradientViews.get(CustomSimpleRNNParamInitializer.BIAS_KEY);
        INDArray gg = (hasLayerNorm() ? gradientViews.get(CustomSimpleRNNParamInitializer.GAIN_KEY) : null);
        INDArray gxg = (gg != null ? gg.get(interval(0, 0, true), interval(0, nOut)) : null);
        INDArray grg = (gg != null ? gg.get(interval(0, 0, true), interval(nOut, nOut * 2)) : null);

        gradientsFlattened.assign(0);

        IActivation a = layerConf().getActivationFn();

        var tsLength = input.size(2);

        INDArray epsOut = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, input.dataType(), input.shape(), 'f');

        INDArray dldzNext = null;
        long end;
        if(tbpttBackLength > 0){
            end = Math.max(0, tsLength-tbpttBackLength);
        } else {
            end = 0;
        }
        epsilon = permuteIfNWC(epsilon);
        for( long i = tsLength - 1; i >= end; i--) {
            INDArray dldaCurrent = epsilon.get(all(), all(), point(i)).dup();
            INDArray aCurrent = p.getFirst().get(all(), all(), point(i));
            INDArray zCurrent = p.getSecond().get(all(), all(), point(i));
            INDArray nCurrent = (hasLayerNorm() ? p.getThird().get(all(), all(), point(i)) : null);
            INDArray rCurrent = (hasLayerNorm() ? p.getFourth().get(all(), all(), point(i)) : null);
            INDArray inCurrent = input.get(all(), all(), point(i));
            INDArray epsOutCurrent = epsOut.get(all(), all(), point(i));

            if(dldzNext != null){
                Nd4j.gemm(dldzNext, rw, dldaCurrent, false, true, 1.0, 1.0);

                Nd4j.gemm(aCurrent, dldzNext, rwg, true, false, 1.0, 1.0);
            }
            INDArray dldzCurrent = a.backprop(zCurrent.dup(), dldaCurrent).getFirst();

            INDArray maskCol = null;
            if( maskArray != null) {


                maskCol = maskArray.getColumn(i, true).castTo(dataType);
                dldzCurrent.muliColumnVector(maskCol);
            }

            INDArray dldnCurrent;
            if(hasLayerNorm()) {
                dldnCurrent = workspaceMgr.createUninitialized(ArrayType.BP_WORKING_MEM, dldzCurrent.dataType(), dldzCurrent.shape());
                INDArray ggCur = workspaceMgr.createUninitialized(ArrayType.BP_WORKING_MEM, gg.dataType(), gxg.shape());
                INDArray bgCur = workspaceMgr.createUninitialized(ArrayType.BP_WORKING_MEM, bg.dataType(), bg.shape());
                Nd4j.getExecutioner().exec(new LayerNormBp(nCurrent, gx, b, dldzCurrent, dldnCurrent, ggCur, bgCur, true, 1));
                gxg.addi(ggCur);
                bg.addi(bgCur);
            }else{
                dldnCurrent = dldzCurrent;

                bg.addi(dldzCurrent.sum(0));
            }


            Nd4j.gemm(inCurrent, dldnCurrent, wg, true, false, 1.0, 1.0);


            Nd4j.gemm(dldnCurrent, w, epsOutCurrent, false, true, 1.0, 0.0);


            if(hasLayerNorm() && i > end){
                dldzNext = workspaceMgr.createUninitialized(ArrayType.BP_WORKING_MEM, dldzCurrent.dataType(), dldzCurrent.shape());
                INDArray ggCur = workspaceMgr.createUninitialized(ArrayType.BP_WORKING_MEM, gg.dataType(), grg.shape());
                Nd4j.getExecutioner().exec(new LayerNormBp(rCurrent, gr, dldzCurrent, dldzNext, ggCur, true, 1));
                grg.addi(ggCur);
            }else{
                dldzNext = dldzCurrent;
            }

            if( maskArray != null){
                
                epsOutCurrent.muliColumnVector(maskCol);
            }
        }

        weightNoiseParams.clear();

        Gradient grad = new DefaultGradient(gradientsFlattened);
        grad.gradientForVariable().put(CustomSimpleRNNParamInitializer.WEIGHT_KEY, wg);
        grad.gradientForVariable().put(CustomSimpleRNNParamInitializer.RECURRENT_WEIGHT_KEY, rwg);
        grad.gradientForVariable().put(CustomSimpleRNNParamInitializer.BIAS_KEY, bg);
        if(hasLayerNorm()){
            grad.gradientForVariable().put(CustomSimpleRNNParamInitializer.GAIN_KEY, gg);
        }

        epsOut = backpropDropOutIfPresent(epsOut);
        epsOut = permuteIfNWC(epsOut);
        return new Pair<>(grad, epsOut);
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr){
        return activateHelper(null, training, false, workspaceMgr).getFirst();
    }

    @SuppressWarnings("resource")
	private Quad<INDArray,INDArray,INDArray, INDArray> activateHelper(INDArray prevStepOut, boolean training, boolean forBackprop, LayerWorkspaceMgr workspaceMgr){
        assertInputSet(false);
        Preconditions.checkState(input.rank() == 3,
                "3D input expected to RNN layer expected, got " + input.rank());
        Preconditions.checkState(prevStepOut == null || prevStepOut.size(0) == input.size(0),
                "Invalid RNN previous state (last time step activations/initialization): rnnTimeStep with different minibatch size, or forgot to call rnnClearPreviousState between batches?" +
                        " Previous step output = [batch, nIn] = %ndShape, current input = [batch, nIn, seqLength] = %ndShape", prevStepOut, input);

        applyDropOutIfNecessary(training, workspaceMgr);

        INDArray input = this.input.castTo(dataType);    
        input = permuteIfNWC(input);
        var m = input.size(0);
        var tsLength = input.size(2);
        var nOut = layerConf().getNOut();

        INDArray w = getParamWithNoise(CustomSimpleRNNParamInitializer.WEIGHT_KEY, training, workspaceMgr);
        INDArray rw = getParamWithNoise(CustomSimpleRNNParamInitializer.RECURRENT_WEIGHT_KEY, training, workspaceMgr);
        INDArray b = getParamWithNoise(CustomSimpleRNNParamInitializer.BIAS_KEY, training, workspaceMgr);
        INDArray g = (hasLayerNorm() ? getParamWithNoise(CustomSimpleRNNParamInitializer.GAIN_KEY, training, workspaceMgr) : null);
        INDArray gx = (g != null ? g.get(interval(0, 0, true), interval(0, nOut)) : null);
        INDArray gr = (g != null ? g.get(interval(0, 0, true), interval(nOut, nOut * 2)) : null);

        INDArray out = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, w.dataType(), new long[]{m, nOut, tsLength}, 'f');
        INDArray outZ = (forBackprop ? workspaceMgr.createUninitialized(ArrayType.BP_WORKING_MEM, w.dataType(), out.shape()) : null);
        INDArray outPreNorm = (forBackprop && hasLayerNorm() ? workspaceMgr.createUninitialized(ArrayType.BP_WORKING_MEM, w.dataType(), out.shape(), 'f') : null);
        INDArray recPreNorm = (forBackprop && hasLayerNorm() ? workspaceMgr.createUninitialized(ArrayType.BP_WORKING_MEM, w.dataType(), out.shape(), 'f') : null);

        if(input.ordering() != 'f' || Shape.strideDescendingCAscendingF(input))
            input = workspaceMgr.dup(ArrayType.ACTIVATIONS, input, 'f');


        if(!hasLayerNorm()) {
            Nd4j.getExecutioner().exec(new BroadcastCopyOp(out, b, out, 1));
        }

        IActivation a = layerConf().getActivationFn();

        for( int i = 0; i < tsLength; i++) {

            INDArray currOut = out.get(all(), all(), point(i)); 
            INDArray currIn = input.get(all(), all(), point(i));
            if(hasLayerNorm()){
                INDArray currOutPreNorm = (forBackprop ? outPreNorm : out).get(all(), all(), point(i));
                Nd4j.gemm(currIn, w, currOutPreNorm, false, false, 1.0, 0.0);
                Nd4j.getExecutioner().exec(new LayerNorm(currOutPreNorm, gx, b, currOut, true, 1));
            }else{
                Nd4j.gemm(currIn, w, currOut, false, false, 1.0, 1.0); 
            }

            if(i > 0 || prevStepOut != null) {
                if(hasLayerNorm()){
                    INDArray currRecPreNorm = forBackprop ? recPreNorm.get(all(), all(), point(i)) : workspaceMgr.createUninitialized(ArrayType.FF_WORKING_MEM, currOut.dataType(), currOut.shape(), 'f');
                    Nd4j.gemm(prevStepOut, rw, currRecPreNorm, false, false, 1.0, 0.0);
                    INDArray recNorm = workspaceMgr.createUninitialized(ArrayType.FF_WORKING_MEM, currOut.dataType(), currOut.shape(), 'f');
                    Nd4j.getExecutioner().exec(new LayerNorm(currRecPreNorm, gr, recNorm, true, 1));
                    currOut.addi(recNorm);
                }else {
                    Nd4j.gemm(prevStepOut, rw, currOut, false, false, 1.0, 1.0); 
                }
            }

            if(forBackprop){
                outZ.get(all(), all(), point(i)).assign(currOut);
            }

            a.getActivation(currOut, training);

            if( maskArray != null){

                INDArray maskCol = maskArray.getColumn(i, true).castTo(dataType);
                currOut.muliColumnVector(maskCol);
            }

            prevStepOut = currOut;
        }

        if(maskArray != null) {

            INDArray mask = maskArray.castTo(dataType);
            Nd4j.getExecutioner().exec(new BroadcastMulOp(out, mask, out, 0, 2));
            if(forBackprop){
                Nd4j.getExecutioner().exec(new BroadcastMulOp(outZ, mask, outZ, 0, 2));
            }
        }
        if (!forBackprop) {
            out = permuteIfNWC(out);
            outZ = permuteIfNWC(outZ);
            outPreNorm = permuteIfNWC(outPreNorm);
            recPreNorm = permuteIfNWC(recPreNorm);
        }
        return new Quad<>(out, outZ, outPreNorm, recPreNorm);
    }

    @Override
    public boolean hasLayerNorm(){
        return layerConf().hasLayerNorm();
    }
}
