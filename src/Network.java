public class Network {

    protected Layer[] layers;
    protected double[] desiredOutput;
    protected int[] layersSize;
    protected ActivationFunction function;

    public Network(double []i, double[] dOutput, int []lSize, ActivationFunction af) {
        this.function = af;
        this.layersSize = lSize;
        this.layers = new Layer[lSize.length];
        this.desiredOutput = dOutput;
        setLayers(i, 0, af);
        for (int j = 1; j < layers.length; ++j) {
            setLayers(layers[j - 1].layerOutput(), j, af);
        }
    }

    public Network(int inputSize, int []lSize, ActivationFunction af) {
        this.function = af;
        this.layersSize = lSize;
        this.layers = new Layer[lSize.length];
        double []inputs = new double[inputSize];
        setLayers(inputs, 0, function);
        for (int j = 1; j < layers.length; ++j) {
            setLayers(layers[j - 1].layerOutput(), j, function);
        }
    }

    public void setDesiredOutput(double[] desiredOutput) {
        this.desiredOutput = desiredOutput;
    }

    public void setInputs(double[] inputs) {
        layers[0].setInputs(inputs);
        for (int j = 1; j < layers.length; ++j) {
            layers[j].setInputs(layers[j - 1].layerOutput());
        }
    }

    public void setWeights(double [][][]weights) {
        for (int i = 0; i < weights.length; ++i) {
            layers[i].setWeights(weights[i]);
        }
    }

    public double[] networkOutput(double []i) {
        setInputs(i);
        return networkOutput();
    }

    public double[] networkOutput() {
        return layers[layers.length - 1].layerOutput();
    }

    public Layer[] getLayers() {
        return layers;
    }

    private void setLayers(double []i, int index, ActivationFunction af) {
        layers[index] = new Layer(i, layersSize[index], af);
    }
}
