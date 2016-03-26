public class Layer {

    private Neuron[] neurons;

    public Layer(double []inputs, int neuronsCount, ActivationFunction af) {
        neurons = new Neuron[neuronsCount];
        for (int j = 0; j < neuronsCount; ++j) {
            neurons[j] = new Neuron(inputs) {
                @Override
                public double activationFunction(double S) {
                    return af.activationFunction(S);
                }

                @Override
                public double functionDerivative(double S) {
                    return af.functionDerivative(S);
                }
            };
        }
    }

    public double[] layerOutput() {
        double []res = new double[neurons.length];
        for (int i = 0; i < neurons.length; ++i) {
            res[i] = neurons[i].neuronOutput();
        }
        return res;
    }

    public void setWeights(double [][]w) {
        for (int j = 0; j < neurons.length; ++j) {
            neurons[j].setWeights(w[j]);
        }
    }

    public void setInputs(double[] inputs) {
        for (Neuron neuron : neurons) {
            neuron.setInputs(inputs);
        }
    }

    public Neuron[] getNeurons() {
        return neurons;
    }
}
