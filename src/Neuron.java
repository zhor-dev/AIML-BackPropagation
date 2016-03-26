public abstract class Neuron implements ActivationFunction {

    private double []inputs;
    private double []weights;

    public Neuron(double []i) {
        this.inputs = i;
        weights = new double[inputs.length];
        for (int j = 0; j < inputs.length; ++j) {
            int sign = Math.random() < 0.5 ? -1 : 1;
            weights[j] = sign * Math.random();
        }
    }

    public void setInputs(double []i) {
        this.inputs = i;
    }

    public void setWeights(double []w) {
        this.weights = w;
    }

    public double[] getInputs() {
        return inputs;
    }

    public double[] getWeights() {
        return weights;
    }

    public double summingBlock() {
        double sum = 0;
        for (int i = 0; i < inputs.length; ++i) {
            sum += inputs[i] * weights[i];
        }
        return sum;
    }

    public double neuronOutput() {
        return activationFunction(summingBlock());
    }

}
