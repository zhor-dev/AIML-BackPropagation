package ActivationFunctions;

public class SigmoidFunction implements ActivationFunction {

    private double alpha = 1;
    public SigmoidFunction(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public double activationFunction(double S) {
        return 1 / (1 + Math.exp(-alpha * S));
    }

    @Override
    public double functionDerivative(double S) {
        return alpha * activationFunction(S) * (1 - activationFunction(S));
    }
}
