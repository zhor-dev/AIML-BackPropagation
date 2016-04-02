package ActivationFunctions;

public class HyperbolicTangentFunction implements ActivationFunction {

    private double alpha = 1;
    public HyperbolicTangentFunction(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public double activationFunction(double S) {
        return Math.tanh(alpha * S);
    }

    @Override
    public double functionDerivative(double S) {
        return alpha / Math.pow(Math.cosh(alpha * S), 2);
    }
}
