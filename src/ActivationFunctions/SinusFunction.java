package ActivationFunctions;

public class SinusFunction implements ActivationFunction {

    private double alpha = 1;
    public SinusFunction(double alpha) {
        this.alpha = alpha;
    }

    @Override
    public double activationFunction(double S) {
        return Math.sin(alpha * S);
    }

    @Override
    public double functionDerivative(double S) {
        return alpha * Math.cos(alpha * S);
    }
}
