package ActivationFunctions;

public class GaussianFunction implements ActivationFunction {

    private double variation;
    public GaussianFunction(double variation) {
        this.variation = variation;
    }

    @Override
    public double activationFunction(double S) {
        return Math.exp(-(S * S) / variation);
    }

    @Override
    public double functionDerivative(double S) {
        return 0;
    }
}
