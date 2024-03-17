package org.deeplearning;

public class EvalResult {
    public int label = 0;

    public double value = 0.0;

    public EvalResult(int l, double v) {
        label = l;
        value = v;
    }
}
