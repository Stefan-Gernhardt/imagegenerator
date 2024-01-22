package org.deeplearning;

public class EvalResult {
    int label = 0;

    double value = 0.0;

    public EvalResult(int l, double v) {
        label = l;
        value = v;
    }
}
