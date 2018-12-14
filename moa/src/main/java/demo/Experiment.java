package demo;

import java.io.IOException;

public class Experiment {

    public static void main(String[] args) throws IOException {
        ComparisonDemo exp = new ComparisonDemo();
        exp.run(100000, true,
                "C:\\Users\\angsa\\Desktop\\moa\\moa\\src\\main\\resources\\randomRBFdrift.arff", true);
    }

}
