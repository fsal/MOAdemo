package demo;

import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.Classifier;
import moa.classifiers.lazy.SAMkNN;
import moa.classifiers.lazy.kNN;
import moa.classifiers.trees.HoeffdingAdaptiveTree;
import moa.classifiers.trees.HoeffdingTree;
import moa.core.TimingUtils;
import moa.streams.generators.RandomRBFGenerator;
import moa.streams.generators.RandomRBFGeneratorDrift;
import moa.streams.generators.RandomTreeGenerator;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.pmml.jaxbbindings.DecisionTree;

import java.io.File;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;


public class ComparisonDemo{

    public ComparisonDemo() {

    }
    public void run(int numInstances, boolean isTesting, String filepath, boolean drift) {
        Path path = Paths.get("experiment56_log.txt");

        try{
            Files.createFile(path);
            List<String> lines= Arrays.asList(numInstances + " instances\n", " Drift: "+ drift+"\n");
            Files.write(path, lines, Charset.forName("UTF-8"), StandardOpenOption.APPEND);
        }catch (Exception e){
            e.printStackTrace();
        }

        RandomTreeGenerator acg = new RandomTreeGenerator();
        RandomRBFGenerator acg1 = new RandomRBFGenerator();
        acg1.instanceRandomSeedOption.setValue(2018);
        if(drift){
            acg1 = new RandomRBFGeneratorDrift();
            acg1.instanceRandomSeedOption.setValue(2018);
            ((RandomRBFGeneratorDrift) acg1).speedChangeOption.setValue(5);

        }
        acg1.prepareForUse();
        Classifier streamHAT = new HoeffdingAdaptiveTree();
        weka.classifiers.Classifier DT = new J48();
        SAMkNN streamKnn = new SAMkNN();
        IBk knn = new IBk();
        boolean preciseCPUTiming = TimingUtils.enablePreciseTiming();

        for(int i=0; i<5; i++) {
            streamHAT.setModelContext(acg1.getHeader());
            streamHAT.prepareForUse();
            streamKnn.setModelContext(acg1.getHeader());
            //streamKnn.prepareForUse();
            int numberSamplesCorrect = 0;
            int numberSamples = 0;
            long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
            while (acg1.hasMoreInstances() && numberSamples < numInstances) {
                Instance trainInst = acg1.nextInstance().getData();
                if (isTesting) {
                    if (streamKnn.correctlyClassifies(trainInst)) {
                        numberSamplesCorrect++;
                    }
                }
                numberSamples++;
                streamKnn.trainOnInstance(trainInst);
            }
            double accuracy = 100.0 * (double) numberSamplesCorrect / (double) numberSamples;
            double time = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - evaluateStartTime);
            streamKnn.resetLearningImpl();
            List<String> line = Collections.singletonList(numberSamples + " instances processed with " + accuracy + "% accuracy in " + time + " seconds.\n");
            System.out.println(line);
            try {
                Files.write(path, line, Charset.forName("UTF-8"), StandardOpenOption.APPEND);
            }catch (Exception e){
                e.printStackTrace();
            }

        }

        try{
            Files.write(path, Collections.singletonList("\n"), Charset.forName("UTF-8"), StandardOpenOption.APPEND);
            ArffLoader loader = new ArffLoader();
            loader.setFile(new File(filepath));
            Instances instances = loader.getDataSet();
            instances.setClassIndex(instances.numAttributes() -1);
            //DT.setWindowSize(0);
            knn.setKNN(10);
            //ArffLoader tloader = new ArffLoader();
            //tloader.setFile(new File("C:\\Users\\angsa\\Desktop\\moa\\moa\\src\\main\\resources\\randomRBFtest.arff"));
            //Instances test = tloader.getDataSet();
            //test.setClassIndex(test.numAttributes() -1);
            for(int i=0; i<5; i++) {
                knn.setKNN(10);
                knn.setWindowSize(0);
                Random rand = new Random(2018);
                int tt = rand.nextInt(10);
                Instances train = instances.testCV(10, 1);
                Instances test = instances.testCV(10, 9);
                long StartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
                knn.buildClassifier(train);
                Evaluation eval = new Evaluation(train);
                System.out.println("training ended");
                eval.evaluateModel(knn, test);
                System.out.println(eval.toSummaryString("\nResults\n======\n", false));
                double t = TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread() - StartTime);
                double tot = eval.confusionMatrix()[0][0]+eval.confusionMatrix()[1][1] + eval.confusionMatrix()[0][1]+eval.confusionMatrix()[1][0];
                double acc = 100 * (eval.confusionMatrix()[0][0]+eval.confusionMatrix()[1][1])/tot;
                String str = train.numInstances() + " instances processed with " + acc + "% accuracy in " + t + " seconds.\n";
                System.out.println(str);
                Files.write(path, Collections.singletonList(str), Charset.forName("UTF-8"), StandardOpenOption.APPEND);
            }
            Files.write(path, Collections.singletonList("\n\n"), Charset.forName("UTF-8"), StandardOpenOption.APPEND);
        }
        catch(Exception e){
            System.out.println("EXCEPTION");
            e.printStackTrace();
        }

    }


}

