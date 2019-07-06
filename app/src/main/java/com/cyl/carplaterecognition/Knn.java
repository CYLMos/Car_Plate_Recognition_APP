package com.cyl.carplaterecognition;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.KNearest;
import org.opencv.ml.Ml;
import org.opencv.utils.Converters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 * the class of the KNN
 */

public class Knn implements AlgorithmInterface {

    private KNearest knn;
    private Mat trainData;
    private Imgproc imgproc;
    private Context context;
    private String result;

    // new format chars
    int[] fileId = {R.drawable.n0,R.drawable.n1,R.drawable.n2,R.drawable.n3,R.drawable.n4,R.drawable.n5,R.drawable.n6,R.drawable.n7,
            R.drawable.n8,R.drawable.n9,R.drawable.a,R.drawable.b,R.drawable.c,R.drawable.d,R.drawable.e,R.drawable.f, R.drawable.g,
            R.drawable.h,R.drawable.i,R.drawable.j,R.drawable.k,R.drawable.l,R.drawable.m,R.drawable.n, R.drawable.o,R.drawable.p,
            R.drawable.q,R.drawable.r,R.drawable.s,R.drawable.t,R.drawable.u,R.drawable.v,R.drawable.w,R.drawable.x,R.drawable.y,R.drawable.z};

    // labels
    Byte charLabs[] = {'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S',
            'T','U','V','W','X','Y','Z'};

    public Knn(Context context){
        this.knn = KNearest.create();
        this.trainData = new Mat();
        this.imgproc = new Imgproc();
        this.context = context;
        this.result = "";
    }

    @Override
    public void train() {
        List<Byte> trainLabs = new ArrayList<>();
        for(int i = 0; i < 8; i++){
            List<Byte> sub = Arrays.asList(this.charLabs);
            trainLabs.addAll(sub);
        }

        // each new format chars put 3 times
        for(int j = 0; j < 3; j++){
            for(int i = 0; i < this.fileId.length; i++){
                Bitmap charBm = BitmapFactory.decodeResource(this.context.getResources(), fileId[i]);
                Mat charMat = new Mat();
                Mat dstMat = new Mat();

                Utils.bitmapToMat(charBm, charMat);
                this.imgproc.resize(charMat, charMat, new Size(50, 50));
                this.imgproc.cvtColor(charMat, dstMat, Imgproc.COLOR_RGBA2GRAY);

                this.imgproc.threshold(dstMat, dstMat, 100, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

                dstMat.convertTo(dstMat, CvType.CV_32F);
                dstMat = dstMat.reshape(1, 1);

                int a = dstMat.cols();
                int b = dstMat.rows();

                this.trainData.push_back(dstMat);
            }
        }

        // add normal chars
        Bitmap trainBM = BitmapFactory.decodeResource(this.context.getResources(), R.drawable.training_chars);
        Mat trainMat = new Mat();
        Utils.bitmapToMat(trainBM, trainMat);
        this.imgproc.cvtColor(trainMat, trainMat, Imgproc.COLOR_RGBA2GRAY);
        this.imgproc.threshold(trainMat, trainMat, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

        List<MatOfPoint> contours = new ArrayList<>();

        this.imgproc.findContours(trainMat, contours, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // sort them
        Collections.sort(contours, new Comparator<MatOfPoint>() {
            @Override
            public int compare(MatOfPoint mp1, MatOfPoint mp2) {
                Rect ra = Imgproc.boundingRect(mp1);
                Rect rb = Imgproc.boundingRect(mp2);

                if(Math.abs(ra.y - rb.y) < 50){
                    if(ra.x > rb.x){
                        return 1;
                    }
                    else{
                        return -1;
                    }
                }
                else{
                    if(ra.y > rb.y){
                        return 1;
                    }
                    else{
                        return -1;
                    }
                }
            }
        });

        // the training_chars.png must have 180 chars.
        if(contours.size() == 180) {
            for (int i = 0; i < contours.size(); i++){
                Rect rect = Imgproc.boundingRect(contours.get(i));
                Mat chMat = trainMat.submat(rect);
                this.imgproc.resize(chMat, chMat, new Size(50, 50));

                chMat.convertTo(chMat, CvType.CV_32F);
                chMat = chMat.reshape(1, 1);

                this.trainData.push_back(chMat);
            }
        }

        Mat labels = Converters.vector_char_to_Mat(trainLabs);

        labels.convertTo(labels, CvType.CV_32F);

        this.knn.train(trainData, Ml.ROW_SAMPLE, labels);
    }

    @Override
    public String getResult() {
        if(this.result != ""){
            String resultTemp = new String(this.result);
            this.result = "";

            return resultTemp;
        }

        return this.result;
    }

    @Override
    public void setImage(Bitmap bitmap) {

    }

    @Override
    public void setImage(Mat mat) {
        Mat charMat = mat;
        this.imgproc.resize(charMat, charMat, new Size(50, 50));
        charMat.convertTo(charMat, CvType.CV_32F);
        charMat = charMat.reshape(1, 1);

        int a = charMat.cols();  // for test
        int b = this.trainData.cols();  // for test

        Mat res = new Mat();
        float label = this.knn.findNearest(charMat, 7, res);
        this.result += String.valueOf( (char)((int)label) );
    }
}
